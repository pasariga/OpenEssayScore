"""Controlled offline replay, stdin only. Never queries production DB or providers."""
import argparse,collections,json,sys
from pathlib import Path
from dispatcher import ShadowDispatcher,Journal,Permission
from unix_client import UnixScoreClient


def main():
    p=argparse.ArgumentParser();p.add_argument('--policy',type=Path,required=True);p.add_argument('--authorizations',type=Path,required=True)
    p.add_argument('--socket',required=True);p.add_argument('--results',type=Path,required=True);a=p.parse_args()
    policy=json.loads(a.policy.read_text());records=json.loads(a.authorizations.read_text()).get('records',{})
    dispatcher=ShadowDispatcher(policy,UnixScoreClient(a.socket).score,Journal(a.results,policy['result_max_bytes']))
    counts=collections.Counter()
    try:
        while True:
            line=sys.stdin.buffer.readline(131073)
            if not line:break
            if len(line)>131072:
                while line and not line.endswith(b'\n'):line=sys.stdin.buffer.readline(131073)
                counts['oversize']+=1;continue
            try:
                job=json.loads(line)
                if not isinstance(job,dict):raise ValueError('Invalid envelope')
                ref=job.pop('authorization_ref');record=records[ref]
                # Registry schema is trusted/curated, but all fields are still checked.
                permission=Permission(**record)
            except (ValueError,KeyError,TypeError):counts['invalid_or_unauthorized']+=1;continue
            counts[dispatcher.submit(job,permission)]+=1
    finally:dispatcher.close()
    print(json.dumps({'admission':dict(counts),'worker':dict(dispatcher.counts),'primary_result_changed':False,'paid_api_calls':0}))


if __name__=='__main__':main()
