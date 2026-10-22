import argparse,json
from unix_client import UnixScoreClient
from dispatcher import VERSION

def main():
    p=argparse.ArgumentParser();p.add_argument('--socket',required=True);a=p.parse_args()
    try:
        r=UnixScoreClient(a.socket,timeout=1).ready()
        ok=r.get('ready') is True and r.get('active_model')==VERSION and r.get('review_status')=='disabled'
    except Exception:ok=False
    print(json.dumps({'ready':ok,'expected_model':VERSION}));return 0 if ok else 1

if __name__=='__main__':raise SystemExit(main())
