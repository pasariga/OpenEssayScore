"""Offline CLI. Never prints input text in diagnostic messages."""
import argparse,json,sys
from pathlib import Path
from . import Scorer,ServiceError
from .service.registry import register_v1,ModelLoadError,VERSION


def main():
    p=argparse.ArgumentParser(prog='openessayscore')
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('init-v1');q.add_argument('--model-dir',type=Path,required=True);q.add_argument('--store',type=Path,required=True)
    q=sub.add_parser('score');q.add_argument('--store',type=Path,required=True);q.add_argument('--text-file',type=Path,required=True)
    q.add_argument('--prompt',required=True);q.add_argument('--model-version');q.add_argument('--review-bundle',type=Path)
    q.add_argument('--include-review',action='store_true')
    q=sub.add_parser('status');q.add_argument('--store',type=Path,required=True)
    a=p.parse_args()
    try:
        if a.command=='init-v1':
            register_v1(a.store,a.model_dir);result={'registered':VERSION}
        elif a.command=='status':result=Scorer(a.store).status()
        else:
            with a.text_file.open('rb') as f:data=f.read(131073)
            if len(data)>131072:raise ServiceError('INPUT_FILE_TOO_LARGE',413)
            result=Scorer(a.store,review_bundle=a.review_bundle).score(data.decode('utf-8-sig'),
                writing_prompt=a.prompt,model_version=a.model_version,include_review=a.include_review)
        print(json.dumps(result,ensure_ascii=False));return 0
    except (ServiceError,ModelLoadError,OSError,UnicodeError) as exc:
        code=exc.code if isinstance(exc,ServiceError) else (str(exc) if isinstance(exc,ModelLoadError) else 'INPUT_OR_ARTIFACT_UNAVAILABLE')
        print(json.dumps({'error':{'code':code}}),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
