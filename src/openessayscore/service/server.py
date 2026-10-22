import argparse,os
from pathlib import Path
from .core import ScoreService
from .api import create_app

def main():
    p=argparse.ArgumentParser();p.add_argument('--store',type=Path,required=True);p.add_argument('--review-bundle',type=Path)
    p.add_argument('--journal',type=Path);p.add_argument('--port',type=int,default=8765)
    p.add_argument('--max-inflight',type=int,default=4);p.add_argument('--memory-budget-mb',type=int)
    a=p.parse_args()
    import uvicorn
    service=ScoreService(a.store,a.review_bundle,a.journal,a.max_inflight,a.memory_budget_mb)
    uvicorn.run(create_app(service),host='127.0.0.1',port=a.port,workers=1,limit_concurrency=32,access_log=False,log_level='warning')

if __name__=='__main__':main()
