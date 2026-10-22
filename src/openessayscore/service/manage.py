import argparse,json
from pathlib import Path
from .registry import ModelRegistry,register_v1,VERSION
p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
s=sub.add_parser('init-v1');s.add_argument('--store',type=Path,required=True);s.add_argument('--source',type=Path,required=True)
s=sub.add_parser('activate');s.add_argument('--store',type=Path,required=True);s.add_argument('--version',required=True)
a=p.parse_args()
if a.command=='init-v1':register_v1(a.store,a.source);print(json.dumps({'registered':VERSION}))
else:print(json.dumps({'previous':ModelRegistry(a.store).activate(a.version),'active':a.version}))
