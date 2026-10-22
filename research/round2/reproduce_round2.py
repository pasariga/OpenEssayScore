"""Replay the fixed training-only protocol under a new experiment identifier."""
import argparse,os,re,sys
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--run-id',required=True)
args=parser.parse_args()
if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{1,79}',args.run_id):
    parser.error('Use a short safe experiment identifier')
if args.run_id=='round2-audit-cv-v1':parser.error('Original result is preserved; choose a new run-id')
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
import cv_research as experiment
experiment.RUN=args.run_id
experiment.OUT=experiment.ROOT/'reports'/args.run_id
experiment.CKPT=experiment.ROOT/'checkpoints'/args.run_id
print('Reproduction of an already inspected internal benchmark; not new independent test evidence.',flush=True)
experiment.main()
