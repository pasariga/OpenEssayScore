import argparse,os,re
parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True)
a=parser.parse_args()
if not re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_-]{1,79}',a.run_id) or a.run_id=='round4-trust-review-v1':parser.error('Use a safe new run ID')
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import experiment
experiment.RUN=a.run_id;experiment.OUT=experiment.ROOT/'reports'/a.run_id;experiment.CKPT=experiment.ROOT/'checkpoints'/a.run_id
experiment.main()
