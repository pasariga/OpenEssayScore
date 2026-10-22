import importlib.util,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from docker_hardlimit import validate_limit

def test_hard_limit_requires_both_docker_and_kernel_evidence():
    validate_limit(512*1024**2,512*1024**2)
    for configured,kernel in [(0,512*1024**2),(512*1024**2,None),(512*1024**2,2**60)]:
        with pytest.raises(ValueError):validate_limit(configured,kernel)
