import hashlib,json,os,re,shutil,threading
from pathlib import Path
import joblib

V1_WEIGHT_SHA='c28c5010ae4f1170aba860ecbe9f2393f5a723311c59f89db5ebbc2f773193cb'
V1_VECTOR_SHA='e098d09fb07b81320e87a1c2e96cba1387525f64ca93414c279d66a0070a667b'
RUBRIC='CFMS34-holistic-sum-0to10'
VERSION='wenheng-score-v1-20261022'
PREPROCESS='v1-nfkc-redact-email-mobile-1'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def valid_id(value):return isinstance(value,str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',value) is not None
def atomic_json(path,value):
    path=Path(path);temp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with temp.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

class ModelLoadError(Exception):pass

class ModelRegistry:
    def __init__(self,root):self.root=Path(root).resolve();self.lock=threading.RLock();self.cache={}
    def snapshot(self,version=None):
        try:
            with (self.root/'registry.json').open() as f:registry=json.load(f)
            version=version or registry['active']
            if not valid_id(version) or version not in registry['versions']:raise ModelLoadError('MODEL_VERSION_NOT_REGISTERED')
            record=registry['versions'][version];folder=(self.root/'versions'/version).resolve()
            if not folder.is_relative_to(self.root/'versions'):raise ModelLoadError('MODEL_PATH_INVALID')
            if digest(folder/'manifest.json')!=record['manifest_sha256']:raise ModelLoadError('MODEL_MANIFEST_MISMATCH')
            manifest=json.loads((folder/'manifest.json').read_text())
            if manifest['version']!=version:raise ModelLoadError('MODEL_VERSION_MISMATCH')
            return version,folder,manifest,record['manifest_sha256']
        except ModelLoadError:raise
        except (OSError,ValueError,KeyError,TypeError) as exc:raise ModelLoadError('MODEL_REGISTRY_UNAVAILABLE') from exc
    def load(self,version=None):
        with self.lock:
            version,folder,manifest,manifest_sha=self.snapshot(version)
            if version in self.cache:
                if self.cache[version]['manifest_sha256']!=manifest_sha:raise ModelLoadError('IMMUTABLE_VERSION_CHANGED')
                return self.cache[version]
            try:
                if manifest['family']!='character_tfidf_ridge' or manifest['rubric_id']!=RUBRIC or manifest['preprocessing']!=PREPROCESS:raise ModelLoadError('UNSUPPORTED_MODEL_CONTRACT')
                expected={'best_scorer.joblib':V1_WEIGHT_SHA,'lexical_vectorizer.joblib':V1_VECTOR_SHA}
                for name,sha in expected.items():
                    if manifest['files'].get(name)!=sha or digest(folder/name)!=sha:raise ModelLoadError('MODEL_WEIGHT_HASH_MISMATCH')
                loaded={'version':version,'manifest':manifest,'manifest_sha256':manifest_sha,
                        'scorer':joblib.load(folder/'best_scorer.joblib'),
                        'vectorizer':joblib.load(folder/'lexical_vectorizer.joblib')}
                if len(self.cache)>=2:self.cache.pop(next(iter(self.cache)))
                self.cache[version]=loaded
                return loaded
            except ModelLoadError:raise
            except MemoryError:raise
            except Exception as exc:raise ModelLoadError('MODEL_LOAD_FAILED') from exc
    def activate(self,version):
        # Validate/load before changing the pointer. No deletion or in-place weight replacement.
        self.load(version)
        with self.lock:
            registry=json.loads((self.root/'registry.json').read_text());old=registry['active'];registry['active']=version
            atomic_json(self.root/'registry.json',registry)
            with (self.root/'activation_log.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'old':old,'active':version,'operation':'operator_activation'})+'\n')
        return old

def register_v1(root,source,version=VERSION,engineering_fixture=False):
    if not valid_id(version):raise ValueError('Unsafe version ID')
    root=Path(root);source=Path(source);folder=root/'versions'/version
    expected={'best_scorer.joblib':V1_WEIGHT_SHA,'lexical_vectorizer.joblib':V1_VECTOR_SHA}
    for name,sha in expected.items():
        if digest(source/name)!=sha:raise ModelLoadError('SOURCE_V1_HASH_MISMATCH')
    # Existing releases are immutable; exact reruns can be verified but not overwritten.
    if folder.exists():
        for name,sha in expected.items():
            if digest(folder/name)!=sha:raise ModelLoadError('EXISTING_VERSION_DIFFERS')
        return ModelRegistry(root).load(version)
    folder.mkdir(parents=True)
    for name in expected:shutil.copy2(source/name,folder/name)
    manifest={'version':version,'family':'character_tfidf_ridge','rubric_id':RUBRIC,'score_range':[0,10],
        'preprocessing':PREPROCESS,'files':expected,'status':'research_baseline',
        'engineering_fixture_same_v1_weights':engineering_fixture,'data_license':'CC-BY-4.0',
        'capabilities':{'overall_score':True,'dimension_scores':False,'teacher_level_verified':False,'prompt_relevance':False}}
    (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    path=root/'registry.json'
    registry=json.loads(path.read_text()) if path.exists() else {'active':version,'versions':{}}
    registry['versions'][version]={'manifest_sha256':digest(folder/'manifest.json')}
    atomic_json(path,registry)
    return ModelRegistry(root).load(version)
