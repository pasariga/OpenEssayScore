"""Offline sidecar contract. Does not call providers, map scales or write a DB."""
import copy,math,json
from pathlib import Path
from jsonschema import Draft202012Validator
PEER_SCHEMA=json.loads((Path(__file__).parent/'peer_contract_schema.json').read_text(encoding='utf-8'))
PEER_VALIDATOR=Draft202012Validator(PEER_SCHEMA)

PEERS=['provider_a','provider_b','provider_c']
DIMENSIONS={'content_and_ideas','structure_and_coherence','language_and_expression','writing_conventions'}
class MappingUnavailable(Exception):pass

def map_native_score(*args,**kwargs):
    raise MappingUnavailable('No validated CFMS10 -> example-four-traits-100-v1/100 mapping exists')

def validate_peer(envelope):
    if envelope.get('scoring_standard_version')!='example-four-traits-100-v1':return False
    value=envelope.get('result')
    if not isinstance(value,dict):return False
    if not PEER_VALIDATOR.is_valid(value):return False
    try:
        scores=value['dimension_scores'];overall=value['overall_score']
        if len(scores)!=4 or {d['dimension'] for d in scores}!=DIMENSIONS:return False
        if not all(d['max_score']==25 and isinstance(d['score'],(float,int)) and not isinstance(d['score'],bool) and math.isfinite(d['score']) and 0<=d['score']<=25 for d in scores):return False
        if isinstance(overall,bool) or not isinstance(overall,(int,float)) or not math.isfinite(overall):return False
        if not math.isclose(sum(d['score'] for d in scores),overall,abs_tol=.01):return False
        if not value.get('summary'):return False
        return True
    except (KeyError,TypeError,ValueError):return False

def integrate_offline(local_call,peer_calls,existing_final=None):
    # All callables are supplied by caller. Tests use marked technical stubs only.
    peers=[];issues=[]
    for name in PEERS:
        try:
            envelope=peer_calls[name]()
            if not validate_peer(envelope):raise ValueError('peer contract invalid')
            peers.append({'provider':name,'status':'available','envelope':copy.deepcopy(envelope)})
        except TimeoutError:issues.append(name+':TIMEOUT')
        except Exception:issues.append(name+':UNAVAILABLE_OR_INVALID')
    try:
        value=local_call()
        if value.get('contract_version')!='openessayscore/1.0':raise ValueError('local contract invalid')
        sidecar={'status':'available','native_result':value,'mapped_score':None,'aggregation_weight':0,
             'eligible_for_legacy_aggregation':False,'reasons':['RUBRIC_NOT_EQUIVALENT','NO_DIMENSION_SCORES','NO_VALIDATED_SCORE_MAPPING']}
    except Exception:
        sidecar={'status':'unavailable','native_result':None,'mapped_score':None,'aggregation_weight':0,
             'eligible_for_legacy_aggregation':False,'reasons':['LOCAL_SCORE_UNAVAILABLE']}
    if len(peers)>=2:
        scores=[x['envelope']['result']['overall_score'] for x in peers]
        if max(scores)-min(scores)>=20:issues.append('PEER_SCORE_DISAGREEMENT_REQUIRES_REVIEW')
    if not peers:issues.append('NO_COMPARABLE_PROVIDER_RESULTS')
    return {'mode':'offline_contract_prototype','peers':peers,'native_sidecar':sidecar,
       'legacy_final':copy.deepcopy(existing_final),'legacy_final_changed':False,
       'new_combined_score':None,'teacher_review_required':True,'issues':issues,
       'paid_api_calls':0,'confidence':None}
