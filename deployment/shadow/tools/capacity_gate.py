"""Conservative capacity screen from bounded audit data; not a peak forecast."""
import argparse,json
from pathlib import Path
MIB=1024**2


def assess(audit,backend_current_mib,shadow_mib=512,reserve_mib=256):
    available=audit['summary']['minimum_available_memory_bytes']/MIB
    backend=next((x for x in audit['container_configuration'] if x.get('compose_service')=='backend'),None)
    if not backend or not backend.get('memory_limit_bytes'):
        return {'passed':False,'reason':'backend_limit_unknown_or_unbounded'}
    growth=max(0,backend['memory_limit_bytes']/MIB-backend_current_mib)
    required=shadow_mib+reserve_mib+growth
    return {'passed':available>=required,'observed_minimum_available_mib':available,
      'backend_measured_mib':backend_current_mib,'backend_limit_mib':backend['memory_limit_bytes']/MIB,
      'backend_additional_growth_envelope_mib':growth,'shadow_limit_mib':shadow_mib,'host_safety_reserve_mib':reserve_mib,
      'required_available_mib':required,'margin_mib':available-required,'additional_available_mib_needed':max(0,required-available),
      'scope':'conservative concurrent-limit envelope; limits are not reserved RAM; unobserved growth of DB/OS is not modeled',
      'cpu_and_disk_require_separate_review':True,'peak_observation_complete':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('audit',type=Path);p.add_argument('--backend-current-mib',type=float,required=True);p.add_argument('--output',type=Path)
    a=p.parse_args();result=assess(json.loads(a.audit.read_text(encoding='utf-8-sig')),a.backend_current_mib)
    if a.output:a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2));return 0 if result['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
