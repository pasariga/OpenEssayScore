"""Small HTTP client usable from Flask/FastAPI without importing model libraries."""
import json,urllib.request,urllib.error,urllib.parse

class ScoreClientError(Exception):
    def __init__(self,code,status=None):self.code=code;self.status=status;super().__init__(code)

class ScoreClient:
    def __init__(self,base_url='http://127.0.0.1:8765',timeout=3.):
        parsed=urllib.parse.urlparse(base_url)
        if parsed.scheme!='http' or parsed.hostname not in ['127.0.0.1','localhost','::1'] or parsed.username or parsed.password:
            raise ValueError('Prototype client only permits loopback HTTP without credentials')
        self.base_url=base_url.rstrip('/');self.timeout=timeout
    def score(self,payload):
        request=urllib.request.Request(self.base_url+'/v1/score',data=json.dumps(payload,ensure_ascii=False,allow_nan=False).encode(),headers={'Content-Type':'application/json'},method='POST')
        try:
            with urllib.request.urlopen(request,timeout=self.timeout) as response:return json.load(response)
        except urllib.error.HTTPError as exc:
            try:code=json.load(exc)['error']['code']
            except (ValueError,KeyError):code='REMOTE_SERVICE_ERROR'
            raise ScoreClientError(code,exc.code) from None
        except (urllib.error.URLError,TimeoutError):raise ScoreClientError('LOCAL_SCORE_SERVICE_UNAVAILABLE') from None
