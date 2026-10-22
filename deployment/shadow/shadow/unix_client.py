"""HTTP over a protected Unix socket; no TCP, proxies or paid-provider calls."""
import http.client,json,socket


class UnixConnection(http.client.HTTPConnection):
    def __init__(self,path,timeout=.25):super().__init__('localhost',timeout=timeout);self.path=str(path)
    def connect(self):
        self.sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);self.sock.settimeout(self.timeout);self.sock.connect(self.path)


class UnixScoreClient:
    def __init__(self,path,timeout=.25):self.path=path;self.timeout=timeout
    def request(self,method,path,payload=None):
        c=UnixConnection(self.path,self.timeout)
        try:
            body=json.dumps(payload,allow_nan=False).encode() if payload is not None else None
            c.request(method,path,body,{'Content-Type':'application/json'} if body else {})
            r=c.getresponse();data=r.read(131073)
            if len(data)>131072 or r.status!=200:raise RuntimeError('SHADOW_HTTP_UNAVAILABLE')
            result=json.loads(data)
            if not isinstance(result,dict):raise RuntimeError('SHADOW_BAD_RESPONSE')
            return result
        finally:c.close()
    def score(self,payload):return self.request('POST','/v1/score',payload)
    def ready(self):return self.request('GET','/health/ready')
