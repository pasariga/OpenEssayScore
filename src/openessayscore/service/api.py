import json,threading
from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse
from .core import ScoreService,ServiceError

MAX_BODY=131072
def create_app(service,max_http_inflight=8):
    app=FastAPI(title='OpenEssayScore CPU API',version='0.1',docs_url=None,redoc_url=None)
    admission=threading.BoundedSemaphore(max_http_inflight)
    @app.exception_handler(ServiceError)
    async def failure(request,exc):return JSONResponse({'error':{'code':exc.code},'retryable':exc.status in [429,503,504]},status_code=exc.status)
    @app.get('/health/live')
    def live():return {'live':True}
    @app.get('/health/ready')
    def ready():
        status=service.status();return JSONResponse(status,status_code=200 if status['ready'] else 503)
    @app.get('/v1/models')
    def models():return service.status()
    @app.post('/v1/score')
    async def score(request:Request):
        if not admission.acquire(blocking=False):raise ServiceError('SERVICE_BUSY',429)
        try:
            if request.headers.get('content-type','').split(';')[0]!='application/json':raise ServiceError('JSON_CONTENT_TYPE_REQUIRED',415)
            body=bytearray()
            async for chunk in request.stream():
                if len(body)+len(chunk)>MAX_BODY:raise ServiceError('REQUEST_TOO_LARGE',413)
                body.extend(chunk)
            try:
                def invalid_constant(value):raise ValueError('Non-finite JSON')
                data=json.loads(body,parse_constant=invalid_constant)
            except (ValueError,UnicodeError,RecursionError):raise ServiceError('INVALID_JSON',400)
            from starlette.concurrency import run_in_threadpool
            return await run_in_threadpool(service.evaluate,data)
        finally:admission.release()
    return app
