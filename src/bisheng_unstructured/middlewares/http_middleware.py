# Define a custom middleware class
from time import time
from uuid import uuid4

from fastapi import Request
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from bisheng_unstructured.common.citic_log import citic_logger, system_tag

class CustomMiddleware(BaseHTTPMiddleware):
    """切面程序"""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        # You can modify the request before passing it to the next middleware or endpoint
        trace_id = str(uuid4().hex)
        start_time = time()
        with logger.contextualize(trace_id=trace_id):
            #开始
            """
            ip_address:通过request能拿到IP地址，数组形式，一般为request.META['REMOTE_ADDR']
            transaction_no:交易序号，全局自增，可以参考使用redis的incr方法得到
            system_tag:系统标识，例如'FQWS'
            api_name:api名称
            """
            uuid_val = str(uuid.uuid1())
            # ip_address = get_remote_ip(request).split('.')
            ip_address= request.remote_addr
            uuid_val , serial_no , process , tran_id = citic_logger.proc_start_log(system_tag , '' , request.url.path , ip_address)

            response = await call_next(request)
            process_time = round(time() - start_time, 2)
            logger.info(f'{request.url.path} {response.status_code} timecost={process_time}')

            #结束
            citic_logger.proc_end_log(system_tag,uuid_val,serial_no,process , tran_id , error_code=0,message=response.status_code)

            return response
