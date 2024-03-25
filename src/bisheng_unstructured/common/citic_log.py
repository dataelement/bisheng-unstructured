#在主模块或者专门的配置模块创建实例
import os
import traceback
import threading
import datetime
from citic_logger import CiticLogger

#系统标识
system_tag = ''

def setup_custom_logger():
    logger = CiticLogger(name='MyAppLogger')
    HOSTNAME = os.environ.get('HOSTNAME')
    #日志目录
    log_dir = 'logs'
    logger.add_handler('proc_handler' , log_dir , f'proc-v01-{system_tag}-{HOSTNAME}.log')
    logger.add_handler('err_handler' , log_dir , f'err-v01-{system_tag}-{HOSTNAME}.log')

    return logger

#配置并获取CustomLogger实例
citic_logger = setup_custom_logger()

def citic_logger_info(message , uuid_val, serial_no):
    #获取当前时间
    now = datetime.datetime.now()
    current_time = now.strftime('%Y-%m-%d %H:%M:%S.%f')
    #获取进程号和线程号
    process = os.getpid()
    thread = threading.get_ident()
    #记录开始日志
    info_message = f"[{current_time}]|INFO|{system_tag}|{uuid_val}|{serial_no}|{process}|{thread}||| {message}"
    citic_logger.info(info_message)
    


def citic_logger_error(error_message):
    #获取当前时间
    now = datetime.datetime.now()
    current_time = now.strftime('%Y-%m-%d %H:%M:%S.%f')
    #获取进程号和线程号
    process = os.getpid()
    thread = threading.get_ident()
    ERR = ''
    file,line,_,_ = traceback.extract_tb(e.__trackback__)[-1]
    POS = f"{file},line{line}"#发送报错代码位置
    message = f"[{current_time}]|ERROR|{system_tag}|{uuid_val}||{process}|{thread}|||#EX_ERR:POS={POS},ERR={ERR},EMSG={error_message}"
    
    citic_logger.error(message)