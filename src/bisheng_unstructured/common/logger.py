import logging
import os

logger_initialized = {}
global_log_file = []


def get_logger(name, log_file=None, log_level=logging.INFO):
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.

    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    # stream_handler = logging.StreamHandler()
    # handlers = [stream_handler]
    handlers = []

    rank = 0

    if log_file is not None and len(global_log_file) == 0:
        log_path = os.path.dirname(log_file)
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        global_log_file.append(log_file)

    if log_file is None and len(global_log_file) > 0:
        log_file = global_log_file[0]

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, "a")
        handlers.append(file_handler)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True

    return logger


def PCHECK(expr, msg, *args):
    if not expr:
        get_logger("main").error(msg, *args)
