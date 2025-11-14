import logging
import time
from functools import wraps

logger = logging.getLogger("persian_asr")

def asr_logged(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        logger.info("ASR started: %s", func.__name__)
        try:
            out = func(*args, **kwargs)
            return out
        finally:
            dt = (time.perf_counter() - t0) * 1000
            logger.info("ASR finished in %.1f ms", dt)
    return wrapper
