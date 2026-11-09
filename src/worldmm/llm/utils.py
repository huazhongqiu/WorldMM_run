import asyncio
import functools
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type

def dynamic_retry_decorator(func):
    """Decorator that applies retry logic with exponential backoff."""
    @functools.wraps(func)
    def sync_wrapper(self, *args, **kwargs):
        from .lmdeploy import StructuredOutputTruncatedError
        max_retries = getattr(self, 'max_retries', 5)
        decorated_func = retry(
            retry=retry_if_not_exception_type(StructuredOutputTruncatedError),
            stop=stop_after_attempt(max_retries), 
            wait=wait_exponential(multiplier=1, min=1, max=10)
        )(func)
        return decorated_func(self, *args, **kwargs)
        
    async def async_wrapper(self, *args, **kwargs):
        from .lmdeploy import StructuredOutputTruncatedError
        max_retries = getattr(self, 'max_retries', 5)
        decorated_func = retry(
            retry=retry_if_not_exception_type(StructuredOutputTruncatedError),
            stop=stop_after_attempt(max_retries), 
            wait=wait_exponential(multiplier=1, min=1, max=10)
        )(func)
        return await decorated_func(self, *args, **kwargs)
        
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper