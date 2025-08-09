import asyncio
import random
from functools import wraps
from typing import TypeVar, Callable, Any
from app.utils.logging import logger

T = TypeVar('T')

def retry_with_exponential_backoff(
    max_retries: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True
):
    """
    Decorator for retrying async functions with exponential backoff on rate limit errors.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff calculation
        jitter: Add random jitter to prevent thundering herd
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)

                except Exception as e:
                    error_str = str(e)

                    # Check if it's a rate limit error
                    if "429" in error_str or "rate" in error_str.lower():
                        last_exception = e

                        # Calculate delay with exponential backoff
                        delay = min(initial_delay * (exponential_base ** attempt), max_delay)

                        # Add jitter if enabled
                        if jitter:
                            delay = delay * (0.5 + random.random())

                        logger.warning(
                            f"Rate limit hit in {func.__name__}. "
                            f"Retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})"
                        )

                        await asyncio.sleep(delay)
                        continue

                    # Not a rate limit error, re-raise immediately
                    raise

            # All retries exhausted
            logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}")
            raise last_exception

        return wrapper
    return decorator