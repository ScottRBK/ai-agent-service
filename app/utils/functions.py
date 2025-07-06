def get_fully_qualified_name(func):
    return f"{func.__module__}.{func.__qualname__}"