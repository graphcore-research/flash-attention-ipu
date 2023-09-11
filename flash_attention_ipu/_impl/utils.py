from functools import wraps
from typing import Any, Callable, List
import logging

logger = logging.getLogger(__name__)


def patch_function(orig_fn: Callable, modules_to_patch: List[Any]):
    """Decorator util helping patching any Python function.
    A simple example of patching a numpy function:
    
    @patch_function(numpy.sin, [numpy])
    def noisy_sin(orig_sin, x):
        print('sining!')
        return orig_sin(x)
    
    Note that the first argument has to be the original function being patched, to
    avoid some kind of circular dependency in the implementation.
    Args:
        orig_fn: Original function to patch.
        modules_to_patch: Python modules to update with the patched function.
    Returns:
        Patching decorator, taking as first argument the original function.
    """

    def decorator_patch_fn(patched_fn: Callable):
        @wraps(orig_fn)
        def patch_wrapper(*args, **kwargs):
            return patched_fn(orig_fn, *args, **kwargs)

        fn_name = orig_fn.__name__
        for m in modules_to_patch:
            logger.info(f"flash_attention_ipu: patching {orig_fn.__name__}!")
            setattr(m, fn_name, patch_wrapper)
        return patch_wrapper

    return decorator_patch_fn
