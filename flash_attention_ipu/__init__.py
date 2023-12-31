# Copyright (c) 2023 Graphcore Ltd. All rights reserved.


def _load_native_library() -> None:
    import ctypes
    import pathlib
    import sysconfig

    root = pathlib.Path(__file__).parent.parent.absolute()
    name = "libflash_attention.so"

    paths = [
        root / "build" / name,
        (root / name).with_suffix(sysconfig.get_config_vars()["SO"]),
    ]
    for path in paths:
        if path.exists():
            ctypes.cdll.LoadLibrary(str(path))
            return
    raise ImportError(f"Cannot find {name} - tried {[str(p) for p in paths]}")


_load_native_library()

from ._impl.flash_attention_qkv_packed import *
from ._impl.flash_attention_qkv_packed import __all__
