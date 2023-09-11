# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import subprocess
from pathlib import Path

import setuptools
import setuptools.command.build_ext


class make_ext(setuptools.command.build_ext.build_ext):  # type:ignore[misc]
    def build_extension(self, ext: setuptools.Extension) -> None:
        if ext.name == "libflash_attention":
            filename = Path(self.build_lib) / self.get_ext_filename(
                self.get_ext_fullname(ext.name)
            )
            objdir = filename.with_suffix("")
            subprocess.check_call(
                [
                    "make",
                    f"OUT={filename}",
                    f"OBJDIR={objdir}",
                ]
            )
        else:
            super().build_extension(ext)


setuptools.setup(
    name="flash-attention-ipu",
    version="0.1.0",
    description="FlashAttention for Graphcore IPUs",
    install_requires=Path("requirements.txt").read_text().rstrip("\n").split("\n"),
    ext_modules=[
        setuptools.Extension(
            "libflash_attention",
            list(map(str, Path("flash_attention_ipu/cpp").glob("*.[ch]pp"))),
        )
    ],
    cmdclass=dict(build_ext=make_ext),
)
