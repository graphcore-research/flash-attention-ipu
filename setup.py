import subprocess
from pathlib import Path

import setuptools
import setuptools.command.build_ext


class make_ext(setuptools.command.build_ext.build_ext):  # type:ignore[misc]
    def build_extension(self, ext: setuptools.Extension) -> None:
        if ext.name == "libflash_attention":
            print(self.build_lib)
            subprocess.check_call(["ninja"])
        else:
            super().build_extension(ext)


setuptools.setup(
    name="flash-attention-ipu",
    version="0.1.0",
    description="FlashAttention for Graphcore IPUs",
    packages=["flash_attention_ipu"],
    install_requires=Path("requirements.txt").read_text().rstrip("\n").split("\n"),
    ext_modules=[
        setuptools.Extension(
            "libflash_attention",
            list(map(str, Path("flash_attention_ipu/cpp").glob("*.[ch]pp"))),
        )
    ],
    cmdclass=dict(build_ext=make_ext),
)
