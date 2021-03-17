from setuptools import find_packages, setup
import torch

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 3], "Requires PyTorch >= 1.3"

setup(
    name="yolof",
    version="0.1.0",
    author="Chensnathan",
    url="https://github.com/chensnathan/YOLOF",
    description="Code for YOLOF.",
    packages=find_packages(exclude=("configs", "datasets")),
    python_requires=">=3.6"
)
