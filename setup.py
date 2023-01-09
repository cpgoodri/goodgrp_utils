import setuptools

INSTALL_REQUIRES = ["numpy", "jax>=0.2.10", "jaxlib>=0.1.62", "jax-md>=0.1.13"]

setuptools.setup(
    name="goodgrp_utils",
    version="0.0.1",
    license="Apache 2.0",
    install_requires=INSTALL_REQUIRES,
    packages=setuptools.find_packages(),
    description="Common utils for working with jax-md.",
    python_requires=">=3.7",
)
