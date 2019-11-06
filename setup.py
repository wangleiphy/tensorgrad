import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tensorgrad",
    version="0.0.1",
    author="Lei Wang",
    author_email="wanglei@iphy.ac.cn",
    description="Differentiable Programming Tensor Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="Differentiable Programming Tensor Networks",
    url="https://github.com/wangleiphy/tensorgrad",
    packages=["tensornets"],
    install_requires=[
            "numpy",
            "torch>=1.3.0",
        ],
    python_requires='>=3',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
