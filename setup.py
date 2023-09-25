import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vectorgebra",
    version="1.2.0",
    author="Ahmet Erdem",
    description="A Python-based math library with linear algebra",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6',
    py_modules=["vectorgebra"],
    install_requires=[]
)