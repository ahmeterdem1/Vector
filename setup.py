import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vectorgebra",
    version="3.0.0",
    author="Ahmet Erdem",
    description="A numerical methods tool for python, in python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.9',
    py_modules=["vectorgebra"],
    install_requires=[]
)