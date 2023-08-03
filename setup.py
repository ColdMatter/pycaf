import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="pycaf",
    version="0.1.0",
    author="Arijit Chakraborty",
    author_email="arijit.phd@gmail.com",
    description="A package to run and analyse CaF experiment data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arijitphd/pycaf",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
