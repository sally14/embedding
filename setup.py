import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="embeddings",
    version="0.0.1",
    author="sally14",
    author_email="sally14.doe@gmail.com",
    description="A word2vec preprocessing and training package",
    long_description="A word2vec preprocessing and training package",
    long_description_content_type="text/markdown",
    url="https://github.com/sally14/embedding",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)