import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mytflib", # Replace with your own username
    version="0.0.1",
    author="John Park",
    author_email="parkjohnyc@gmail.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/johnypark/mytflib",
    packages=setuptools.find_packages(),
    install_requires = ['tensorflow',
                       'keras',
                       'pandas',
                       'numpy',
                       'tensorflow_addons'
                       ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Apache 2.0",
        "Operating System :: OS Independent",
    ],
    python_requires='3',
)

