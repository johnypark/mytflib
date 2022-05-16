import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mytflib", # Replace with your own username
    version="0.0.2.0",
    author="John Park",
    author_email="parkjohnyc@gmail.com",
    description="My TensorFlow Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/johnypark/mytflib",
    packages=setuptools.find_packages(),
    install_requires = ['tensorflow',
                       'opencv-python',
                       'tensorflow-addons',
                        'one_cycle_tf @ git+https://github.com/johnypark/one_cycle_scheduler_tf',
                        'typeguard'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)

