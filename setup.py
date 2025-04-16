from setuptools import setup, find_packages
import os

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Filter out comments and empty lines
requirements = [req for req in requirements if req and not req.startswith('#')]

setup(
    name="box-order-prediction",
    version="0.1.0",
    description="Box order prediction using machine learning",
    author="Tofu(X)",
    packages=find_packages(include=['app', 'app.*', 'src', 'src.*']),
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'train-model=src.train_model:main',
            'fine-tune-gb=src.fine_tune_gb:main',
            'box-api=app.main:start',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)