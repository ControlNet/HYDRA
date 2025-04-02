import os.path
from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as f:
        return f.read().splitlines()

version = {}
with open(os.path.join("hydra_vl4ai", "_version.py")) as fp:
    exec(fp.read(), version)

setup(
    name="hydra_vl4ai",
    version=version['__version__'],
    description="Official implementation for HYDRA.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="ControlNet",
    author_email="smczx@hotmail.com",
    url="https://hydra-vl4ai.github.io/",
    project_urls={
        "Source Code": "https://github.com/ControlNet/HYDRA",
        "Bug Tracker": "https://github.com/ControlNet/HYDRA/issues",
    },
    keywords=["deep learning", "pytorch", "AI"],
    packages=find_packages(include=["hydra_vl4ai", "hydra_vl4ai.*"]),
    package_data={
        "hydra_vl4ai": [
            "agent/prompt/**", # prompts
            "assets/*", # other assets
            "tool/model/xvlm/config_bert.json" # xvlm assets
        ]
    },
    python_requires='>=3.10',
    install_requires=parse_requirements('requirements.txt'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
)