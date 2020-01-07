"""
   Copyright 2019-2020 Boris Shminke

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neural-semigroups",
    version="0.1.0",
    author="Boris Shminke",
    author_email="inpefess@protonmail.com",
    description="Neural networks powered research of semigroups",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/inpefess/neural-semigroups",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Typing :: Typed"
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "tqdm",
        "pytorch-ignite",
        "tensorboard",
        "future"
    ]
)
