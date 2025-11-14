from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="future_ai_metrics",  
    version="0.1.0",
    description="A lightweight evaluation toolkit for AI model assessment (based on FUTURE-AI principles)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="FUTURE-AI consortium",
    author_email="parodbe@gmail.com",
    url="https://github.com/pabloiislafe/FUTURE-AI-metrics",
    packages=find_packages(),
    license="GNU")
