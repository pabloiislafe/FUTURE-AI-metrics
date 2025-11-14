from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="Metrics",                  # nombre del paquete
    version="0.1.0",
    description="Evaluation toolkit for AI model assessment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="FUTURE-AI consortium",
    author_email="leonor_cerda@iislafe.es",
    url="https://github.com/pabloiislafe/FUTURE-AI-metrics",
    packages=["Metrics", "Metrics.Robustness"],   # 👈 INSTALA SOLO Metrics
    include_package_data=True,
    license="GNU"
)
