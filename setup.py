from pathlib import Path

from setuptools import find_packages, setup


def read_requirements() -> list[str]:
    requirements_file = Path(__file__).with_name("requirements.txt")
    return [
        line.strip()
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


setup(
    name="PolicyPilot",
    version="0.1",
    author="Sumit Prasad",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=read_requirements(),
)
