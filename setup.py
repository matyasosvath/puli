from setuptools import find_packages, setup


def get_requirements(path: str):
    return [l.strip() for l in open(path)]


setup(
    name="puli2",
    version="0.0.2",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)