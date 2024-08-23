from setuptools import find_packages, setup


def get_requirements(path: str):
    return [l.strip() for l in open(path)]


setup(
    name="puli",
    version="0.0.3",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)