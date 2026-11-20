from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip()]

setup(
    name="eduscore-predictor",
    version="2.0.0",
    description="Student math score prediction and analytics dashboard",
    author="Muhammad Afaq Hanif",
    author_email="afaqmuhammad74@gmail.com",
    packages=find_packages(where=".", exclude=["tests*", "notebooks*"]),
    install_requires=requirements,
    python_requires=">=3.10",
)