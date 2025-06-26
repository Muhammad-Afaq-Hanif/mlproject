from setuptools import find_packages, setup

def load_requirements(filename):
    with open(filename) as file:
        return file.read().splitlines()
    

setup(
    name = 'ML_Project',
    version = '1.0',
    author = "Muhammad Afaq Hanif",
    author_email = "afaqmuhammad74@gmail.com",
    packages= find_packages(),
    install_requires = load_requirements('requirements.txt')
)