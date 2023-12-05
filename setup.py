
from setuptools import setup,find_packages
from typing import List


HYPEN_E_DOT = "-e ."

def get_requirements(filepath:str)->List[str]:
    requirements = []
    with open(filepath) as f:
        requirements = f.readlines()
        requirements= [req.replace("\n","") for req in requirements ]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

setup(
   name =  "mlproject",
   version = "0.1.0",
   packages = find_packages(),
   author = "harcuracy",
   author_email = "akandesoji4christ@gmail.com",
   url = "https://github.com/harcuracy/mlproject_end_end",
   description = "A framework for building machine learning projects",
   install_requires = get_requirements("requirement.txt")
)