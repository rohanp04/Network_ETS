from setuptools import find_packages,setup
from typing import List

def get_requirements()->List[str]:
    """
     This function is used to get all the requirements mentioned in the requirements.txt
    """
    requirement_lst:List[str] = []
    try:
        with open('requirements.txt','r') as file:
            lines=file.readlines()

            for line in lines:
                requirement=line.strip()
                ## reading all the requirements from the file except -e .
                if requirement and requirement != "-e .":
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirement.txt Not Found")
    
    return requirement_lst


setup(
    name="NetworkSecurity_ETSpipeline",
    version="01",
    author="Rohan Prajapati",
    packages=find_packages(),
    install_requires=get_requirements()
)