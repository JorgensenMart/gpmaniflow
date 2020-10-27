from setuptools import setup, find_packages

setup(
    name='gpmaniflow',
    version='0.0.2',
    packages=find_packages(),
    package_data={
        '': [
            'install_requirements.txt', '*.json'
        ]
    },
    install_requires=[],
    dependency_links=[]
)