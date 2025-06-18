from setuptools import setup, find_packages

setup(
    name='plugin',
    version='1.0.0',
    packages=find_packages(),
    install_requires=['mlflow',
                      'mmsegmentation',
                      'pytorch3d'],
)
