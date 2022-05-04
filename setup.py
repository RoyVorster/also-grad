from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Just autograd. Simple implementation.'

# Simplest possible setup
setup(
    name='also-grad',
    version=VERSION,
    author='Roy Vorster',
    author_email='royvorster@gmail.com',
    description=DESCRIPTION,
    packages=find_packages(),
    url='https://github.com/RoyVorster/also-grad',
)
