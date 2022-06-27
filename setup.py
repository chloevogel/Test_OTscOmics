from setuptools import setup
from test_functions import __version__


setup(name='test_functions',
version= __version__,
description='Testing the computation of OTscOmics',
url='https://github.com/chloevogel/Test_OTscOmics',
author='Chloé Vogel',
author_email='chloe.vogel@etu.minesparis.psl.eu',
packages=['test_functions'],
zip_safe=False)
