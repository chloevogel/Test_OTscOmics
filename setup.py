from setuptools import setup
from test_functions import __version__


setup(name='test_functions',
      version= __version__,
      description='Testing the computation of OTscOmics',
      url='https://github.com/chloevogel/Test_OTscOmics',
      author='Chloé Vogel',
      author_email='chloe.vogel@etu.minesparis.psl.eu',
      packages=['test_functions'],
      install_requires=[
        'pot>=0.8',
        'torch>=1.0',
        'numpy>=1.20',
        'scipy>=1.6',
        'tqdm>=4.62'
      ]
     )
