import os
from subprocess import Popen, PIPE
from distutils import spawn
import platform
from setuptools import setup, find_packages
from yapic.version import __version__

reqs = ['yapic_io>=0.2.6',
        'docopt>=0.6.2',
        'numpy>=1.15.4']


def readme():
    README_md = os.path.join(os.path.dirname(__file__), 'README.md')
    with open(README_md) as f:
        return f.read()


ns = os.environ.get('CI_PROJECT_NAMESPACE', 'idaf')

setup(name='yapic',
      version=__version__,
      description='Yet another Pixel Classifier (based on deep learning)',
      long_description=readme(),
      long_description_content_type='text/markdown',
      url='https://yapic.github.io/yapic/',
      author='Manuel Schoelling, Christoph Moehl',
      author_email=('manuel.schoelling@gmx.de, '
                    'christoph.oliver.moehl@gmail.com'),
      packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
      include_package_data=True,
      zip_safe=False,
      install_requires=reqs,
      test_suite='nose.collector',
      entry_points={
          'console_scripts': ['yapic=yapic.main:entry_point'],
      },
      tests_require=['nose', 'coverage', 'nose-timer', 'nose-deadline'])
