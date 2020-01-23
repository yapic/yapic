import os

from setuptools import setup, find_packages

reqs = ['yapic_io>=0.1.0',
        'docopt>=0.6.2',
        'numpy>=1.15.4',
        'Keras>=2.3.1',
        'tensorflow>=2.1.0']


def readme():
    README_md = os.path.join(os.path.dirname(__file__), 'README.md')
    with open(README_md) as f:
        return f.read()


ns = os.environ.get('CI_PROJECT_NAMESPACE', 'idaf')

setup(name='yapic',
      version='1.0.1',
      description='Yet another Pixel Classifier (based on deep learning)',
      long_description=readme(),
      author='Manuel Schoelling, Christoph Moehl',
      author_email='manuel.schoelling@gmx.de, christoph.moehl@dzne.de',
      packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
      zip_safe=False,
      install_requires=reqs,
      test_suite='nose.collector',
      entry_points={
          'console_scripts': ['yapic=yapic.main:entry_point'],
      },
      tests_require=['nose', 'coverage', 'nose-timer', 'nose-deadline'])
