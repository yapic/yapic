import os

from setuptools import setup, find_packages
try:  # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements

requirements_txt = os.path.join(os.path.dirname(__file__), 'requirements.txt')
install_reqs = parse_requirements(requirements_txt, session=False)
reqs = [str(ir.req) for ir in install_reqs]


def readme():
    README_md = os.path.join(os.path.dirname(__file__), 'README.md')
    with open(README_md) as f:
        return f.read()


ns = os.environ.get('CI_PROJECT_NAMESPACE', 'idaf')

setup(name='yapic',
      version='1.0.0',
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
