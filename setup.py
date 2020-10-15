import os
from setuptools import setup, find_packages

try:
    import tensorflow as tf
except ModuleNotFoundError:
    msg = ('You have to install tensorflow or tensorflow-gpu version '
           '1.12, 1.13, 1.14, 1.15 or 2.1'
           'before installing YAPiC')
    raise ModuleNotFoundError(msg)


reqs = ['yapic_io>=0.1.0',
        'docopt>=0.6.2',
        'numpy>=1.15.4']

tf_version = [int(num) for num in tf.__version__.split('.')]

if tf_version[0] == 2 and tf_version[1] == 1:
    reqs.append('Keras>=2.3.0')

elif tf_version[0] == 1:
    if tf_version[1] == 12:
        # tensorflow==1.12
        reqs.append('Keras==2.2.4')
    else:
        # tensorflow==1.12, 1.13, 1.14 or 1.15
        reqs.append('Keras>=2.3.0')
else:
    msg = ('incompatible tensorflow version, '
           'use 1.12, 1.13, 1.14, 1.15 or 2.1')
    raise Exception(msg)


def readme():
    README_md = os.path.join(os.path.dirname(__file__), 'README.md')
    with open(README_md) as f:
        return f.read()


ns = os.environ.get('CI_PROJECT_NAMESPACE', 'idaf')

setup(name='yapic',
      version='1.0.2',
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
