import warnings

try:
    from Cython.Distutils import build_ext
    from setuptools import setup, Extension
    HAVE_CYTHON = True
except ImportError as e:
    warnings.warn(e.message)
    from setuptools import setup, Extension
    from setuptools.command.build_ext import build_ext
    HAVE_CYTHON = False

import numpy

_utils = Extension('sstsne._utils',
                   sources=['sstsne/_utils.pyx'],
                   include_dirs=[numpy.get_include()])

_barnes_hut_tsne = Extension('sstsne._barnes_hut_tsne',
                             sources=['sstsne/_barnes_hut_tsne.pyx'],
                             include_dirs=[numpy.get_include()])

def readme():
    with open('README.md') as readme_file:
        return readme_file.read()

configuration = {
    'name' : 'sstsne',
    'version' : '0.1',
    'description' : 'Semi-Supervised t-SNE using a Bayesian prior based on partial labelling',
    'long_description' : readme(),
    'classifiers' : [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Programming Language :: C',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
    ],
    'keywords' : 'tsne semi-supervised dimension reduction',
    'url' : 'http://github.com/lmcinnes/sstsne',
    'maintainer' : 'Leland McInnes',
    'maintainer_email' : 'leland.mcinnes@gmail.com',
    'license' : 'BSD',
    'packages' : ['sstsne'],
    'install_requires' : ['scikit-learn>=0.17.1',
                          'cython >= 0.17'],
    'ext_modules' : [_utils,
                     _barnes_hut_tsne],
    'cmdclass' : {'build_ext' : build_ext},
    'test_suite' : 'nose.collector',
    'tests_require' : ['nose'],
    }

if not HAVE_CYTHON:
    _utils.sources[0] = '_utils.c'
    _barnes_hut_tsne.sources[0] = '_barnes_hut_tsne.c'
    configuration['install_requires'] = ['scikit-learn>=0.17.1']

setup(**configuration)