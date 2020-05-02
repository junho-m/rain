import os
import setuptools


setuptools.setup(
    name='rain',
    version="0.1.dev0",
    author="Moon junho",
    author_email="juno1moon@gmail.com",
    packages=setuptools.find_packages(),
    description='Deep learning for precipitation rainfall',
    long_description=open(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'README.md')).read(),
    license='MIT',
    url='https://github.com/junho-m/rain',
    install_requires=['tensorflow', 'pandas','numpy', 'scipy', 'matplotlib' ],
    classifiers=['Development Status :: 2 - Pre-Alpha',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Natural Language :: English',
                 'Operating System :: OS Independent',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence'],
)
