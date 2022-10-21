from setuptools import setup, find_packages

setup(
    author='Simon Clifford',
    author_email='sjc306@cam.ac.uk',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    name='process_model',
    description='''\
Code to read a TensorFlow SavedModel and export the
necessary Fortran code to interface it to the fortran-tf-lib.
    ''',
    license="MIT license",
    version='0.1.0',
    packages=find_packages(include=['process_model', 'process_model.*']),
    include_package_data=True,
    install_requires=[
        'Click',
        'tensorflow',
        'jinja2',
    ],
    entry_points={
        'console_scripts': [
            'process_model = process_model.process_model:main',
        ],
    },
)
