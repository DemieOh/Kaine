import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='kaine',
    version='0',
    author='Demian Oh',
    author_email='demie.oh@gmail.com',
    description='The Replicant',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/DemieOh/Kaine',
    packages=setuptools.find_packages(),
    classifier=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7'
)
