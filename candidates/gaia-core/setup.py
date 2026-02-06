from setuptools import setup, find_packages

setup(
    name='gaia_core',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # Add any runtime dependencies here
        # For example:
        # 'numpy',
        # 'pydantic',
    ],
    extras_require={
        'dev': [
            # Add development dependencies here
            # 'pytest',
            # 'flake8',
        ],
    },
    python_requires='>=3.11',
    description='Core cognitive and reasoning engine for GAIA.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/your-org/gaia_core',
    author='GAIA Team',
    author_email='gaia-team@example.com',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
