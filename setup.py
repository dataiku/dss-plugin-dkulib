from setuptools import setup, find_packages

setup(
    name="dkulib",
    version="0.0.1",
    description='Library for plugin development',
    author="Dataiku (Henri Chabert)",
    url='https://github.com/dataiku/dss-plugin-dkulib.git',  # Provide either the link to your github or to your website
    packages=find_packages(),
    classifiers=[
            'Intended Audience :: Developers',
            'License :: OSI Approved :: Apache Software License',
            'Topic :: Software Development :: Libraries',
            'Programming Language :: Python',
            'Operating System :: OS Independent'
        ],
    python_requires='>=3.5'
)