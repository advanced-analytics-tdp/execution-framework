import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="execution-framework",
    version="0.0.1",
    author="Movistar",
    author_email="leibnitz.rojas@telefonica.com",
    description="Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/advanced-analytics-tdp/execution-framework',
    project_urls={
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.6",
    install_requires=['pandas',
                      'teradatasql',
                      'pyyaml',
                      'pyhive',
                      'thrift',
                      'great_expectations',
                      'xgboost',
                      'lightgbm',
                      'cerberus', 'sasl', 'pytz', 'termcolor', 'pymssql',],
)