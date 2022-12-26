# Execution Framework
Framework for execution of models.

## Installation

1. Before install the package, make sure you have python >= 3.8 on your system.

2. Although the package can be installed on any environment, we recommend to do it on a separate environment.

```sh
conda create --name myenv python=3.8
```

3. Activate the environment where you want to install the package.

```sh
conda activate myenv
```

4. Some packages need to be installed before 

```sh
conda install -c conda-forge pandas
conda install -c anaconda teradata
conda install -c temporary-recipes teradatasql
conda install -c conda-forge pyyaml
conda install -c conda-forge pyhive
conda install -c conda-forge libthrift
conda install -c conda-forge great-expectations
conda install -c conda-forge xgboost
conda install -c conda-forge lightgbm
conda install -c conda-forge cerberus
conda install -c conda-forge sasl
conda install -c conda-forge pytz
conda install -c conda-forge termcolor
conda install -c anaconda pymssql
```

5. Install **execution-framework** using the latest version on GitHub.

```sh
python -m pip install git+https://github.com/advanced-analytics-tdp/execution-framework.git
````
