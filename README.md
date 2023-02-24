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
conda install -c conda-forge pandas pyyaml pyhive libthrift great-expectations xgboost lightgbm cerberus sasl pytz termcolor
conda install -c anaconda teradata pymssql
conda install -c temporary-recipes teradatasql

```

5. Install **execution-framework** using the latest version on GitHub.

```sh
python -m pip install git+https://github.com/advanced-analytics-tdp/execution-framework.git
````
