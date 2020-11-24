import pandas as pd
import numpy as np
import logging
import yaml

from typing import List
from datetime import date

logger = logging.getLogger('COMMON UTILS')


def get_df_py_dtypes(df: pd.DataFrame) -> dict:
    """
    Get python native data types of columns in pandas dataframe

    :param df: pandas dataframe
    :return: dict with all columns and its data types
    """

    # Initialize dictionary
    metadata_dict = dict()

    # Measurer to get max length of object columns
    measurer = np.vectorize(len)

    # Iterate over all columns
    for column in df.columns:

        # Get first element of column to know data type
        element = df[column].head(1).values.tolist()[0]

        if isinstance(element, str):
            max_column_length = measurer(df[column].values).max(axis=0)
            metadata_dict[column] = ('str', max_column_length)
        elif isinstance(element, int):
            metadata_dict[column] = 'int'
        elif isinstance(element, float):
            metadata_dict[column] = 'float'
        elif isinstance(element, date):
            metadata_dict[column] = 'date'
        else:
            raise NotImplementedError('Data types supporting for now : str, int , float and date.'
                                      ' {} is not supported yet'.format(type(element)))

    return metadata_dict


def read_configuration_file(file_path: str):
    """
    Read  configuration file in YAML format

    :param file_path: path of configuration file
    :return: configuration parameters
    """
    try:
        with open(file_path) as f:
            conf_file = yaml.load(f, Loader=yaml.FullLoader)
    except Exception:
        logger.error("Can't read variables from YAML file in path '{}'".format(file_path), exc_info=True)
        raise

    return conf_file


def read_variables(file_path: str) -> list:
    """
    Read variables from YAML file

    :param file_path: path of YAML file with names
    :return: list with lowercase variable names
    """

    variables = read_configuration_file(file_path)
    variables = [var.lower() for var in variables]

    return variables


def separate_schema_table(name: str, dbms: str) -> List[str]:
    """
    Separate databse or schema from table name in case there is no schema or database function assigns the default
    depending on dbms. dbi_min for Teradata and dev_perm for Hive

    :param dbms: datawarehouse name
    :param name: name with table and schema
    :return:
    """
    assert isinstance(name, str), f"name parameter should be a string"

    if name.count('.') > 1:
        raise Exception(f"'{name}' can't contain more than one period")
    elif name.count('.') == 1:
        return name.split('.')
    else:
        if dbms == 'teradata':
            return ['dbi_min', name]
        elif dbms == 'hive':
            return ['dev_perm', name]
        else:
            raise NotImplementedError('Data warehouses supporting by now : hive and teradata')
