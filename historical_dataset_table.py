import pandas as pd
import teradatasql
import logging

from typing import Union
from pyhive import hive

from utils.db_utils import read_table_to_df, check_table_exists
from utils.common_utils import separate_schema_table
from utils.date_utils import last_day_of_month


logger = logging.getLogger('DATASET TABLE')


def read_dataset_table(table_name: str, dbms: str, period_column: str, day_month: str, current_period: str,
                       database_connection: Union[teradatasql.TeradataConnection, hive.Connection]) -> pd.DataFrame:
    """
    Read a period of historical dataset table

    :param table_name: historical dataset table name
    :param dbms: data warehouse name
    :param period_column: period column name
    :param day_month: first or last day of month
    :param current_period: current period to filter table
    :param database_connection: Hive or Teradata database connection
    :return: current period of historical dataset table
    """

    # Check dataset table exists
    schema, table = separate_schema_table(table_name, dbms)
    if not check_table_exists(dbms, schema, table, database_connection):
        raise Exception(f"Table {table} doesn't exists in schema {schema}")

    # Create filter to read table from dbms
    if day_month == 'first':
        filter_date = current_period
    elif day_month == 'last':
        filter_date = last_day_of_month(current_period)
    else:
        raise NotImplementedError('Day months types supporting for now: first and last')

    table_filter = f"{period_column} = '{filter_date}'"

    # Read dataset table from dbms
    dataset_table = read_table_to_df(table_name, database_connection, table_filter, arraysize=20000)

    # Check is dataset table is empty for current period
    if dataset_table.empty:
        raise Exception(f'{table_name} is empty for {current_period} period')

    # Lowercase column names
    dataset_table.columns = map(str.lower, dataset_table.columns)

    return dataset_table
