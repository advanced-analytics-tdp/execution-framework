import pandas as pd
import teradatasql
import logging


from execution_framework.utils.common_utils import separate_schema_table
from typing import Union
from pyhive import hive
import pymssql

logging.getLogger('pyhive').setLevel(logging.CRITICAL)
logger = logging.getLogger('DB UTILS')


def teradata_connection(user_name: str, password: str, host: str = '10.226.0.34',
                        database: str = 'dbi_min', **kwargs) -> teradatasql.TeradataConnection:
    """
    Create connection to Teradata Database

    :param user_name: Teradata user name
    :param password: Teradata password
    :param host: host Teradata runs on
    :param database: database to use by default
    :param kwargs: additional arguments for connect function
    :return: Teradata Connection
    """
    try:
        conn = teradatasql.connect(host=host, user=user_name, password=password, database=database, **kwargs)
    except Exception:
        logger.error("Can't connect to Teradata database, check traceback", exc_info=True)
        raise

    return conn


def hive_connection(host: str = '10.4.88.31', port: int = 10000, database: str = 'dev_perm',
                    configuration: dict = None) -> hive.Connection:
    """
    Create Hive connection

    :param host: host HiveServer2 runs on
    :param port: port HiveServer2 runs on
    :param database: database to use by default
    :param configuration:
    :return: Hive Connection
    """
    # Default configuration for  Hive Connection
    if configuration is None:
        configuration = {'hive.resultset.use.unique.column.names': 'false',
                         'hive.exec.compress.output': 'true',
                         'hive.groupby.orderby.position.alias': 'true',
                         'hive.server2.thrift.resultset.default.fetch.size': '10000'}

    # Create connection
    try:
        conn = hive.Connection(host=host, port=port, database=database, configuration=configuration)
    except Exception:
        logger.error(f"Can't create Hive Connection to {host} host and {port} port")
        raise

    return conn


def sqlserver_connection(host: str, user_name: str, password:str, database: str = 'master', **kwargs):

    """
    Create connection to SQL Server Database
    :param host: SQL Server
    :param user_name: SQL Server user
    :param password: SQL Server password
    :param database: SQL Server database. Defaul: master
    :param kwargs: additional arguments for connect function
    :return: SQL Server Connection
    """
    try:
        conn = pymssql.connect(host, user_name, password, database, **kwargs)
    except Exception:
        logger.error("Can't connect to SQL Server data, check traceback", exe_info=True)
        raise

    return conn


def execute_db_statement(database_connection: Union[teradatasql.TeradataConnection, hive.Connection],
                         statement: str) -> None:
    """
    Execute statement in Teradata or Hive

    :param database_connection: Hive or Teradata database connection
    :param statement: statement to be executed
    :return:
    """

    # Create cursor to execute statement
    try:
        cursor = database_connection.cursor()
    except Exception:
        logger.error("Can't initialize the cursor Object", exc_info=True)
        raise

    # Execute statement
    try:
        logger.debug(f"Executing '{statement}'")
        cursor.execute(statement)
    except Exception:
        logger.error(f"Can't execute statement {statement}", exc_info=True)
        raise


def execute_store_procedure(database_connection: teradatasql.TeradataConnection, metadata_procedure: dict):
    """
    Execute store procedure in Teradata database

    :param database_connection: Teradata database connection
    :param metadata_procedure: procedure name and parameters
    :return:
    """
    # Get procedure name and parameters
    sp_name = metadata_procedure['name']
    parameters = list(metadata_procedure['parameters'].values())

    logger.info(f"Executing {sp_name} procedure with parameter(s) {parameters}")

    # Create cursor
    try:
        cursor = database_connection.cursor()
        cursor.callproc(sp_name, parameters)
    except Exception:
        logger.error("Can't execute store procedure in Teradata", exc_info=True)
        raise

    logger.info(f"Store procedure executed successfully")


def check_data_exits_in_table(table_name: str, period_column: str, period: str,
                              database_connection: Union[teradatasql.TeradataConnection, hive.Connection]) -> bool:
    """
    Given a certain period, check if data with that period exists in table

    :param table_name: table name in Teradata
    :param period_column: period column name in table
    :param period: period to check in string format YYYY-MM-DD
    :param database_connection: Hive or Teradata database connection
    :return: True if data for the specific period exists
    """

    # Generate query to validate if data exists
    validate_query = f"SELECT COUNT(*) total_rows FROM {table_name} WHERE {period_column}='{period}'"

    logger.debug(f'Executing this query to check if there is information : {validate_query}')

    # Execute query and save results in pandas dataframe
    result = read_query_to_df(database_connection, validate_query)

    # Check results
    number_rows = result['total_rows'][0]

    if number_rows != 0:
        logger.debug(f"Table '{table_name}' has {number_rows} rows in {period} period")
        return True
    else:
        return False


def read_query_to_df(database_connection: Union[teradatasql.TeradataConnection, hive.Connection], query: str,
                     arraysize: int = 10000) -> pd.DataFrame:
    """
    Executes query in database and result in dataframe

    :param database_connection: Hive or Teradata database connection
    :param query: sql query to be executed in database
    :param arraysize: cursor array size to fetch results
    :return: dataframe with result of query execution
    """
    # Create cursor
    cursor = database_connection.cursor()
    cursor.arraysize = arraysize
    logger.debug(f'Cursor array size of connection set to {cursor.arraysize}')

    # Execute query
    logger.debug(f"Starting to execute '{query}'")
    try:
        cursor.execute(query)
    except Exception:
        logger.error("Can't execute query. Check if it is correct", exc_info=True)
        raise

    logger.debug("Finished query execution")

    # Fetch all results
    logger.debug("Starting to fetch query results")

    results = cursor.fetchall()

    logger.debug('Finished fetching all query results')

    # Transform into dataframe
    logger.debug('Transforming result into dataframe')

    df_results = pd.DataFrame(results, columns=[desc[0] for desc in cursor.description])

    logger.debug('Dataframe was successfully created')

    return df_results


def read_table_to_df(table_name: str, database_connection: Union[teradatasql.TeradataConnection, hive.Connection],
                     filters: str = None, arraysize: int = 10000) -> pd.DataFrame:
    """
    Load database table in dataframe

    :param table_name: table name, specify database or schema if it is not dbi_min or dev_perm which are the default
    values
    :param database_connection: Hive or Teradata database connection
    :param filters: sql filters to execute before load table in dataframe
    :param arraysize: cursor array size to fetch results
    :return: dataframe with table
    """
    if filters is not None:
        query = f"SELECT * FROM {table_name} WHERE {filters}"
    else:
        query = f"SELECT * FROM {table_name}"

    results = read_query_to_df(database_connection, query, arraysize)

    return results


def check_table_exists(dbms: str, database_name: str, table_name: str,
                       database_connection: Union[teradatasql.TeradataConnection, hive.Connection]) -> bool:
    """
    Verify if table exists in Teradata or Hive

    :param dbms: data warehouse name
    :param database_name: database name
    :param table_name: table name
    :param database_connection: Teradata or Hive connection
    :return: True if table exists
    """

    # Create validation query
    if dbms == 'teradata':

        if isinstance(database_connection, teradatasql.TeradataConnection):
            query = f"SELECT * FROM dbc.TablesV WHERE databasename = '{database_name}' AND tablename = '{table_name}'"
        else:
            raise TypeError('Database connection should be an instance of teradatasql.TeradataConnection')

    elif dbms == 'hive':

        if isinstance(database_connection, hive.Connection):
            query = f"SHOW TABLES IN {database_name} LIKE '{table_name}'"
        else:
            raise TypeError('Database connection should be an instance of hive.Connection')

    else:
        raise NotImplementedError('Data warehouses supporting by now : hive and teradata')

    # Check if result is empty
    result = read_query_to_df(database_connection, query)
    table_exists = False if result.empty else True

    return table_exists


def create_teradata_table(database_connection: teradatasql.TeradataConnection, metadata_columns: dict,
                          table_name: str, set_primary_index: bool = False, primary_index_columns: list = None,
                          overwrite: bool = False) -> None:
    """
    Create a multiset table in Teradata
    Table will be created in dbi_min scheme
    All columns will be set as varchar

    :param database_connection: Teradata database connection
    :param metadata_columns: names and length of columns
    :param table_name: new table name, without schema
    :param set_primary_index: true if you want to set primary index of the table, False by default
    :param primary_index_columns: columns that will be primary index
    :param overwrite: if table exists, overwrite it
    :return:
    """

    # Separate schema and table name
    schema, table = separate_schema_table(table_name, 'teradata')

    # Check if table name already exists
    if check_table_exists('teradata', schema, table, database_connection):
        if overwrite:
            logger.info(f'Table {table} already exists in {schema} scheme, it will be deleted')
            execute_db_statement(database_connection, f'DROP TABLE {schema}.{table}')
        else:
            raise RuntimeError(f'Table {table} already exists in {schema} schema, please choose another name'
                               f' or set overwrite parameter to True')

    # Generate query string to create table
    create_query = f'CREATE MULTISET TABLE {schema}.{table} ( '

    # Add the columns and its data types
    declare_columns = ''
    for k, v in metadata_columns.items():
        if isinstance(v, tuple):
            declare_columns += k + f' VARCHAR({v[1]}) CASESPECIFIC, '
        else:
            if v == 'float':
                declare_columns += k + ' FLOAT, '
            elif v == 'int':
                declare_columns += k + ' INTEGER, '
            elif v == 'date':
                declare_columns += k + ' DATE, '

    create_query = create_query + declare_columns[:-2] + ')'

    # Set primary index
    if set_primary_index:

        # Check if index columns exists in metadata columns of table
        for index_column in primary_index_columns:
            if index_column not in metadata_columns:
                raise Exception(f"{index_column} does not exists in metadata_columns."
                                f" Please check primary index column name(s) ")

        create_query = create_query + f" PRIMARY INDEX({', '.join(primary_index_columns)})"

    # Execute create query statement
    execute_db_statement(database_connection, create_query)

    # Log finish of process
    logger.info(f"Table {table_name} was successfully created")
