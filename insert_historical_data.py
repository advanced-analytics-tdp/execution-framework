import teradatasql
import logging

from utils.db_utils import execute_db_statement, check_data_exits_in_table
from utils.date_utils import subtract_units_time

logger = logging.getLogger('HISTORICAL DATA INSERTION')


def insert_into_historical_table(historical_table: str, period_column: str, offset: int, current_replica_table: str,
                                 columns: list, replica_period: str, overwrite_data: bool,
                                 database_connection: teradatasql.TeradataConnection) -> None:
    """
    Insert data current replica results into historical table, all tables must be in DBI_MIN schema

    :param historical_table: historical table name
    :param period_column: period column name in historical table
    :param offset: difference in months so that the replica month reaches the month of the historical table
    :param current_replica_table: current replica table name
    :param columns: columns name to insert into historical table
    :param replica_period: current replica month in format 'YYYY-MM-DD'
    :param overwrite_data: true if you want to overwrite data in historical table
    :param database_connection: Teradata database connection
    :return:
    """

    # Generate period to check in historical table
    period_to_check = subtract_units_time(replica_period, 'months', offset*-1)

    # Check if data with the same period we are trying to insert in historical table already exists
    if check_data_exits_in_table(historical_table, period_column, period_to_check, database_connection):

        logger.warning(f'Period {period_to_check} already exists in {historical_table}')

        if overwrite_data:

            logger.info(f'Deleting period {period_to_check} from {historical_table}')

            execute_db_statement(database_connection,
                                 f"DELETE FROM {historical_table} WHERE {period_column} = '{period_to_check}'")

        else:
            logger.error("Can't insert data because data with same period already exists, please change "
                         "overwrite_data parameter to True")
            raise
    else:
        logger.info(f'There is no data with the same period {period_to_check} '
                    f'in {historical_table} we are trying to insert')

    # Create insert query to insert in historical table
    if columns == 'all':
        insert_query = f"INSERT INTO {historical_table} SELECT * FROM {current_replica_table}"
    else:

        selected_columns = ', '.join(columns)

        insert_query = f"INSERT INTO {historical_table} SELECT {selected_columns} FROM {current_replica_table}"

    logger.debug(f'Executing this query to insert information in historical table : {insert_query}')

    # Execute insert statement
    execute_db_statement(database_connection, insert_query)

    logger.info(f"Data of period {period_to_check} from '{current_replica_table}' was "
                f"inserted successfully in '{historical_table}'")
