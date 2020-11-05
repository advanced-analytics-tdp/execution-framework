import pandas as pd
import teradatasql
import logging

from execution_framework.utils.common_utils import get_df_py_dtypes
from execution_framework.utils.db_utils import create_teradata_table


logger = logging.getLogger('FASTLOAD TERADATA')


def fastload_dataframe(database_connection: teradatasql.TeradataConnection, df: pd.DataFrame,
                       table_name: str, overwrite_table: bool = False, primary_index: bool = False,
                       primary_index_cols: list = None) -> None:
    """
    Use fastload utility to upload data to Teradata

    :param database_connection: Teradata database connection
    :param df: dataframe to upload
    :param table_name: table name where data will be uploaded
    :param overwrite_table: overwrite if table_name already exists
    :param primary_index: true if you want to set primary index of the table, False by default
    :param primary_index_cols: columns that will be primary index
    :return:
    """

    # Get data types from dataframe
    metadata_dict = get_df_py_dtypes(df)

    # Create teradata table
    create_teradata_table(database_connection=database_connection,
                          metadata_columns=metadata_dict,
                          table_name=table_name,
                          set_primary_index=primary_index,
                          primary_index_columns=primary_index_cols,
                          overwrite=overwrite_table)

    # Start fastload job
    number_columns = df.shape[1]
    question_mark_values = ('?,' * number_columns)[:-1]
    fastload_statement = f'{{fn teradata_require_fastload}} INSERT INTO {table_name} ({question_mark_values})'

    # Execute statements
    cursor = database_connection.cursor()
    cursor.execute('{fn teradata_nativesql}{fn teradata_autocommit_off}')

    logger.debug('Executing {}'.format(fastload_statement))
    logger.info('Start Fastload execution')

    try:
        cursor.execute(fastload_statement, df.values.tolist())
    except Exception:
        logger.error("Can't execute teradata fastload successfully", exc_info=True)
        raise

    logger.info('Fastload execution finished')

    # Commit results
    logger.info('Commiting operation')
    database_connection.commit()
    cursor.execute('{fn teradata_nativesql}{fn teradata_autocommit_on}')

    logger.info('Dataframe successfully upload')
