import pandas as pd
import teradatasql
import logging

from pathlib import Path

from execution_framework.utils.common_utils import get_df_py_dtypes, save_string_to_file
from execution_framework.utils.date_utils import get_current_date_str_format
from execution_framework.utils.db_utils import create_teradata_table


logger = logging.getLogger('FASTLOAD TERADATA')


def fetch_statement_results(cursor: teradatasql.TeradataCursor, statement: str) -> list:
    """
    Given a sql statement execute it and fetch its results

    :param cursor: cursor from Teradata connection
    :param statement: statement to execute
    :return: execution results
    """

    logger.debug(f"Executing {statement}")
    cursor.execute(statement)
    results = cursor.fetchall()

    return results


def check_statement_results(results: list, type_result: str) -> bool:
    """
    Check if execution results of statement contains warnings or error

    :param results: results of an statement execution
    :param type_result: warning or error
    :return:
    """

    # Results always are in a list of list
    result_str = results[0][0]

    issues_found = False

    # Check if results are empty
    if not result_str:
        logger.info(f'There are no {type_result}s after execution')

    else:

        issues_found = True

        if type_result == 'warning':
            logger.warning('Warnings found after fastload execution')

        elif type_result == 'error':
            logger.error('Errors found after fastload execution')

        else:
            NotImplementedError('Type result supporting by now are warning and error')

    return issues_found


def check_warnings_and_errors(cursor: teradatasql.TeradataCursor, fastload_statement: str, log_filepath: str) -> None:
    """
    Look for warnings and errors after a fastload execution

    :param cursor: cursor from Teradata connection
    :param fastload_statement: fastload statement that was executed
    :param log_filepath: filepath to save logs in case of warning or errors
    :return:
    """

    # Get current day in string format to log warning or errors
    current_date = get_current_date_str_format('%Y%m%d%H%M%S')

    # Define filepath
    folder = Path(log_filepath)

    # Check warnings
    warning_statement = "{fn teradata_nativesql}{fn teradata_get_warnings}" + fastload_statement
    warn_results = fetch_statement_results(cursor, warning_statement)
    has_warnings = check_statement_results(warn_results, type_result='warning')

    if has_warnings:
        warning_filename = f'fastload_warning_{current_date}.log'
        warning_filepath = folder / warning_filename
        save_string_to_file(warn_results[0][0], warning_filepath)

    # Check errors
    error_statement = "{fn teradata_nativesql}{fn teradata_get_errors}" + fastload_statement
    error_results = fetch_statement_results(cursor, error_statement)
    has_errors = check_statement_results(error_results, type_result='error')

    if has_errors:
        error_filename = f'fastload_error_{current_date}.log'
        error_filepath = folder / error_filename
        save_string_to_file(error_results[0][0], error_filepath)
        cursor.close()

        raise Exception(f'Errors found in fastload check logs in {error_filepath} for more details')


def fastload_dataframe(database_connection: teradatasql.TeradataConnection, df: pd.DataFrame,
                       table_name: str, overwrite_table: bool = False, primary_index: bool = False,
                       primary_index_cols: list = None, log_filepath: str = './') -> None:
    """
    Use fastload utility to upload data to Teradata

    :param database_connection: Teradata database connection
    :param df: dataframe to upload
    :param table_name: table name where data will be uploaded
    :param overwrite_table: overwrite if table_name already exists
    :param primary_index: true if you want to set primary index of the table, False by default
    :param primary_index_cols: columns that will be primary index
    :param log_filepath: filepath to save logs in case of warning or errors
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

    logger.debug(f'Executing {fastload_statement}')
    logger.info('Start Fastload execution')

    try:
        cursor.execute(fastload_statement, df.values.tolist())
    except Exception:
        logger.error("Can't execute teradata fastload successfully", exc_info=True)
        raise

    # Check warning or errors
    logger.info('Fastload execution finished, checking for errors or warnings')
    check_warnings_and_errors(cursor, fastload_statement, log_filepath)

    # Commit results
    logger.info('Committing fastload operation')
    database_connection.commit()

    # Check warning or errors
    logger.info('Fastload operation committed, checking for errors or warnings')
    check_warnings_and_errors(cursor, fastload_statement, log_filepath)

    # Activate autocommit again and close cursor
    cursor.execute('{fn teradata_nativesql}{fn teradata_autocommit_on}')
    cursor.close()

    logger.info('Dataframe successfully upload')
