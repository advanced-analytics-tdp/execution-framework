import datetime as dt
import pandas as pd
import logging
import yaml

from teradatasql import TeradataConnection
from typing import Tuple, Union, Optional
from pyhive import hive

from utils.date_utils import generate_date_range, subtract_units_time, last_day_of_month
from utils.stats_utils import check_outliers, compute_interquartile_range
from schema_validation import validate_data_source_schema
from utils.db_utils import read_query_to_df


logger = logging.getLogger('DATA VALIDATION')


def generate_distinct_query(key_columns: list) -> str:
    """
    Generate distinct part of validation query

    :param key_columns: names of key columns
    :return: distinct string part of validation query
    """

    # Check number of key columns
    if len(key_columns) == 0:
        logger.error("There isn't key columns please enter key columns names")
        raise ValueError('Key columns parameter is empty')
    else:
        logger.debug(f'{key_columns} are the key columns')

    # Create string for distinct part of query
    if len(key_columns) > 1:
        joined_key_columns = ', '.join(key_columns)
        distinct_string = f'concat({joined_key_columns})'
    else:
        distinct_string = key_columns[0]

    distinct_string = f', count(distinct {distinct_string}) total_distinct'

    return distinct_string


def create_cast_date_column_query(dbms: str, date_format: str, date_column: str) -> str:
    """
    Create query that cast date column into yyyy-MM-dd format

    :param dbms: data warehouse name
    :param date_format: format of date column
    :param date_column: name of the column with periods
    :return: query to cast date column
    """

    # Convert date format to yyyy-MM-dd
    if dbms == 'hive':
        if date_format in ['yyyy-MM-dd', 'YYYY-MM-DD']:
            casted_date_column = date_column
        else:
            casted_date_column = f"from_unixtime(unix_timestamp({date_column}, '{date_format}'), 'yyyy-MM-dd')"
    elif dbms == 'teradata':
        if date_format in ['yyyy-MM-dd', 'YYYY-MM-DD']:
            casted_date_column = date_column
        else:
            casted_date_column = f"cast(cast({date_column} as varchar(20)) as date format '{date_format}')"
    else:
        raise NotImplementedError('Data warehouses supporting by now : hive and teradata')

    return casted_date_column


def generate_validation_query(date_column: str, date_format: str, table_name: str, start_date: str, final_date: str,
                              validate_duplicates: bool, key_columns: list, dbms: str) -> str:
    """
    Generates validation query from specific data source

    :param date_column: name of the column with periods
    :param date_format: format of date column
    :param table_name: table name
    :param start_date: start date to filter query in format YYYY-MM-DD
    :param final_date: final date to filter query in format YYYY-MM-DD
    :param validate_duplicates: check or not duplicate values
    :param key_columns: key(s) column of the table
    :param dbms: data warehouse name
    :return: validation query
    """

    # Generate distinct part of query
    distinct_part = generate_distinct_query(key_columns) if validate_duplicates else ''

    # Convert date format to yyyy-MM-dd
    cast_date_column_query = create_cast_date_column_query(dbms, date_format, date_column)

    # Create validation query
    validation_query = f"SELECT {cast_date_column_query} period, count(*) total {distinct_part} " \
                       f"FROM {table_name} " \
                       f"WHERE {cast_date_column_query} >= DATE'{start_date}' " \
                       f"AND {cast_date_column_query} <= DATE'{final_date}' " \
                       f"GROUP BY {cast_date_column_query}"

    # Logging validation query
    logger.debug('Validation query: {}'.format(validation_query))

    return validation_query


def check_complete_results(results: pd.DataFrame, frequency: str, start_date: str, final_date: str,
                           day_month: str) -> bool:
    """
    Check whether or not the results on dataframe are complete

    :param results:  results dataframe with two or three columns (count and count distinct of rows)
    :param frequency: frequency of results dataframe
    :param start_date: start date to check results
    :param final_date: final date to check results
    :param day_month: start or end of month
    :return: True if the results are complete
    """

    data_source_complete = True

    # Generate range of date
    date_range = generate_date_range(start_date, final_date, frequency, day_month)

    # Check if all periods exists
    all_periods = results['period'].tolist()

    # Check difference between what is in data source and what should be there
    difference = list(set(date_range) - set(all_periods))
    difference.sort()

    if len(difference) == 0:
        logger.info('All critical periods were found')
        logger.info(f'{len(date_range)} period(s) were found')

        if len(date_range) == 1:
            logger.info(f'{date_range[0]} period was found in data source')
        else:
            logger.info(f'Periods from {date_range[0]} to {date_range[-1]} were found in data source')

    else:
        data_source_complete = False
        logger.error(f"{', '.join(difference)} period(s) not found in data source")

    return data_source_complete


def check_outliers_results(results: pd.DataFrame, frequency: str, outlier_start_date: str, complete_start_date: str,
                           final_date: str, day_month: str) -> bool:
    """
    Check whether or not there are outliers in the results

    :param results: results dataframe with two or three columns (period, total, total_distinct)
    :param frequency: frequency of results dataframe
    :param outlier_start_date: start date to compute interquartile range
    :param complete_start_date: start date to check outliers in results
    :param final_date: final date to check outliers in results
    :param day_month: start or end of month
    :return: True if there is outliers values in results
    """

    data_source_outliers = True

    # Generate range of dates
    outlier_date_range = generate_date_range(outlier_start_date, final_date, frequency, day_month)
    complete_date_range = generate_date_range(complete_start_date, final_date, frequency, day_month)
    logger.debug(f"Critical dates are {' '.join(complete_date_range)}")

    # Keep only periods to check outliers
    critical_dates = results[results['period'].isin(complete_date_range)].reset_index(drop=True)
    critical_dates_values = critical_dates['total'].values

    # Keep only periods to compute interquartile range
    periods_interquartile_range = results[results['period'].isin(outlier_date_range)].reset_index(drop=True)
    values_interquartile_range = periods_interquartile_range['total'].values

    logger.debug(f'Values to compute interquartile range are {values_interquartile_range}')

    # Create empty list to save outlier status of period
    outlier_tmp = []

    # Get lower and upper bound for values
    lower_bound, upper_bound = compute_interquartile_range(values_interquartile_range)
    logger.info(f'Lower bound for IQR is {lower_bound:,}')
    logger.info(f'Upper bound for IQR is {upper_bound:,}')

    # Search for outlier values in all periods
    for value in critical_dates_values:
        if check_outliers(lower_bound, upper_bound, value):
            outlier_tmp.append('outlier')
        else:
            outlier_tmp.append('normal')

    # Add column with outlier status
    critical_dates['outlier_status'] = outlier_tmp

    # Output results in logging
    if 'outlier' in critical_dates['outlier_status'].values:

        outlier_dates = critical_dates[critical_dates['outlier_status'] == 'outlier'].\
            sort_values(by='period').to_dict(orient='records')

        for date in outlier_dates:
            logger.warning(f"{date['period']} date has {date['total']:,} rows and it's an outlier according IQR")

    else:
        data_source_outliers = False

        for date in critical_dates.to_dict(orient='records'):
            logger.debug(f"{date['period']} date has {date['total']:,} rows and it's a normal according IQR")

        logger.info('All critical periods have normal values')

    return data_source_outliers


def check_duplicate_results(results: pd.DataFrame, frequency: str, start_date: str, final_date: str,
                            day_month: str) -> bool:
    """
    Check whether or not there are duplicates in the results

    :param results: results dataframe with two or three columns (date_column, total, total_distinct)
    :param frequency: frequency of results dataframe
    :param start_date: start date to check duplicates in results
    :param final_date: final date to check duplicates in results
    :param day_month: start or end of month
    :return: True if there is duplicates values in results
    """
    data_source_duplicates = True

    # Generate date range
    date_range = generate_date_range(start_date, final_date, frequency, day_month)

    # Get dates with duplicate values
    results['difference'] = results['total'] - results['total_distinct']
    critical_dates = results[results['period'].isin(date_range)].reset_index(drop=True)
    dates_duplicated = critical_dates[critical_dates['difference'] != 0][['period', 'difference']]
    dates_duplicated = dates_duplicated.sort_values(by='period')

    # Check duplicates values
    if dates_duplicated.empty:
        data_source_duplicates = False
        logger.info('There are not duplicated values in period(s)')
    else:

        for date in dates_duplicated.to_dict(orient='records'):
            logger.warning(f"{date['period']} period has {date['difference']} duplicated values")

        logger.error(f'{dates_duplicated.shape[0]} critical period(s) has duplicated values')

    return data_source_duplicates


def validate_data_source(table_name: str, date_column: str, date_format: str, frequency: str, complete_start_date: str,
                         outliers_start_date: str, final_date: str, day_month: str, validate_duplicates: bool,
                         key_columns: list, dbms: str, db_connection: Union[TeradataConnection, hive.Connection]) \
        -> Tuple[bool, bool, Optional[bool]]:
    """
    Validate if data source is up-to-date to replicate the model

    :param table_name: name of table to validate - string
    :param date_column: name of the column with dates - string
    :param date_format: format of date column - string
    :param frequency: data source update frequency - string
    :param complete_start_date: start date to validate if data source is complete, in format YYYY-MM-DD - string
    :param outliers_start_date: start date to validate if data source has outliers, in format YYYY-MM-DD - string
    :param final_date: final date to validate in format YYYY-MM-DD - string
    :param day_month: start or end of month - string
    :param validate_duplicates: check or not duplicate values - boolean
    :param key_columns: key(s) column of the table - list
    :param dbms: data warehouse name
    :param db_connection: database connection object
    :return: complete, outlier and duplicate status of data source
    """
    # Logging table name and frequency
    logger.info(f"--- TABLE NAME : '{table_name}' ---")
    logger.info(f"Frequency : {frequency}")

    # Get oldest date between start an outlier date
    initial_date = min(dt.datetime.strptime(complete_start_date, '%Y-%m-%d'),
                       dt.datetime.strptime(outliers_start_date, '%Y-%m-%d')).strftime("%Y-%m-%d")

    # Generates validation query
    validation_query = generate_validation_query(date_column, date_format, table_name, initial_date, final_date,
                                                 validate_duplicates, key_columns, dbms)

    # Execute query to proceed with validation
    logger.info("Start execution of validation query")
    results = read_query_to_df(db_connection, validation_query)
    results['period'] = results['period'].astype(str)

    # Make validations in data source
    complete_status = check_complete_results(results, frequency, complete_start_date, final_date, day_month)
    outlier_status = check_outliers_results(results, frequency, outliers_start_date, complete_start_date, final_date,
                                            day_month)

    if validate_duplicates:
        duplicates_status = check_duplicate_results(results, frequency, complete_start_date, final_date, day_month)
    else:
        duplicates_status = None

    return complete_status, outlier_status, duplicates_status


def generate_start_final_dates(frequency: str, day_month: str, replica_date: str, periods_needed_replica: int,
                               periods_check_outliers: int) -> Tuple[str, str, str]:
    """
    Generate start date to check complete data source, start date to check outliers and final date to execute
    validation query

    :param frequency: data source update frequency
    :param day_month: start or end of month
    :param replica_date: replica month in format 'YYYY-MM-DD'
    :param periods_needed_replica: periods needed to execute replica
    :param periods_check_outliers: periods needed to check outliers
    :return:
    """

    if frequency == 'monthly':

        if day_month == 'first':
            final_date = replica_date

        elif day_month == 'last':
            final_date = last_day_of_month(replica_date)

        else:
            raise NotImplementedError('Day months supporting for now : first and last')

        complete_start_date = subtract_units_time(final_date, 'months', periods_needed_replica)
        outliers_start_date = subtract_units_time(final_date, 'months', periods_check_outliers)

    elif frequency == 'daily':
        final_date = last_day_of_month(replica_date)
        complete_start_date = subtract_units_time(final_date, 'days', periods_needed_replica)
        outliers_start_date = subtract_units_time(final_date, 'days', periods_check_outliers)

    else:
        raise NotImplementedError('Frequencies supporting for now: monthly and daily')

    return complete_start_date, outliers_start_date, final_date


def check_all_data_sources(data_sources_conf_file: str, replica_date: str,
                           teradata_connection: TeradataConnection = None,
                           hive_connection: hive.Connection = None) -> pd.DataFrame:
    """
    Check multiple data sources

    :param data_sources_conf_file: path of yaml file with all data source attributes
    :param replica_date: date of replica to validate data source in 'YYYY-MM-DD' format
    :param teradata_connection: teradata connection object
    :param hive_connection: hive connection object
    :return: summary of quality status of data sources
    """

    # Check at least one database connection is provided
    if teradata_connection is None and hive_connection is None:
        raise ValueError('Provide at least one database connection : Teradata or Hive')

    # Read data source yaml file
    try:
        with open(data_sources_conf_file) as f:
            data_sources = yaml.load(f, Loader=yaml.FullLoader)
    except Exception:
        logger.error(f"Can't read yaml file in '{data_sources_conf_file}'", exc_info=True)
        raise

    # Validate values of json file
    schema_status = validate_data_source_schema(data_sources)

    if schema_status:
        logger.debug(f'Correct schema in all data sources in {data_sources_conf_file} file')
    else:
        raise ValueError(f'Some data sources schemas are wrong, check conf file in {data_sources_conf_file}')

    # Create list for warning and complete status
    data_source_status_summary = []

    for source_name, source_details in data_sources.items():

        table_name = source_details['table_name']
        dbms = source_details['dbms']
        date_column = source_details['date_column']
        key_columns = source_details['key_columns']
        date_format = source_details['date_format']
        frequency = source_details['frequency']
        day_month = source_details.get('day_month', 'first')
        validate_duplicates = source_details['validate_duplicates']
        periods_needed_replica = source_details['periods_needed_replica'] - 1
        periods_check_outliers = source_details['periods_check_outliers'] - 1

        # Create db_connection depending on data_sources
        if dbms == 'hive':
            db_connection = hive_connection
        elif dbms == 'teradata':
            db_connection = teradata_connection
        else:
            raise NotImplementedError('Data warehouses supporting by now : hive and teradata')

        # Get start and final date to execute validation query
        complete_start_date, outliers_start_date, final_date = generate_start_final_dates(frequency, day_month,
                                                                                          replica_date,
                                                                                          periods_needed_replica,
                                                                                          periods_check_outliers)

        # Verify data sources status
        complete_status, outlier_status, duplicates_status = validate_data_source(table_name, date_column, date_format,
                                                                                  frequency, complete_start_date,
                                                                                  outliers_start_date, final_date,
                                                                                  day_month, validate_duplicates,
                                                                                  key_columns, dbms, db_connection)

        data_source_status_summary.append([table_name, complete_status, outlier_status, duplicates_status])

    # Save data source status summary to dataframe
    df_status_summary = pd.DataFrame(data_source_status_summary, columns=['table_name', 'complete_status',
                                                                          'outlier_status', 'duplicates_status'])

    return df_status_summary
