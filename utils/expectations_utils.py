from great_expectations.dataset.pandas_dataset import PandasDataset
from great_expectations.dataset import util

from pandas import DataFrame
from typing import Union

import logging


logger = logging.getLogger('DATA QUALITY')


def fix_continuous_partition_object(partition_object: dict) -> dict:
    """
    Fix continuous partition object to be valid

    :param partition_object: The partition_object to fix
    :return: fixed partition object
    """

    # Fix bins
    bins = partition_object['bins']
    fixed_bins = sorted(list(set(bins)))

    # Fix weights
    weights = partition_object['weights']
    fixed_weights = [w for w in weights if w != 0]

    # Fix partition object
    fixed_partition_object = {'bins': fixed_bins, 'weights': fixed_weights}

    # Verify is partition object is fixed
    if util.is_valid_continuous_partition_object(fixed_partition_object):
        logger.info('Partition object was fixed')
    else:
        logger.error("Partition object couldn't be fixed")

    return fixed_partition_object


def create_partition_objects(data: Union[PandasDataset, DataFrame], key_columns: list, bins: int,
                             cat_cols: list) -> dict:
    """
    Create partition objects for all columns of dataframe to validate data source

    :param data: dataframe
    :param key_columns: key column_name names of dataframe
    :param bins: number of bins
    :param cat_cols: categorical column_name names
    :return: partition objects for every column_name
    """

    # Create empty dict to store all partition objects
    partitions = dict()

    for column_name in data.columns:

        if column_name in key_columns:
            logger.info(f'Ignoring key column {column_name}')

        elif column_name in cat_cols:

            logger.info(f'Getting partition object of categorical column {column_name}')
            partition_object = util.build_categorical_partition_object(data, column_name)
            partitions[column_name] = partition_object
        else:

            logger.info(f'Getting partition object of numerical column {column_name}')
            partition_object = util.build_continuous_partition_object(data, column_name, bins='ntile', n_bins=bins)

            # Check all the weight is not in single interval
            if 1 in partition_object['weights']:
                partition_object = util.build_continuous_partition_object(data, column_name, bins='ntile', n_bins=100)

            # If it has tail_weights removing it
            partition_object = {'bins': partition_object['bins'], 'weights': partition_object['weights']}

            # Check if partition object is correct
            correct = util.is_valid_continuous_partition_object(partition_object)

            if not correct:
                logger.info(f'{column_name} has an invalid partition object, fixing it')
                partition_object = fix_continuous_partition_object(partition_object)

            partitions[column_name] = partition_object

    return partitions


def make_distributional_expectations(data: PandasDataset, key_columns: list, partitions: dict):
    """
    Make distributional expectation for all columns in dataframe using kl divergence

    :param data: dataframe to make distributional expectations
    :param key_columns: hey column names to skip in expectations
    :param partitions: partition object for all columns
    :return:
    """

    for column_name in data.columns:

        if column_name in key_columns:
            logger.info(f'Ignoring key column {column_name}')

        else:

            logger.info(f"Making distributional expectation of {column_name}")

            result = data.expect_column_kl_divergence_to_be_less_than(column=column_name,
                                                                      partition_object=partitions[column_name],
                                                                      threshold=0.05,
                                                                      tail_weight_holdout=0.000001,
                                                                      result_format='COMPLETE')

            # Log validation results
            logger.info(f"Validation success is {result.success} and observed kl divergence "
                        f"is {result.result['observed_value']}")


def make_data_type_expectations(data: PandasDataset):
    """
    Make data type expectations for all column in dataset

    :param data: dataframe to make data type expectations
    :return:
    """

    # Iterate over all columns
    for column_name in data.columns:

        logger.info(f"Making data type expectation of {column_name}")

        result = data.expect_column_values_to_be_of_type(column=column_name, type_=data[column_name].dtype.name)

        # Log validation results
        logger.info(f"Validation success is {result.success} and observed value is {result.result['observed_value']}")
