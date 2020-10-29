import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import numpy as np

import yaml
import pickle
import logging

from utils.common_utils import read_variables
from utils.stats_utils import discretize_data
from typing import Union


logger = logging.getLogger('MODEL PREDICTION')


def load_model(trained_model_path: str, model_estimator: str) -> Union[xgb.Booster, lgb.LGBMClassifier]:
    """
    Read trained object model from disk

    :param trained_model_path: path of trained model
    :param model_estimator: estimator of model. e.g. lightgbm, xgboost
    :return: model object
    """

    logger.info("Loading trained model object from '{}'".format(trained_model_path))

    try:

        if model_estimator == 'lightgbm':
            with open(trained_model_path, 'rb') as model:
                trained_model = pickle.load(model)

        elif model_estimator == 'xgboost':
            trained_model = xgb.Booster(model_file=trained_model_path)

        else:
            logger.error(f"{model_estimator} model is not supported yet, please contact developer")
            raise NotImplementedError('Model types supporting for now: lightgbm and xgboost')

    except Exception:
        logger.error(f"Can't load trained model file from '{trained_model_path}'", exc_info=True)
        raise

    return trained_model


def prepare_df_to_predict(data: pd.DataFrame, key_columns: list, model_columns: list = None) -> pd.DataFrame:
    """
    Prepare dataframe with necessary columns and filters to replicate the model

    :param data: dataframe with all variables
    :param key_columns: identifiers like id's that aren't necessary to replicate model
    :param model_columns: column names in the same order in which the model was trained needed to predict
    :return: dataframe prepared to replicate model
    """

    logger.debug('Convert dataframe columns and key columns to lowercase')
    data.columns = map(str.lower, data.columns)
    key_columns = [c.lower() for c in key_columns]

    logger.info('Selecting necessary columns to replicate model')

    if model_columns is None:

        logger.debug(f"Key columns are {', '.join(key_columns)}")
        logger.debug('Columns to replicate model are : All columns except key columns')

        try:
            replica_data = data.drop(key_columns, axis=1)
        except Exception:
            logger.error("Can't drop key columns or identifiers from data", exc_info=True)
            raise
    else:

        model_columns = [c.lower() for c in model_columns]
        logger.debug(f'Columns to replicate model are : {model_columns}')

        try:
            replica_data = data[model_columns].copy()
        except Exception:
            logger.error("Can't select models columns from data", exc_info=True)
            raise

    return replica_data


def predict_model(trained_model: Union[xgb.Booster, lgb.LGBMClassifier], model_estimator: str,
                  data: pd.DataFrame) -> np.ndarray:
    """
    Predict using trained model

    :param trained_model: model object
    :param model_estimator: model type e.g. xgboost or lightgbm
    :param data: dataframe with variables to replicate model
    :return: probabilities resulting from the prediction
    """

    logger.info('Predicting probabilities for the new data')

    try:

        if model_estimator == 'lightgbm':
            probabilities = trained_model.predict_proba(data)
            probabilities = probabilities[:, 1]

        elif model_estimator == 'xgboost':
            replica_data_dmatrix = xgb.DMatrix(data)
            probabilities = trained_model.predict(replica_data_dmatrix)

        else:
            logger.error(f"{model_estimator} model is not supported yet, please contact developer")
            raise NotImplementedError('Model types supporting for now: lightgbm and xgboost')

    except Exception:
        logger.error("Can't predict model in new data, please check data quality", exc_info=True)
        raise

    logger.info('Model prediction finished')

    return probabilities


def common_steps_replica(input_samples: pd.DataFrame, key_columns: list, trained_model_path: str, model_estimator: str,
                         model_columns_path: str = None) -> np.ndarray:
    """
    Execute common steps to make single model replica and ensemble model replica

    :param input_samples: input samples with all necessary variables to replicate model
    :param key_columns: identifiers like id's that aren't necessary to replicate model
    :param trained_model_path: path of trained model
    :param model_estimator: estimator of model. e.g. lightgbm, xgboost
    :param model_columns_path: column names in the same order in which the model was trained
    :return:
    """
    # Load trained file model
    trained_model = load_model(trained_model_path, model_estimator)

    # Read variables to replicate the model
    model_columns = read_variables(model_columns_path)

    # Selecting correct columns to replicate model
    replica_data = prepare_df_to_predict(input_samples, key_columns, model_columns)

    # Predict model in new data
    probabilities = predict_model(trained_model, model_estimator, replica_data)

    return probabilities


def single_model_replica(data: pd.DataFrame, key_columns: list, trained_model_path: str, model_estimator: str,
                         filters: str = None, model_columns_path: str = None, add_group_column: bool = True,
                         group_columns_type: str = 'quantile', quantile: int = 10, fixed_intervals: list = None,
                         labels: Union[list, np.ndarray] = None) -> pd.DataFrame:
    """
    Model prediction for all samples in replica data
    If model_columns is not specified it takes all columns except key_columns to replicate model

    :param data: all necessary variables to replicate model
    :param key_columns: identifiers like id's that aren't necessary to replicate model
    :param trained_model_path: path of trained model
    :param model_estimator: estimator of model. e.g. lightgbm, xgboost
    :param filters: filters: filters to query data
    :param model_columns_path: column names in the same order in which the model was trained
    :param add_group_column: add rank column based on probabilities
    :param group_columns_type: type of group column e.g. quantile or fixed intervals
    :param quantile: number of quantiles. 10 for deciles, 4 for quartiles. If None don't create new column
    :param fixed_intervals: edges to create column with groups based on probabilities. If None don't create new column
    :param labels: used as labels for the resulting column
    :return: dataframe with probabilities and key columns
    """

    # Filter dataframe if it's necessary
    if filters is not None:
        logger.info(f'Applying filters to dataframe : {filters}')
        filtered_data = data.query(filters).reset_index(drop=True)
        logger.debug(f'New dataframe shape is {filtered_data.shape}')
    else:
        filtered_data = data

    # Keep only key columns
    replica_result = filtered_data[key_columns].copy()

    # Execute common steps replica
    probabilities = common_steps_replica(input_samples=filtered_data,
                                         key_columns=key_columns,
                                         trained_model_path=trained_model_path,
                                         model_estimator=model_estimator,
                                         model_columns_path=model_columns_path)

    # Add replica column to results
    replica_result['final_prob'] = probabilities

    # Add group column
    if add_group_column:
        logger.info(f'Add group column type {group_columns_type}')
        groups = discretize_data(probabilities, group_columns_type, quantile, fixed_intervals, labels)
        replica_result['groups'] = groups

    return replica_result


def ensemble_model_replica(data: pd.DataFrame, models_data: dict, key_columns: list) -> pd.DataFrame:
    """
    Generate prediction for ensemble models

    :param data: dataframe with all necessary variables to replicate each model
    :param models_data: model path, variables path, model type and weight of each model
    :param key_columns: identifiers like id's that aren't necessary to replicate model
    :return: dataframe with probabilities for each model and final probability
    """

    # Filter dataframe if it's necessary
    if models_data.get('filter_rows') is not None:

        filters = models_data['filter_rows']['query']

        logger.info(f'Applying filters to dataframe : {filters}')
        filtered_data = data.query(filters).reset_index(drop=True)
        logger.debug(f'New dataframe shape is {filtered_data.shape}')
    else:
        filtered_data = data

    # Create dataframe to add all the probabilities as columns
    ensemble_replica_result = filtered_data[key_columns].reset_index(drop=True)
    ensemble_probabilities = np.zeros(ensemble_replica_result.shape[0])

    # Iterate over through all models
    for model_name, model_data in models_data['inner_models'].items():

        logger.info(f'Making replica of {model_name} model')

        # Execute common steps replica
        probabilities = common_steps_replica(input_samples=filtered_data,
                                             key_columns=key_columns,
                                             trained_model_path=model_data['model_path'],
                                             model_estimator=model_data['model_estimator'],
                                             model_columns_path=model_data['model_variables'])

        # Add prediction as a column
        prob_column_name = 'prob_' + model_name
        ensemble_replica_result[prob_column_name] = probabilities

        # Create ensemble probabilities
        ensemble_probabilities += probabilities * model_data['model_weight']

    # Add column with final probability
    ensemble_replica_result['final_prob'] = ensemble_probabilities

    # Add group column
    if models_data['add_group_column']:
        groups = discretize_data(array=ensemble_probabilities,
                                 q=models_data.get('quantile'),
                                 bin_type=models_data.get('group_column_type'),
                                 bins=models_data.get('probability_cuts'),
                                 labels=models_data.get('labels'))
        ensemble_replica_result['groups'] = groups

    return ensemble_replica_result


def replicate_all_models(data: pd.DataFrame, key_columns: list, conf_replica_models_path: str) -> pd.DataFrame:
    """
    Replicate all models in configuration models file

    :param data: dataframe with all necessary variables to replicate all models
    :param key_columns: identifiers like id's that aren't necessary to replicate models
    :param conf_replica_models_path: path to replica configuration file with all models parameters
    :return: dataframe with the union of all replicas
    """

    # Read configuration parameters from yaml file
    with open(conf_replica_models_path) as f:
        conf_replica_models = yaml.load(f, Loader=yaml.FullLoader)

    # Dictionary with all results of models
    results = dict()

    # Replicate all models
    for model_name, model_data in conf_replica_models.items():

        logger.info(f'Starting replica of {model_name} model')

        if model_data['ensemble_model']:
            ensemble_results = ensemble_model_replica(data=data,
                                                      models_data=model_data,
                                                      key_columns=key_columns)
            results[model_name] = ensemble_results[key_columns + ['final_prob', 'groups']].copy()
        else:
            results[model_name] = single_model_replica(data=data,
                                                       key_columns=key_columns,
                                                       trained_model_path=model_data['model_path'],
                                                       model_estimator=model_data['model_estimator'],
                                                       filters=model_data.get('filter_rows', {}).get('query'),
                                                       model_columns_path=model_data['model_variables'],
                                                       add_group_column=model_data.get('add_group_column'),
                                                       group_columns_type=model_data.get('group_column_type'),
                                                       quantile=model_data.get('quantile'),
                                                       fixed_intervals=model_data.get('probability_cuts'),
                                                       labels=model_data.get('labels'))

    logger.info('Replication of all models finished ')

    # Union all dataframe
    logger.info('Union all dataframes with results')
    total_replica_results = pd.concat(list(results.values())).reset_index(drop=True)

    return total_replica_results
