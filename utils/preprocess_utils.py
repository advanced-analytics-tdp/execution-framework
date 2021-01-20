import pandas as pd
import logging

from sklearn.preprocessing import RobustScaler


logger = logging.getLogger('PREPROCESS')


def normalize_data(data: pd.DataFrame, method: str = 'robust'):
    """
    Normalize features

    :param data: data to be normalized
    :param method: method for scaling
    :return: normalized data
    """

    if method == 'robust':

        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(data)

    else:
        logger.error(f'{method} is not a available method to scale data')
        raise NotImplementedError(f'{method} is not supported by now')

    return scaled_data
