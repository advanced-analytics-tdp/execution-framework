from typing import Tuple, Union, List
import pandas as pd
import numpy as np
import logging


logger = logging.getLogger('STATS UTILS')


def compute_interquartile_range(values: str) -> Tuple[float, float]:
    """
    Compute interquartile range for a list of values

    :param values: list of values
    :return: lower and upper bound of interquartile range
    """

    # Calculate IQR
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    return lower_bound, upper_bound


def check_outliers(lower_bound: float, upper_bound: float, current_value: Union[int, float]) -> bool:
    """
    Check if a value is an outliers by comparing it with interquartile range

    :param lower_bound: lower bound of interquartile range
    :param upper_bound: upper bound of interquartile range
    :param current_value: value to compare whether or not it is an outlier
    :return: whether or not it is an outlier
    """

    if lower_bound <= current_value <= upper_bound:
        outlier = False
    else:
        outlier = True

    return outlier


def discretize_data(array: np.ndarray, bin_type: str, q: Union[int, List[Union[int, float]]] = None,
                    bins: Union[int, List[Union[int, float]]] = None, labels: Union[list, np.ndarray] = None,
                    **kwargs) -> pd.Series:
    """
    Discretize variable into buckets based on quantiles or fixed intervals
    For more information about the parameters, please visit :
    - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html
    - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.qcut.html

    :param array: input array to be binned, must be 1-dimensional
    :param bin_type: binning method e.g. quantile or fixed_intervals
    :param q: number of quantiles, just needed if bin_type is quantile
    :param bins: defines the bin edges allowing for non-uniform width, just needed if bin_type is fixed_intervals
    :param labels: used as labels for the resulting bins
    :param kwargs: additional arguments for qcut and cut pandas function
    :return: an array-like object representing the respective bin for each value of x
    """

    if bin_type == 'quantile':

        if q is None:
            raise ValueError("'q' can't be None if bin_type is equal to quantile")

        if labels is None:
            labels = np.arange(q, 0, -1)

        try:
            binned_data = pd.qcut(array, q=q, labels=labels, **kwargs)
        except Exception:
            logger.error("Can't create quantile discrete intervals", exc_info=True)
            raise

    elif bin_type == 'fixed_intervals':

        if bins is None:
            raise ValueError("'bins' can't be None if bin_type is equal to quantile")

        try:
            binned_data = pd.cut(array, bins=bins, labels=labels, **kwargs)
        except Exception:
            logger.error("Can't create fixed discrete intervals", exc_info=True)
            raise

    else:

        logger.error("{} bin type is not supported yet, please contact developer".format(bin_type))
        raise NotImplementedError('Bin types supporting for now: quantile and fixed intervals')

    return binned_data
