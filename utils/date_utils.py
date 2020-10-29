import datetime as dt
import dateutil.relativedelta
import pandas as pd


def subtract_units_time(date: str, unit_time: str, time_subtracted: int):
    """
    Subtract units of time from date

    :param date: date in string format YYYY-MM-DD
    :param unit_time: specify whether to subtract days or month
    :param time_subtracted: amount of days or months to subtract
    :return: date after subtract days or months
    """
    d = dt.datetime.strptime(date, "%Y-%m-%d")

    if unit_time == 'months':
        subtracted_date = d - dateutil.relativedelta.relativedelta(months=time_subtracted)
    elif unit_time == 'days':
        subtracted_date = d - dateutil.relativedelta.relativedelta(days=time_subtracted)
    else:
        raise NotImplementedError('Unit time supporting for now: months and days')

    # Convert into string
    subtracted_date = subtracted_date.strftime("%Y-%m-%d")

    return subtracted_date


def last_day_of_month(date: str) -> str:
    """
    Given a date get the last day of the month

    :param date: date in string format YYYY-MM-DD
    :return: last day of the month in string format YYYY-MM-DD
    """
    # Convert into date format
    date = dt.datetime.strptime(date, "%Y-%m-%d")

    # Find last day of month
    next_month = date.replace(day=28) + dt.timedelta(days=4)
    last_day = next_month - dt.timedelta(days=next_month.day)

    # Convert into string
    last_day = last_day.strftime("%Y-%m-%d")

    return last_day


def generate_date_range(start_date: str, final_date: str, frequency: str, day_month: str = 'first') -> list:
    """
    Generate a range of dates from date1 to date2

    :param start_date: start date in format YYYY-MM-DD
    :param final_date: start date in format YYYY-MM-DD
    :param frequency: frequency of dates
    :param day_month: start or end of month
    :return: list of dates in string format
    """

    if frequency == 'monthly':

        if day_month == 'first':
            range_dt = pd.date_range(start_date, final_date, freq='MS').strftime("%Y-%m-%d").tolist()
        elif day_month == 'last':
            range_dt = pd.date_range(start_date, final_date, freq='M').strftime("%Y-%m-%d").tolist()
        else:
            raise NotImplementedError('Day months types supporting for now: first and last')

    elif frequency == 'daily':
        range_dt = pd.date_range(start_date, final_date, freq='D').strftime("%Y-%m-%d").tolist()
    else:
        raise NotImplementedError('Frequencies supporting for now: monthly and daily')

    return range_dt
