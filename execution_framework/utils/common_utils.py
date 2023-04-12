from email import encoders
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from pathlib import Path

import pandas as pd
import numpy as np
import datetime
import logging
import yaml
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from typing import List

logger = logging.getLogger('COMMON UTILS')


def get_df_py_dtypes(df: pd.DataFrame) -> dict:
    """
    Get python native data types of columns in pandas dataframe

    :param df: pandas dataframe
    :return: dict with all columns and its data types
    """

    # Initialize dictionary
    metadata_dict = dict()

    # Measurer to get max length of object columns
    measurer = np.vectorize(len)

    # Get first row of dataframe to find out the datatype
    column_names = df.columns
    # values = df.head(1).values.tolist()[0]

    # Iterate over all columns
    # for name, value in zip(column_names, values):
    for name in column_names:
        value = df[name].dropna().values.tolist()[0]
        logger.debug(f'Getting column {name} data type')
        if value is None:
            metadata_dict[name] = ('str', 200)
        if type(value) is str:
            max_column_length = measurer(df[name].dropna().values).max(axis=0)
            if max_column_length == 0:
                max_column_length = 10
            metadata_dict[name] = ('str', max_column_length)
        elif type(value) is int:
            metadata_dict[name] = 'int'
        elif type(value) is float:
            metadata_dict[name] = 'float'
        elif type(value) is datetime.date:
            metadata_dict[name] = 'date'
        else:
            raise NotImplementedError(f'{type(value)} is not supported yet. Data types supporting for now : str, int ,'
                                      f" float and date. Please change '{name}' column datatype or contact developer"
                                      f' to support a new data type.')

    return metadata_dict


def read_configuration_file(file_path: str):
    """
    Read  configuration file in YAML format

    :param file_path: path of configuration file
    :return: configuration parameters
    """
    try:
        with open(file_path) as f:
            conf_file = yaml.load(f, Loader=yaml.FullLoader)
    except Exception:
        logger.error("Can't read variables from YAML file in path '{}'".format(file_path), exc_info=True)
        raise

    return conf_file


def read_variables(file_path: str) -> list:
    """
    Read variables from YAML file

    :param file_path: path of YAML file with names
    :return: list with lowercase variable names
    """

    variables = read_configuration_file(file_path)
    variables = [var.lower() for var in variables]

    return variables


def separate_schema_table(name: str, dbms: str) -> List[str]:
    """
    Separate database or schema from table name in case there is no schema or database function assigns the default
    depending on dbms. dbi_min for Teradata and dev_perm for Hive

    :param dbms: data warehouse name
    :param name: name with table and schema
    :return:
    """
    assert isinstance(name, str), f"name parameter should be a string"

    if name.count('.') > 1:
        raise Exception(f"'{name}' can't contain more than one period")
    elif name.count('.') == 1:
        return name.split('.')
    else:
        if dbms == 'teradata':
            return ['dbi_min', name]
        elif dbms == 'hive':
            return ['dev_perm', name]
        else:
            raise NotImplementedError('Data warehouses supporting by now : hive and teradata')


def save_string_to_file(string: str, filepath: str):
    """
    Save string to plain text file

    :param string: variable to save
    :param filepath: path
    :return:
    """

    text_file = open(filepath, 'w')
    text_file.write(string)
    text_file.close()


def send_mail(to: list, cc: list = [], subject: str = None, body: str = None,
              filename: str = None,
              ):
    """
    Send a mail to some destinatary
    :param to:
    :param cc:
    :param subject:
    :param body:
    :param bodyhtml:
    :param filename: Full path of file to be attached to mail. Formats supported: txt, csv, png, jpg, jpeg
    :return:
    """
    msg = MIMEMultipart()
    password = 'your_password'
    msg['From'] = 'Analitica Avanzada BI <usrewa1@lnxapvlcmo0003.lnx.gp.inet>'
    msg['To'] = ';'.join(to)
    msg['CC'] = ';'.join(cc)
    msg['Subject'] = subject

    message = body
    msg.attach(MIMEText(message, 'html'))

    filepath = Path(filename)
    if not filepath.exists():
        logger.error(f'Attachment file {filepath} does not exist')
        raise Exception(f'Attachment file {filepath} does not exist')

    if filepath.suffix in ['.txt', '.csv']:
        msg.attach(MIMEText(open(filepath).read()))
    elif filepath.suffix in ['.jpg', '.png', '.jpeg']:
        with open(filepath, 'rb') as fp:
            img = MIMEImage(fp.read())
            img.add_header('Content-Disposition', 'attachment', filename=filepath.name)
            msg.attach(img)
    else:
        logger.error(f'Attachment extension {filepath.suffix} not supported')
        raise Exception(f'Attachment extension {filepath.suffix} not supported')


    # create server
    server = smtplib.SMTP('10.226.5.191')
    # server.starttls()

    # Login Credentials for sending the mail
    # server.login(msg['From'], password)
    try:
        # send the message via the server.
        # print(msg['To'])
        toaddrs = to + cc
        server.sendmail(msg['From'], toaddrs, msg.as_string())
        server.quit()
        print("Email sent successfully")
        logger.info("Email sent successfully")
    except smtplib.SMTPException as e:
        print(e)
        logger.error("Can't send email. Check SMTP Configuration")
        raise
    except Exception as e:
        logger.error("Can't send email", exc_info=True)
        raise
