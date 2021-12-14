from cerberus import Validator
import logging


logger = logging.getLogger('SCHEMA VALIDATION')


# Define schema
data_source_schema = {
    'table_name': {'required': True, 'type': 'string'},
    'dbms': {'required': True, 'type': 'string', 'allowed': ['hive', 'teradata']},
    'date_column': {'required': True, 'type': 'string'},
    'key_columns': {'required': True, 'type': 'list', 'schema': {'type': 'string'}},
    'date_format': {'required': True, 'type': 'string', 'allowed': ['YYYY-MM-DD', 'YYYYMM']},
    'frequency': {'required': True, 'type': 'string',
                  'oneof': [
                      {'excludes': 'day_month', 'allowed': ['daily']},
                      {'dependencies': 'day_month', 'allowed': ['monthly']}
                  ]
                  },
    'day_month': {'type': 'string', 'allowed': ['first', 'last']},
    'validate_duplicates': {'required': True, 'type': 'boolean'},
    'periods_needed_replica': {'required': True, 'type': 'integer', 'min': 1},
    'periods_check_outliers': {'required': True, 'type': 'integer', 'min': 4}
}


def validate_data_source_schema(data_sources: dict) -> bool:
    """
    Validate if data source has the correct structure

    :param data_sources: schema of all data sources
    :return: True if all data sources are okay, otherwise false
    """

    # Generate validator
    v = Validator()
    v.schema = data_source_schema

    # Create yaml_file_status
    yaml_file_status = True

    # Validate all data sources
    for name, description in data_sources.items():

        if not v.validate(description):
            fields_errors = list(v.errors)
            for field in fields_errors:
                logger.error("'{datasource_name}' data source schema has error(s). Field : '{field_name}'"
                             " Message : '{error_message}'".format(datasource_name=name,
                                                                   field_name=field,
                                                                   error_message=v.errors[field][0]))
            yaml_file_status = False
        else:
            logger.debug("'{datasource_name}' data source schema is correct".format(datasource_name=name))

    return yaml_file_status
