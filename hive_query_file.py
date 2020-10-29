from utils.db_utils import execute_db_statement
from pyhive import hive
import logging
import re


logger = logging.getLogger('HIVE SCRIPT')


class HiveQueryScript:

    def __init__(self, script_path: str, variables: dict = None):
        """
        Constructs all the necessary attributes for the hive query object.

        :param script_path: path to script
        :param variables: variables to execute query
        """
        self.script_path = script_path
        self.variables = variables
        self.raw_scipt = None
        self.raw_queries = None
        self.queries_and_comments = None
        self.variables_status = None

    def read_script(self):
        """ Read script given in script_path """
        with open(self.script_path) as f:
            self.raw_scipt = f.read()

    def split_raw_script(self):
        """ Split raw script in multiples queries """
        self.raw_queries = self.raw_scipt.split(';')

    def separate_queries_and_comments(self):
        """ Separate raw queries into comments and queries to execute """

        result = []

        for raw_query in self.raw_queries:

            # Use regex to find comment and query to execute in each raw query
            comment = re.findall('^-- (.*)', raw_query, flags=re.M)
            query = re.findall(r'^\s*(?:CREATE|INSERT|ALTER|DROP).*', raw_query, flags=re.I | re.M | re.S)

            # Check if comment is found
            if not comment:
                comment = [None]

            # If query is found append to results
            if query:
                result.append([comment[0], query[0]])

        self.queries_and_comments = result

    def get_variables_status(self):
        """ Get if variables were found in script """

        status_variables = dict()

        for key in self.variables:

            status_variables[key] = False

            for raw_query in self.raw_queries:

                if f'${{{key}}}' in raw_query:
                    status_variables[key] = True

        self.variables_status = status_variables

    def check_all_variables_exists(self):
        """ Check if all given variables exists in script """

        self.get_variables_status()

        for var_name, var_status in self.variables_status.items():
            if not var_status:
                raise Exception(f"'{var_name}' variable not found in hive script")

    def replace_variables_in_queries(self):
        """ Replace given variables to execute queries in script """

        for idx, comment_query in enumerate(self.queries_and_comments):

            new_query = comment_query[1]

            for var_name, var_value in self.variables.items():
                new_query = new_query.replace(f"${{{var_name}}}", var_value)

            self.queries_and_comments[idx][1] = new_query

    def execute_script(self, hive_connection: hive.Connection):
        """
        Execute all queries in hive script
        :param hive_connection: hive database connection
        :return:
        """

        self.read_script()
        self.split_raw_script()
        self.separate_queries_and_comments()

        if self.variables is not None:
            self.check_all_variables_exists()
            self.replace_variables_in_queries()

        logger.info('Starting to execute queries in hive script')

        for comment, query in self.queries_and_comments:
            logger.info(comment)
            execute_db_statement(hive_connection, query)

        logger.info('Finished execution of hive query')
