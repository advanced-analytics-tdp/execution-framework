import great_expectations as ge
import pandas as pd
import logging

from pathlib import Path

from great_expectations.core import ExpectationSuiteValidationResult
from great_expectations.render.view import DefaultJinjaPageView
from great_expectations.render.renderer import ValidationResultsPageRenderer

from execution_framework.utils.date_utils import get_current_date_str_format


logger = logging.getLogger('DATA QUALITY')


def _generate_html(obj, renderer, filename):
    document_model = renderer.render(obj)
    with open(filename, "w") as writer:
        writer.write(DefaultJinjaPageView().render(document_model))


def logging_validation_results(results: ExpectationSuiteValidationResult):
    """
    Logging validation results give an expectation suite validation result

    :param results: expectation suite validation result
    :return:
    """

    # Log statistics
    statistics = results.statistics
    logger.info(f"Total evaluated expectations : {statistics['evaluated_expectations']}")
    logger.info(f"Successfull expectations : {statistics['successful_expectations']}")
    logger.info(f"Unsuccessfull expectations : {statistics['unsuccessful_expectations']}")
    logger.info(f"Success percent : {statistics['success_percent']:.2f} %")

    # Log individual results
    # ind_results = results.results
    #
    # for ind_result in ind_results:
    #
    #     expectation_column = ind_result.expectation_config.kwargs['column']
    #     observed_value = ind_result.result['observed_value']
    #     success = ind_result.success
    #
    #     if success:
    #         logger.info(f"Expectation in {expectation_column} was success and its observed value is {observed_value}")
    #     else:
    #         logger.error(f"Expectation in {expectation_column} was a failure and its observed value is"
    #                      f" {observed_value:.5f}, please check the report for more details")


def get_validation_results(df: pd.DataFrame, expectation_suite: str) -> ExpectationSuiteValidationResult:
    """
    Get expectation suite validation result

    :param df: dataframe to get validation results
    :param expectation_suite: path to expectation suite
    :return: Expectation suite validation result
    """

    # Convert pandas dataframe to ge dataframe
    ge_df = ge.from_pandas(df)

    # Getting validation results
    validation_results = ge_df.validate(expectation_suite=expectation_suite, result_format='COMPLETE')

    return validation_results


def validate_data_quality(df: pd.DataFrame, expectation_suite: str, throw_exception: bool, report_path: str,
                          prefix: str):
    """
    Make data validation given a expectation suite

    :param df: dataframe to validate data quality
    :param expectation_suite: path to expectation suite
    :param throw_exception: True throws an exception when result is unsuccessfull
    :param report_path: path to save validation results report
    :param prefix: prefix to add to result name
    :return:
    """

    # Validate data quality
    logger.info(f"Validating data quality with expectation suite in '{expectation_suite}'")
    validation_results = get_validation_results(df, expectation_suite)

    # Logging validation results details
    logging_validation_results(validation_results)

    # Generate report name and path
    current_date = get_current_date_str_format('%Y%m%d_%H%M%S')
    folder = Path(report_path)
    report_filename = f'validation_results_{prefix}_{current_date}.html'
    report_filepath = folder / report_filename

    # Save report
    logger.info(f"Saving validation results report in {report_filepath}")
    _generate_html(validation_results, ValidationResultsPageRenderer(), report_filepath)

    # Throw an error if it's neccessary
    if validation_results.success:
        logger.info("All expectations were successfull")
    else:
        logger.error("Some expectations failed, check validation report for more details")

        if throw_exception:
            raise Exception("Bad data quality, check logs and reports for more details")
