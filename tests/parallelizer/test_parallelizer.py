# -*- coding: utf-8 -*-
# This is a test file intended to be used with pytest
# pytest automatically runs all the function starting with "test_"
# see https://docs.pytest.org for more information

import json
from typing import AnyStr, Dict, List
from enum import Enum
from urllib.error import URLError

import pytest
import pandas as pd

from dkulib.parallelizer.parallelizer import parallelizer  # noqa


# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================

API_EXCEPTIONS = (URLError, ValueError)
COLUMN_PREFIX = "test_api"
INPUT_COLUMN = "test_case"


class APICaseEnum(Enum):
    SUCCESS = {
        "test_api_response": '{"result": "Great success"}',
        "test_api_error_message": "",
        "test_api_error_type": "",
    }
    INVALID_INPUT = {
        "test_api_response": "",
        "test_api_error_message": "invalid literal for int() with base 10: 'invalid_integer'",
        "test_api_error_type": "ValueError",
    }
    API_FAILURE = {
        "test_api_response": "",
        "test_api_error_message": "<urlopen error foo>",
        "test_api_error_type": "urllib.error.URLError",
    }


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


def call_mock_api(row: Dict, api_function_param: int = 42) -> AnyStr:
    test_case = row.get(INPUT_COLUMN)
    response = {}
    if test_case == APICaseEnum.SUCCESS:
        response = {"result": "Great success"}
    elif test_case == APICaseEnum.INVALID_INPUT:
        try:
            response = {"result": int(api_function_param)}
        except ValueError as e:
            raise e
    elif test_case == APICaseEnum.API_FAILURE:
        raise URLError("foo")
    return json.dumps(response)


def call_mock_api_batch(batch: List[Dict], api_function_param: int = 42) -> AnyStr:
    batch_response = []
    for row in batch:
        test_case = row.get(INPUT_COLUMN)
        response = {}
        if test_case == APICaseEnum.SUCCESS:
            response = {"result": "Great success"}
        elif test_case == APICaseEnum.INVALID_INPUT:
            try:
                response = {"result": int(api_function_param)}
            except ValueError as e:
                raise e
        elif test_case == APICaseEnum.API_FAILURE:
            raise URLError("foo")
        batch_response.append(response)
    return batch_response


def test_api_success():
    """Test the parallelizer logging system in case the mock API function returns successfully"""
    input_df = pd.DataFrame({INPUT_COLUMN: [APICaseEnum.SUCCESS]})
    df = parallelizer(input_df=input_df, function=call_mock_api, exceptions=API_EXCEPTIONS, column_prefix=COLUMN_PREFIX)
    output_dictionary = df.iloc[0, :].to_dict()
    expected_dictionary = APICaseEnum.SUCCESS.value
    for k in expected_dictionary:
        assert output_dictionary[k] == expected_dictionary[k]


def test_api_failure():
    """Test the parallelizer logging system in case the mock API function raises an URLError"""
    input_df = pd.DataFrame({INPUT_COLUMN: [APICaseEnum.API_FAILURE]})
    df = parallelizer(input_df=input_df, function=call_mock_api, exceptions=API_EXCEPTIONS, column_prefix=COLUMN_PREFIX)
    output_dictionary = df.iloc[0, :].to_dict()
    expected_dictionary = APICaseEnum.API_FAILURE.value
    for k in expected_dictionary:
        assert output_dictionary[k] == expected_dictionary[k]


def test_invalid_input():
    """Test the parallelizer logging system in case the mock API function raises a ValueError"""
    input_df = pd.DataFrame({INPUT_COLUMN: [APICaseEnum.INVALID_INPUT]})
    df = parallelizer(
        input_df=input_df,
        function=call_mock_api,
        exceptions=API_EXCEPTIONS,
        column_prefix=COLUMN_PREFIX,
        api_function_param="invalid_integer",
    )
    output_dictionary = df.iloc[0, :].to_dict()
    expected_dictionary = APICaseEnum.INVALID_INPUT.value
    for k in expected_dictionary:
        assert output_dictionary[k] == expected_dictionary[k]


def test_batch_api():
    """Test the parallelizer logging system in batch mode for the three cases above"""
    input_df = pd.DataFrame({INPUT_COLUMN: [APICaseEnum.SUCCESS, APICaseEnum.API_FAILURE, APICaseEnum.INVALID_INPUT]})
    df = parallelizer(
        input_df=input_df,
        function=call_mock_api_batch,
        batch_support=True,
        batch_response_parser=lambda x: x,
        exceptions=API_EXCEPTIONS,
        column_prefix=COLUMN_PREFIX,
        api_function_param="invalid_integer",
    )
    expected_dictionary_list = [
        APICaseEnum.SUCCESS.value,
        APICaseEnum.INVALID_INPUT.value,
        APICaseEnum.INVALID_INPUT.value
    ]
    for i in range(len(input_df.index)):
        output_dictionary = df.iloc[i, :].to_dict()
        expected_dictionary = expected_dictionary_list[i]
        for k in expected_dictionary:
            assert output_dictionary[k] == expected_dictionary[k]
