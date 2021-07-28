# -*- coding: utf-8 -*-
# This is a test file intended to be used with pytest
# pytest automatically runs all the function starting with "test_"
# see https://docs.pytest.org for more information

import json
from typing import AnyStr, Dict, List, NamedTuple
from copy import deepcopy
from enum import Enum
from urllib.error import URLError

import pytest
import pandas as pd

from core.parallelizer.parallelizer import DataFrameParallelizer, ErrorHandling


# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================

API_EXCEPTIONS = (URLError, ValueError)
INPUT_COLUMN = "test_case"


class APICaseEnum(Enum):
    SUCCESS = {
        "output_response": '{"result": "Great success"}',
        "output_error_message": "",
        "output_error_type": "",
    }
    INVALID_INPUT = {
        "output_response": "",
        "output_error_message": "invalid literal for int() with base 10: 'invalid_integer'",
        "output_error_type": "ValueError",
    }
    API_FAILURE = {
        "output_response": "",
        "output_error_message": "<urlopen error foo>",
        "output_error_type": "urllib.error.URLError",
    }


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


def call_mock_api(row: Dict, api_function_param: int = 42) -> AnyStr:
    """Calls a mock API which works row-by-row"""
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
    """Calls a mock API which works by batches of rows"""
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


def batch_response_parser(batch: List[Dict], response: List, output_column_names: NamedTuple) -> List[Dict]:
    """Parses the response of `call_mock_api_batch` into a clean list of dict"""
    output_batch = deepcopy(batch)
    for i in range(len(response)):
        output_batch[i][output_column_names.response] = json.dumps(response[i]) if response[i] else ""
        output_batch[i][output_column_names.error_message] = ""
        output_batch[i][output_column_names.error_type] = ""
        output_batch[i][output_column_names.error_raw] = ""
    return output_batch


@pytest.mark.parametrize("error_handling", [ErrorHandling.LOG, ErrorHandling.FAIL])
def test_api_success(error_handling):
    """Tests the parallelizer logging system in case the mock API function returns successfully"""
    input_df = pd.DataFrame({INPUT_COLUMN: [APICaseEnum.SUCCESS]})
    parallelizer = DataFrameParallelizer(
        function=call_mock_api, error_handling=error_handling, exceptions_to_catch=API_EXCEPTIONS
    )
    output_df = parallelizer.run(input_df)
    output_dictionary = output_df.iloc[0, :].to_dict()
    if error_handling == error_handling.LOG:
        assert output_df.shape[1] == 4
        expected_dictionary = APICaseEnum.SUCCESS.value
    else:
        assert output_df.shape[1] == 2
        expected_dictionary = {k: v for k, v in APICaseEnum.SUCCESS.value.items() if "error" not in k}
    for k in expected_dictionary:
        assert output_dictionary[k] == expected_dictionary[k]


def test_api_failure():
    """Tests the parallelizer logging system in case the mock API function raises an URLError"""
    input_df = pd.DataFrame({INPUT_COLUMN: [APICaseEnum.API_FAILURE]})
    parallelizer = DataFrameParallelizer(function=call_mock_api, exceptions_to_catch=API_EXCEPTIONS)
    output_df = parallelizer.run(input_df)
    output_dictionary = output_df.iloc[0, :].to_dict()
    expected_dictionary = APICaseEnum.API_FAILURE.value
    for k in expected_dictionary:
        assert output_dictionary[k] == expected_dictionary[k]


def test_invalid_input():
    """Tests the parallelizer logging system in case the mock API function raises a ValueError"""
    input_df = pd.DataFrame({INPUT_COLUMN: [APICaseEnum.INVALID_INPUT]})
    parallelizer = DataFrameParallelizer(function=call_mock_api, exceptions_to_catch=API_EXCEPTIONS)
    output_df = parallelizer.run(input_df, api_function_param="invalid_integer")
    output_dictionary = output_df.iloc[0, :].to_dict()
    expected_dictionary = APICaseEnum.INVALID_INPUT.value
    for k in expected_dictionary:
        assert output_dictionary[k] == expected_dictionary[k]


def test_batch_api():
    """Tests the parallelizer logging system in batch mode for the three cases above"""
    batch_size = 3
    input_df = pd.DataFrame(
        {
            INPUT_COLUMN: batch_size * [APICaseEnum.SUCCESS]
            + batch_size * [APICaseEnum.INVALID_INPUT]
            + batch_size * [APICaseEnum.API_FAILURE]
        }
    )
    parallelizer = DataFrameParallelizer(
        function=call_mock_api_batch,
        exceptions_to_catch=API_EXCEPTIONS,
        batch_support=True,
        batch_size=batch_size,
        batch_response_parser=batch_response_parser,
    )
    output_df = parallelizer.run(input_df, api_function_param="invalid_integer")
    expected_dictionary_list = sorted(
        batch_size * [APICaseEnum.SUCCESS.value]
        + batch_size * [APICaseEnum.INVALID_INPUT.value]
        + batch_size * [APICaseEnum.API_FAILURE.value],
        key=lambda x: x["output_error_type"],
    )
    for i in range(len(input_df.index)):
        output_dictionary = output_df.sort_values(by="output_error_type").iloc[i, :].to_dict()
        expected_dictionary = expected_dictionary_list[i]
        for k in expected_dictionary:
            assert output_dictionary[k] == expected_dictionary[k]
