# -*- coding: utf-8 -*-
"""Module with functions to parallelize functions with error handling and progress tracking"""

import logging
import inspect
import math

from typing import Callable, AnyStr, List, Tuple, NamedTuple, Dict, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from time import perf_counter
from collections import OrderedDict, namedtuple
from enum import Enum

import pandas as pd
from more_itertools import chunked, flatten
from tqdm.auto import tqdm as tqdm_auto

from dkulib.io_utils.plugin_io_utils import generate_unique


# ==============================================================================
# CONSTANT DEFINITION
# ==============================================================================

DEFAULT_PARALLEL_WORKERS = 4
DEFAULT_BATCH_SIZE = 10
DEFAULT_BATCH_SUPPORT = False
DEFAULT_VERBOSE = False

API_COLUMN_NAMES_DESCRIPTION_DICT = OrderedDict(
    [
        ("response", "Raw response from the API in JSON format"),
        ("error_message", "Error message from the API"),
        ("error_type", "Error type or code from the API"),
        ("error_raw", "Raw error from the API"),
    ]
)
"""Default dictionary of API column names (key) and their descriptions (value)"""


class ErrorHandling(Enum):
    """Enum class to identify how to handle API errors"""

    LOG = "Log"
    FAIL = "Fail"


# ==============================================================================
# CLASS AND FUNCTION DEFINITION
# ==============================================================================


class BatchError(ValueError):
    """Custom exception raised if the Batch function fails"""


def build_unique_column_names(existing_names: List[AnyStr], column_prefix: AnyStr) -> NamedTuple:
    """Return a named tuple with prefixed API column names and their descriptions"""
    ApiColumnNameTuple = namedtuple("ApiColumnNameTuple", API_COLUMN_NAMES_DESCRIPTION_DICT.keys())
    return ApiColumnNameTuple(
        *[generate_unique(column_name, existing_names, column_prefix) for column_name in ApiColumnNameTuple._fields]
    )


def apply_function_and_parse_response(
    function: Callable,
    output_column_names: NamedTuple,
    exceptions: Union[Exception, Tuple[Exception]],
    row: Dict = None,
    batch: List[Dict] = None,
    batch_response_parser: Callable = None,
    error_handling: ErrorHandling = ErrorHandling.LOG,
    **function_kwargs,
) -> List[Dict]:
    """Wraps a row-by-row or batch function with error handling and response parsing

    It takes the `function` as input and:
    - ensures it has a  `row` parameter which is a dict, or `batch` parameter which is a list of dict
    - If batch, parse the response to extract results and errors using the `batch_response_parser` function
    - handles errors from the function with two methods:
        * (default) log the error message as a warning and return the row with error keys
        * fail if there is an error

    """
    if row and batch:
        raise (ValueError("Please use either row or batch as arguments, but not both"))
    output = deepcopy(row) if row else deepcopy(batch)
    for output_column in output_column_names:
        if row:
            output[output_column] = ""
        else:
            for output_row in output:
                output_row[output_column] = ""
    try:
        response = function(row=row, **function_kwargs) if row else function(batch=batch, **function_kwargs)
        if row:
            output[output_column_names.response] = response
        else:
            output = batch_response_parser(batch=batch, response=response, output_column_names=output_column_names)
            errors = [
                row[output_column_names.error_message] for row in output if row[output_column_names.error_message]
            ]
            if errors:
                raise BatchError(str(errors))
    except exceptions + (BatchError,) as error:
        if error_handling == ErrorHandling.FAIL:
            raise error
        logging.warning(f"Function {function.__name__} failed on: {row if row else batch} because of error: {error}")
        error_type = str(type(error).__qualname__)
        module = inspect.getmodule(error)
        if module:
            error_type = f"{module.__name__}.{error_type}"
        if row:
            output[output_column_names.error_message] = str(error)
            output[output_column_names.error_type] = error_type
            output[output_column_names.error_raw] = str(error.args)
        else:
            for output_row in output:
                output_row[output_column_names.error_message] = str(error)
                output_row[output_column_names.error_type] = error_type
                output_row[output_column_names.error_raw] = str(error.args)
    return output


def convert_results_to_df(
    input_df: pd.DataFrame,
    results: List[Dict],
    output_column_names: NamedTuple,
    error_handling: ErrorHandling = ErrorHandling.LOG,
    verbose: bool = DEFAULT_VERBOSE,
) -> pd.DataFrame:
    """Combine results (list of dict) with input dataframe

    Helper function to the `parallelizer` main function

    """
    output_schema = {**{column_name: str for column_name in output_column_names}, **dict(input_df.dtypes)}
    output_df = (
        pd.DataFrame.from_records(results)
        .reindex(columns=list(input_df.columns) + list(output_column_names))
        .astype(output_schema)
    )
    if not verbose:
        output_df.drop(labels=output_column_names.error_raw, axis=1, inplace=True)
    if error_handling == ErrorHandling.FAIL:
        error_columns = [
            output_column_names.error_message,
            output_column_names.error_type,
            output_column_names.error_raw,
        ]
        output_df.drop(labels=error_columns, axis=1, inplace=True, errors="ignore")
    return output_df


def parallelizer(
    input_df: pd.DataFrame,
    function: Callable,
    exceptions: Union[Exception, Tuple[Exception]],
    column_prefix: AnyStr,
    parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
    batch_support: bool = DEFAULT_BATCH_SUPPORT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    error_handling: ErrorHandling = ErrorHandling.LOG,
    verbose: bool = DEFAULT_VERBOSE,
    **function_kwargs,
) -> pd.DataFrame:
    """Apply a function to a pandas.DataFrame with parallelization, batching, error logging and progress tracking

    The DataFrame is iterated on and passed to the function as dictionaries, row-by-row or by batches of rows.
    This iterative process is accelerated by the use of concurrent threads and is tracked with a progress bar.
    Errors are catched if they match the `exceptions` parameter and automatically logged.
    Once the whole DataFrame has been iterated on, results and errors are added as additional columns.

    Args:
        input_df: Input dataframe which will be iterated on
        function: Function taking a dict as input and returning a dict
            If `function_support_batch` then the function works on list of dict
            For instance, a function to call an API or do some enrichment
        exceptions: Tuple of Exception classes to catch
        column_prefix: Column prefix to add to the output columns for the `function` responses and errors
        parallel_workers: Number of concurrent threads
        batch_support: If True, send batches of row to the `function`
            Else (default) send rows as dict to the function
        batch_size: Number of rows to include in each batch
            Taken into account if `batch_support` is True
        error_handling: If ErrorHandling.LOG (default), log the error message as a warning
            and return the row with error keys.
            Else fail is there is any error.
        verbose: If True, log additional information on errors
            Else (default) log the error message and the error type
        **function_kwargs: Arbitrary keyword arguments passed to the `function`

    Returns:
        Input dataframe with additional columns:
        - response from the `function`
        - error message if any
        - error type if any

    """
    # First, we create a generator expression to yield each row of the input dataframe.
    # Each row will be represented as a dictionary like {"column_name_1": "foo", "column_name_2": 42}
    df_row_generator = (index_series_pair[1].to_dict() for index_series_pair in input_df.iterrows())
    df_num_rows = len(input_df.index)
    start = perf_counter()
    if batch_support:
        logging.info(
            f"Applying function {function.__name__} in parallel to {df_num_rows} row(s)"
            + f" using batch size of {batch_size}..."
        )
        df_row_batch_generator = chunked(df_row_generator, batch_size)
        len_generator = math.ceil(df_num_rows / batch_size)
    else:
        logging.info(f"Applying function {function.__name__} in parallel to {df_num_rows} row(s)...")
        len_generator = df_num_rows
    output_column_names = build_unique_column_names(input_df.columns, column_prefix)
    pool_kwargs = {
        **{
            "function": function,
            "error_handling": error_handling,
            "exceptions": exceptions,
            "output_column_names": output_column_names,
        },
        **function_kwargs.copy(),
    }
    for kwarg in ["fn", "row", "batch"]:  # Reserved pool keyword arguments
        pool_kwargs.pop(kwarg, None)
    if not batch_support and "batch_response_parser" in pool_kwargs:
        pool_kwargs.pop("batch_response_parser", None)
    results = []
    with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
        if batch_support:
            futures = [
                pool.submit(apply_function_and_parse_response, batch=batch, **pool_kwargs)
                for batch in df_row_batch_generator
            ]
        else:
            futures = [
                pool.submit(apply_function_and_parse_response, row=row, **pool_kwargs) for row in df_row_generator
            ]
        for future in tqdm_auto(as_completed(futures), total=len_generator, miniters=1, mininterval=1.0):
            results.append(future.result())
    if batch_support:
        results = flatten(results)
    output_df = convert_results_to_df(input_df, results, output_column_names, error_handling, verbose)
    num_error = sum(output_df[output_column_names.response] == "")
    num_success = len(input_df.index) - num_error
    logging.info(
        (
            f"Applied function in parallel: {num_success} row(s) succeeded, {num_error} failed "
            f"in {(perf_counter() - start):.2f} seconds."
        )
    )
    return output_df
