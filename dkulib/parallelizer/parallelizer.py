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
    pass


def build_unique_column_names(existing_names: List[AnyStr], column_prefix: AnyStr) -> NamedTuple:
    """Return a named tuple with prefixed API column names and their descriptions"""
    ApiColumnNameTuple = namedtuple("ApiColumnNameTuple", API_COLUMN_NAMES_DESCRIPTION_DICT.keys())
    api_column_names = ApiColumnNameTuple(
        *[generate_unique(column_name, existing_names, column_prefix) for column_name in ApiColumnNameTuple._fields]
    )
    return api_column_names


def apply_function_to_row(
    function: Callable,
    column_names: NamedTuple,
    row: Dict,
    exceptions: Union[Exception, Tuple[Exception]],
    error_handling: ErrorHandling = ErrorHandling.LOG,
    verbose: bool = DEFAULT_VERBOSE,
    **function_kwargs,
) -> Dict:
    """Wrap a *single-row* function with error handling

    It takes the `function` as input and:
    - ensures it has a `row` parameter which is a dict
    - parses the function response to extract results and errors
    - handles errors from the function with two methods:
        * (default) log the error message as a warning and return the row with error keys
        * fail if there is an error

    """
    output_row = deepcopy(row)
    if error_handling == ErrorHandling.FAIL:
        response = function(row=row, **function_kwargs)
        output_row[column_names.response] = response
    else:
        for column_name in column_names:
            output_row[column_name] = ""
        try:
            response = function(row=row, **function_kwargs)
            output_row[column_names.response] = response
        except exceptions as error:
            logging.warning(f"Function {function.__name__} failed on: {row} because of error: {error}")
            error_type = str(type(error).__qualname__)
            module = inspect.getmodule(error)
            if module is not None:
                error_type = str(module.__name__) + "." + error_type
            output_row[column_names.error_message] = str(error)
            output_row[column_names.error_type] = error_type
            output_row[column_names.error_raw] = str(error.args)
    return output_row


def apply_function_to_batch(
    function: Callable,
    column_names: NamedTuple,
    batch: List[Dict],
    batch_response_parser: Callable,
    exceptions: Union[Exception, Tuple[Exception]],
    error_handling: ErrorHandling = ErrorHandling.LOG,
    verbose: bool = DEFAULT_VERBOSE,
    **function_kwargs,
) -> List[Dict]:
    """Wrap a *batch* function with error handling and response parsing

    It takes the `function` as input and:
    - ensures it has a `batch` parameter which is a list of dict
    - parses the response to extract results and errors using the `batch_response_parser` function
    - handles errors from the function with two methods:
        * (default) log the error message as a warning and return the row with error keys
        * fail if there is an error

    """
    output_batch = deepcopy(batch)
    if error_handling == ErrorHandling.FAIL:
        response = function(batch=batch, **function_kwargs)
        output_batch = batch_response_parser(batch=batch, response=response, column_names=column_names)
        errors = [row[column_names.error_message] for row in batch if row[column_names.error_message] != ""]
        if len(errors) != 0:
            raise BatchError(f"Batch function {function.__name__} failed on: {batch} because of error: {errors}")
    else:
        try:
            response = function(batch=batch, **function_kwargs)
            output_batch = batch_response_parser(batch=batch, response=response, column_names=column_names)
        except exceptions as error:
            logging.warning(f"Batch function {function.__name__} failed on: {batch} because of error: {error}")
            error_type = str(type(error).__qualname__)
            module = inspect.getmodule(error)
            if module is not None:
                error_type = str(module.__name__) + "." + error_type
            for row in output_batch:
                row[column_names.response] = ""
                row[column_names.error_message] = str(error)
                row[column_names.error_type] = error_type
                row[column_names.error_raw] = str(error.args)
    return output_batch


def convert_results_to_df(
    input_df: pd.DataFrame,
    results: List[Dict],
    column_names: NamedTuple,
    error_handling: ErrorHandling = ErrorHandling.LOG,
    verbose: bool = DEFAULT_VERBOSE,
) -> pd.DataFrame:
    """Combine results (list of dict) with input dataframe

    Helper function to the `parallelizer` main function

    """
    if error_handling == ErrorHandling.FAIL:
        columns_to_exclude = [column_name for key, column_name in column_names._asdict().items() if "error" in key]
    else:
        columns_to_exclude = []
        if not verbose:
            columns_to_exclude = [column_names.error_raw]
    output_schema = {**{column_name: str for column_name in column_names}, **dict(input_df.dtypes)}
    output_schema = {
        column_name: schema_type
        for column_name, schema_type in output_schema.items()
        if column_name not in columns_to_exclude
    }
    record_list = [
        {column_name: results.get(column_name) for column_name in output_schema.keys()} for results in results
    ]
    column_list = [column_name for column_name in column_names if column_name not in columns_to_exclude]
    output_column_list = list(input_df.columns) + column_list
    output_df = pd.DataFrame.from_records(record_list).astype(output_schema).reindex(columns=output_column_list)
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
    df_iterator = (index_series_pair[1].to_dict() for index_series_pair in input_df.iterrows())
    len_iterator = len(input_df.index)
    start = perf_counter()
    if batch_support:
        logging.info(
            f"Applying function {function.__name__} in parallel to {len_iterator} row(s)"
            + f" using batch size of {batch_size}..."
        )
        df_iterator = chunked(df_iterator, batch_size)
        len_iterator = math.ceil(len_iterator / batch_size)
    else:
        logging.info(f"Applying function {function.__name__} in parallel to {len_iterator} row(s)...")
    column_names = build_unique_column_names(input_df.columns, column_prefix)
    pool_kwargs = {
        **{
            "function": function,
            "error_handling": error_handling,
            "exceptions": exceptions,
            "column_names": column_names,
        },
        **function_kwargs.copy(),
    }
    for kwarg in ["fn", "row", "batch"]:  # Reserved pool keyword arguments
        pool_kwargs.pop(kwarg, None)
    if not batch_support and "batch_response_parser" in pool_kwargs.keys():
        pool_kwargs.pop("batch_response_parser", None)
    results = []
    with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
        if batch_support:
            futures = [pool.submit(apply_function_to_batch, batch=batch, **pool_kwargs) for batch in df_iterator]
        else:
            futures = [pool.submit(apply_function_to_row, row=row, **pool_kwargs) for row in df_iterator]
        for future in tqdm_auto(as_completed(futures), total=len_iterator):
            results.append(future.result())
    if batch_support:
        results = flatten(results)
    output_df = convert_results_to_df(input_df, results, column_names, error_handling, verbose)
    num_error = sum(output_df[column_names.response] == "")
    num_success = len(input_df.index) - num_error
    logging.info(
        (
            f"Applying function in parallel: {num_success} row(s) succeeded, {num_error} failed "
            f"in {(perf_counter() - start):.2f} seconds."
        )
    )
    return output_df
