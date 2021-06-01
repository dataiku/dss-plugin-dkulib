# -*- coding: utf-8 -*-
"""Apply a function to a pandas DataFrame with parallelization, error handling and progress tracking"""

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


class ErrorHandling(Enum):
    """Enum class to identify how to handle API errors"""

    LOG = "Log"
    FAIL = "Fail"


class BatchError(ValueError):
    """Custom exception raised if the Batch function fails"""


class DataFrameParallelizer:
    """Apply a function to a pandas DataFrame with parallelization, error handling and progress tracking

    Attributes:
        error_handling: If ErrorHandling.LOG (default), log the error message as a warning
            and return the row with error keys. Else fail is there is any error.
        exceptions_to_catch: Tuple of Exception classes to catch. Mandatory if ErrorHandling.LOG (default).
        parallel_workers: Number of concurrent threads to parallelize the function. Default is 4.
        batch_support: If True, send batches of row to the `function`
            Else (default) send rows as dict to the function
        batch_size: Number of rows to include in each batch. Default is 10.
            Taken into account if `batch_support` is True
        output_column_prefix: Column prefix to add to the output columns for the `function` responses and errors.
            Default is "output".
        verbose: If True, log additional information on errors
            Else (default) log the error message and the error type
    """

    DEFAULT_PARALLEL_WORKERS = 4
    DEFAULT_BATCH_SIZE = 10
    DEFAULT_BATCH_SUPPORT = False
    DEFAULT_VERBOSE = False
    DEFAULT_OUTPUT_COLUMN_PREFIX = "output"
    API_COLUMN_NAMES_DESCRIPTION_DICT = OrderedDict(
        [
            ("response", "Raw response from the API in JSON format"),
            ("error_message", "Error message from the API"),
            ("error_type", "Error type or code from the API"),
            ("error_raw", "Raw error from the API"),
        ]
    )
    """Default dictionary of API column names (key) and their descriptions (value)"""

    def __init__(
        self,
        error_handling: ErrorHandling = ErrorHandling.LOG,
        exceptions_to_catch: Tuple[Exception] = (),
        parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
        batch_support: bool = DEFAULT_BATCH_SUPPORT,
        batch_size: int = DEFAULT_BATCH_SIZE,
        batch_response_parser: Callable = None,
        output_column_prefix: AnyStr = DEFAULT_OUTPUT_COLUMN_PREFIX,
        verbose: bool = DEFAULT_VERBOSE,
    ):
        self.error_handling = error_handling
        self.exceptions_to_catch = exceptions_to_catch
        if self.error_handling == ErrorHandling.LOG and not self.exceptions_to_catch:
            raise ValueError("Please set at least one exception in exceptions_to_catch")
        self.parallel_workers = parallel_workers
        self.batch_support = batch_support
        self.batch_size = batch_size
        self.batch_response_parser = batch_response_parser
        self.output_column_prefix = output_column_prefix
        self.verbose = verbose

    def _build_unique_column_names(self, existing_names: List[AnyStr]) -> NamedTuple:
        """Return a named tuple with prefixed API column names and their descriptions"""
        ApiColumnNameTuple = namedtuple("ApiColumnNameTuple", self.API_COLUMN_NAMES_DESCRIPTION_DICT.keys())
        return ApiColumnNameTuple(
            *[
                generate_unique(column_name, existing_names, self.output_column_prefix)
                for column_name in ApiColumnNameTuple._fields
            ]
        )

    def _apply_function_and_parse_response(
        self,
        function: Callable,
        output_column_names: NamedTuple,
        row: Dict = None,
        batch: List[Dict] = None,
        **function_kwargs,
    ) -> List[Dict]:  # sourcery skip: or-if-exp-identity
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
                output = self.batch_response_parser(
                    batch=batch, response=response, output_column_names=output_column_names
                )
                errors = [
                    row[output_column_names.error_message] for row in output if row[output_column_names.error_message]
                ]
                if errors:
                    raise BatchError(str(errors))
        except self.exceptions_to_catch + (BatchError,) as error:
            if self.error_handling == ErrorHandling.FAIL:
                raise error
            logging.warning(
                f"Function {function.__name__} failed on: {row if row else batch} because of error: {error}"
            )
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

    def _convert_results_to_df(
        self, df: pd.DataFrame, results: List[Dict], output_column_names: NamedTuple,
    ) -> pd.DataFrame:
        """Combine results (list of dict) with input dataframe

        Helper function to the `parallelizer` main function

        """
        output_schema = {**{column_name: str for column_name in output_column_names}, **dict(df.dtypes)}
        output_df = (
            pd.DataFrame.from_records(results)
            .reindex(columns=list(df.columns) + list(output_column_names))
            .astype(output_schema)
        )
        if not self.verbose:
            output_df.drop(labels=output_column_names.error_raw, axis=1, inplace=True)
        if self.error_handling == ErrorHandling.FAIL:
            error_columns = [
                output_column_names.error_message,
                output_column_names.error_type,
                output_column_names.error_raw,
            ]
            output_df.drop(labels=error_columns, axis=1, inplace=True, errors="ignore")
        return output_df

    def apply(self, df: pd.DataFrame, function: Callable, **function_kwargs,) -> pd.DataFrame:
        """Apply a function to a pandas.DataFrame with parallelization, batching, error logging and progress tracking

        The DataFrame is iterated on and passed to the function as dictionaries, row-by-row or by batches of rows.
        This iterative process is accelerated by the use of concurrent threads and is tracked with a progress bar.
        Errors are catched if they match the `exceptions` parameter and automatically logged.
        Once the whole DataFrame has been iterated on, results and errors are added as additional columns.

        Args:
            df: Input dataframe which will be iterated on
            function: Function taking a dict as input and returning a dict
                If `function_support_batch` then the function works on list of dict
                For instance, a function to call an API or do some enrichment
            **function_kwargs: Arbitrary keyword arguments passed to the `function`

        Returns:
            Input dataframe with additional columns:
            - response from the `function`
            - error message if any
            - error type if any

        """
        # First, we create a generator expression to yield each row of the input dataframe.
        # Each row will be represented as a dictionary like {"column_name_1": "foo", "column_name_2": 42}
        df_row_generator = (index_series_pair[1].to_dict() for index_series_pair in df.iterrows())
        df_num_rows = len(df.index)
        start = perf_counter()
        if self.batch_support:
            logging.info(
                f"Applying function {function.__name__} in parallel to {df_num_rows} row(s)"
                + f" using batch size of {self.batch_size}..."
            )
            df_row_batch_generator = chunked(df_row_generator, self.batch_size)
            len_generator = math.ceil(df_num_rows / self.batch_size)
        else:
            logging.info(f"Applying function {function.__name__} in parallel to {df_num_rows} row(s)...")
            len_generator = df_num_rows
        output_column_names = self._build_unique_column_names(existing_names=df.columns)
        pool_kwargs = {**{"output_column_names": output_column_names}, **function_kwargs.copy()}
        for kwarg in ["function", "row", "batch"]:  # Reserved pool keyword arguments
            pool_kwargs.pop(kwarg, None)
        if not self.batch_support and "batch_response_parser" in pool_kwargs:
            pool_kwargs.pop("batch_response_parser", None)
        results = []
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as pool:
            if self.batch_support:
                futures = [
                    pool.submit(self._apply_function_and_parse_response, function=function, batch=batch, **pool_kwargs)
                    for batch in df_row_batch_generator
                ]
            else:
                futures = [
                    pool.submit(self._apply_function_and_parse_response, function=function, row=row, **pool_kwargs)
                    for row in df_row_generator
                ]
            for future in tqdm_auto(as_completed(futures), total=len_generator, miniters=1, mininterval=1.0):
                results.append(future.result())
        if self.batch_support:
            results = flatten(results)
        output_df = self._convert_results_to_df(df, results, output_column_names)
        num_error = sum(output_df[output_column_names.response] == "")
        num_success = len(df.index) - num_error
        logging.info(
            (
                f"Applied function in parallel: {num_success} row(s) succeeded, {num_error} failed "
                f"in {(perf_counter() - start):.2f} seconds."
            )
        )
        return output_df
