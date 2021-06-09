# -*- coding: utf-8 -*-
"""Apply a function to a pandas DataFrame with parallelization, error logging and progress tracking"""

import logging
import inspect
import math

from typing import Callable, AnyStr, Any, List, Tuple, NamedTuple, Dict, Union, Optional
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
    """Apply a function to a pandas DataFrame with parallelization, error logging and progress tracking.

    This class is particularly well-suited for synchronous functions calling an API, either row-by-row or by batch.

    Attributes:
        function: Any function taking a dict as input (row-by-row mode) or a list of dict (batch mode),
            and returning a response with additional information, typically a JSON string.
            In batch mode, the response from the function should be parsable by the `batch_response_parser` attribute.
        error_handling: If ErrorHandling.LOG (default), log the error from the function as a warning,
            and add additional columns to the dataframe with the error message and error type.
            If ErrorHandling.FAIL, the function will fail is there is any error.
            We recommend letting the end user choose as there are contexts which justify one option or the other.
        exceptions_to_catch: Tuple of Exception classes to catch.
            Mandatory if ErrorHandling.LOG (default).
        parallel_workers: Number of concurrent threads to parallelize the function. Default is 4.
            We recommend letting the end user tune this parameter to get better performance.
        batch_support: If True, send batches of row (list of dict) to the `function`
            Else (default) send rows as dict to the function.
            This parameter should be chosen according to the nature of the function to apply.
        batch_size: Number of rows to include in each batch. Default is 10.
            Taken into account if `batch_support` is True.
            We recommend letting the end user tune this parameter if they need to increase performance.
        batch_response_parser: Function used to parse the raw response from the function in batch mode,
            and assign the actual responses and errors back to the original batch of row (list of dict).
            This is often required for batch APIs which return nested objects with a mix of responses and errors.
            This parameter is required if batch_support is True.
        output_column_prefix: Column prefix to add to the output columns of the dataframe,
            containing the `function` responses and errors. Default is "output".
            This should be overriden by the developer: if the function to apply calls an API for text translation,
            a good output_column_prefix would be "api_translation".
        verbose: If True, log raw details on any error encountered along with the error message and error type.
            Else (default) log only the error message and the error type.
            We recommend trying without verbose first. Usually, the error message is enough to diagnose the issue.

    """

    DEFAULT_PARALLEL_WORKERS = 4
    """Default number of worker threads to use in parallel - may be tuned by the end user"""
    DEFAULT_BATCH_SUPPORT = False
    """By default, we assume the function to apply is row-by-row - should be overriden in the batch case"""
    DEFAULT_BATCH_SIZE = 10
    """Default number of rows in one batch - may be tuned by the end user"""

    DEFAULT_OUTPUT_COLUMN_PREFIX = "output"
    """Default prefix to add to output columns - should be overriden for personalized output"""
    OUTPUT_COLUMN_NAME_DESCRIPTIONS = OrderedDict(
        [
            ("response", "Raw response in JSON format"),
            ("error_message", "Error message"),
            ("error_type", "Error type or code"),
            ("error_raw", "Raw error"),
        ]
    )
    """Default dictionary of output column names (key) and their descriptions (value)"""
    DEFAULT_VERBOSE = False
    """By default, set verbose to False assuming error message and type are enough information in the logs"""

    def __init__(
        self,
        function: Callable[[Union[Dict, List[Dict]]], Union[Dict, List[Dict]]],
        error_handling: ErrorHandling = ErrorHandling.LOG,
        exceptions_to_catch: Tuple[Exception] = (),
        parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
        batch_support: bool = DEFAULT_BATCH_SUPPORT,
        batch_size: int = DEFAULT_BATCH_SIZE,
        batch_response_parser: Optional[Callable[[List[Dict], Any, NamedTuple], List[Dict]]] = None,
        output_column_prefix: AnyStr = DEFAULT_OUTPUT_COLUMN_PREFIX,
        verbose: bool = DEFAULT_VERBOSE,
    ):
        self.function = function
        self.error_handling = error_handling
        self.exceptions_to_catch = exceptions_to_catch
        if error_handling == ErrorHandling.LOG and not exceptions_to_catch:
            raise ValueError("Please set at least one exception in exceptions_to_catch")
        self.parallel_workers = parallel_workers
        self.batch_support = batch_support
        self.batch_size = batch_size
        self.batch_response_parser = batch_response_parser
        if batch_support and not batch_response_parser:
            raise ValueError("Please provide a valid batch_response_parser function")
        self.output_column_prefix = output_column_prefix
        self.verbose = verbose
        self._output_column_names = None  # Will be set at runtime by the run method

    def _get_unique_output_column_names(self, existing_names: List[AnyStr]) -> NamedTuple:
        """Return a named tuple with prefixed column names and their descriptions"""
        OutputColumnNameTuple = namedtuple("OutputColumnNameTuple", self.OUTPUT_COLUMN_NAME_DESCRIPTIONS.keys())
        return OutputColumnNameTuple(
            *[
                generate_unique(name=column_name, existing_names=existing_names, prefix=self.output_column_prefix)
                for column_name in OutputColumnNameTuple._fields
            ]
        )

    def _apply_function_and_parse_response(
        self, row: Dict = None, batch: List[Dict] = None, **function_kwargs,
    ) -> Union[Dict, List[Dict]]:  # sourcery skip: or-if-exp-identity
        """Wrap a row-by-row or batch function with error logging and response parsing

        It applies `self.function` and:
        - If batch, parse the function response to extract results and errors using `self.batch_response_parser`
        - handles errors from the function with two methods:
            * (default) log the error message as a warning and return the row with error keys
            * fail if there is an error (if `self.error_handling == ErrorHandling.FAIL`)

        """
        if row and batch:
            raise (ValueError("Please use either row or batch as arguments, but not both"))
        output = deepcopy(row) if row else deepcopy(batch)
        for output_column in self._output_column_names:
            if row:
                output[output_column] = ""
            else:
                for output_row in output:
                    output_row[output_column] = ""
        try:
            response = (
                self.function(row=row, **function_kwargs) if row else self.function(batch=batch, **function_kwargs)
            )
            if row:
                output[self._output_column_names.response] = response
            else:
                output = self.batch_response_parser(
                    batch=batch, response=response, output_column_names=self._output_column_names
                )
                errors = [
                    row[self._output_column_names.error_message]
                    for row in output
                    if row[self._output_column_names.error_message]
                ]
                if errors:
                    raise BatchError(str(errors))
        except self.exceptions_to_catch + (BatchError,) as error:
            if self.error_handling == ErrorHandling.FAIL:
                raise error
            logging.warning(
                f"Function {self.function.__name__} failed on: {row if row else batch} because of error: {error}"
            )
            error_type = str(type(error).__qualname__)
            module = inspect.getmodule(error)
            if module:
                error_type = f"{module.__name__}.{error_type}"
            if row:
                output[self._output_column_names.error_message] = str(error)
                output[self._output_column_names.error_type] = error_type
                output[self._output_column_names.error_raw] = str(error.args)
            else:
                for output_row in output:
                    output_row[self._output_column_names.error_message] = str(error)
                    output_row[self._output_column_names.error_type] = error_type
                    output_row[self._output_column_names.error_raw] = str(error.args)
        return output

    def _convert_results_to_df(self, df: pd.DataFrame, results: List[Dict]) -> pd.DataFrame:
        """Combine results from the function with the input dataframe"""
        output_schema = {**{column_name: str for column_name in self._output_column_names}, **dict(df.dtypes)}
        output_df = (
            pd.DataFrame.from_records(results)
            .reindex(columns=list(df.columns) + list(self._output_column_names))
            .astype(output_schema)
        )
        if not self.verbose:
            output_df.drop(labels=self._output_column_names.error_raw, axis=1, inplace=True)
        if self.error_handling == ErrorHandling.FAIL:
            error_columns = [
                self._output_column_names.error_message,
                self._output_column_names.error_type,
                self._output_column_names.error_raw,
            ]
            output_df.drop(labels=error_columns, axis=1, inplace=True, errors="ignore")
        return output_df

    def run(self, df: pd.DataFrame, **function_kwargs,) -> pd.DataFrame:
        """Apply a function to a pandas.DataFrame with parallelization, error logging and progress tracking

        The DataFrame is iterated on and fed to the function as dictionaries, row-by-row or by batches of rows.
        This process is accelerated by the use of concurrent threads and is tracked with a progress bar.
        Errors are catched if they match the `self.exceptions_to_catch` attribute and automatically logged.
        Once the whole DataFrame has been iterated on, results and errors are added as additional columns.

        Args:
            df: Input dataframe on which the function will be applied
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
                f"Applying function {self.function.__name__} in parallel to {df_num_rows} row(s)"
                + f" using batch size of {self.batch_size}..."
            )
            df_row_batch_generator = chunked(df_row_generator, self.batch_size)
            len_generator = math.ceil(df_num_rows / self.batch_size)
        else:
            logging.info(f"Applying function {self.function.__name__} in parallel to {df_num_rows} row(s)...")
            len_generator = df_num_rows
        self._output_column_names = self._get_unique_output_column_names(existing_names=df.columns)
        pool_kwargs = function_kwargs.copy()
        for kwarg in ["function", "row", "batch"]:  # Reserved pool keyword arguments
            pool_kwargs.pop(kwarg, None)
        if not self.batch_support and "batch_response_parser" in pool_kwargs:
            pool_kwargs.pop("batch_response_parser", None)
        results = []
        with ThreadPoolExecutor(max_workers=self.parallel_workers) as pool:
            if self.batch_support:
                futures = [
                    pool.submit(self._apply_function_and_parse_response, batch=batch, **pool_kwargs)
                    for batch in df_row_batch_generator
                ]
            else:
                futures = [
                    pool.submit(self._apply_function_and_parse_response, row=row, **pool_kwargs)
                    for row in df_row_generator
                ]
            for future in tqdm_auto(as_completed(futures), total=len_generator, miniters=1, mininterval=1.0):
                results.append(future.result())
        results = flatten(results) if self.batch_support else results
        output_df = self._convert_results_to_df(df, results)
        num_error = sum(output_df[self._output_column_names.response] == "")
        num_success = len(df.index) - num_error
        logging.info(
            (
                f"Applied function in parallel: {num_success} row(s) succeeded, {num_error} failed "
                f"in {(perf_counter() - start):.2f} seconds."
            )
        )
        return output_df
