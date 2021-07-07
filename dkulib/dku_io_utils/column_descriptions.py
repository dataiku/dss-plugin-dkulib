# -*- coding: utf-8 -*-
"""Module with read/write utility to set dataset column descriptions based on the Dataiku API"""

from typing import Dict, Optional

import dataiku


def get_description_for_column(
    dataset_schema: dataiku.core.dataset.Schema, column_name: str
) -> str:
    """Gets the column description from a dataiku dataset schema for a given column name

    The dataiku dataset schema is a list of dictionaries as following:
    [{"name": "column1", "type": "string", "comment": "blabla"}, {...}].
    The optional "comment" key corresponds to the column description.

    Args:
        dataset_schema: dataiku.Dataset schema instance obtained from the `read_schema` method
        column_name: Name of the column whose description you want to retrieve

    Returns:
        The column description if it exists

    """
    for column in dataset_schema:
        if column["name"] == column_name and "comment" in column:
            return column["comment"]


def set_column_descriptions(
    output_dataset: dataiku.Dataset,
    column_descriptions: Dict[str, str],
    input_dataset: Optional[dataiku.Dataset] = None,
) -> None:
    """Sets column descriptions of the output dataset based on a dictionary of column descriptions

    Can also retain the column descriptions from the input dataset if specified

    Args:
        output_dataset: Output dataiku.Dataset instance
        column_descriptions: Dictionary holding column descriptions (value) by column name (key)
        input_dataset: Optional input dataiku.Dataset instance
            in case you want to retain input column descriptions

    """
    output_schema = output_dataset.read_schema()
    # First, set all output column descriptions to those of the input dataset if specified
    if input_dataset is not None:
        input_schema = input_dataset.read_schema()
        for output_column in output_schema:
            input_column_description = get_description_for_column(
                input_schema, output_column["name"]
            )
            if input_column_description is not None:
                output_column["comment"] = input_column_description
    # Then, update the output column descriptions according to the column_descriptions argument
    for output_column in output_schema:
        if output_column["name"] in column_descriptions:
            output_column["comment"] = column_descriptions[output_column["name"]]
    output_dataset.write_schema(output_schema)
