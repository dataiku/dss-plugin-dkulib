# DKU IO Utils

## Description

The lib contains utility functions to read/write from and to Dataiku objects.

## Examples

Here is an example of usage:

```
import dataiku

from core.dku_io_utils.chunked_processing import process_dataset_chunks
from core.dku_io_utils.column_descriptions import set_column_descriptions

process_dataset_chunks(
    input_dataset=dataiku.Dataset("input"),
    output_dataset=dataiku.Dataset("output"),
    func=lambda df, param: df,
    param=42
)

set_column_descriptions(
    input_dataset=dataiku.Dataset("input"),
    output_dataset=dataiku.Dataset("output"),
    column_descriptions={"your_column": "Your description"},
)
```

## Projects using the library

Don't hesitate to check these plugins using the library for more examples:
- [dss-plugin-nlp-preparation](https://github.com/dataiku/dss-plugin-nlp-preparation/blob/main/custom-recipes/nlp-preparation-cleaning/recipe.py)
- [dss-plugin-similarity-search](https://github.com/dataiku/dss-plugin-similarity-search/blob/main/custom-recipes/similarity-search-query/recipe.py)

## Version

- Version: 0.1.0
- State: <span style="color:green">Supported</span>

## Credit

Library created and maintained by Alex Combessie.
