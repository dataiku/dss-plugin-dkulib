# Dataiku DSS Plugin Library

This repo contains reusable code to help develop Dataiku DSS plugins. 

## Usage

When used in a Dataiku DSS plugin, add this repository as a git submodule in python-lib by executing the following command from within python-lib:

`git submodule add https://github.com/dataiku/dss-plugin-dkulib dkulib/`

You can then import modules in the plugins recipe.py file via e.g.:

`from dkulib.core.parallelizer import DataFrameParallelizer`

## Included libs

- [dku_config](core/dku_config) (Last update: 2021-07): Gives the ability to check form parameters in the backend and display understandable messages if it
 fails.
- [nlp](core/nlp) (Last update: 2021-01): Detects languages, tokenize, correct misspellings and clean text data.
- [io_utils](core/io_utils) (Last update: 2021-01): Input / output utility functions which do not need the Dataiku API.
- [dku_io_utils](core/dku_io_utils) (Last update: 2021-07): Input / output utility functions to read/write from and to Dataiku objects e.g., chunked read/transform/write of dataiku Datasets.
- [parallelizer](core/parallelizer) (Last update: 2021-07): Applies a function to a pandas DataFrame with parallelization, error logging and progress tracking.

## License

This library is distributed under the [Apache License version 2.0](LICENSE).
