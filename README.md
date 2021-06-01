# Dataiku DSS Plugin Library

This repo contains reusable code to help develop Dataiku DSS plugins.

## Included libs

- [dku_config](dkulib/dku_config) (Last update: 2021-01-28): Gives the ability to check forms parameters in backend and to display understandable messages if fails
- [nlp](dkulib/nlp) (Last update: 2021-01-11): Detect languages, tokenize, correct misspellings and clean text data
- [io_utils](dkulib/io_utils) (Last update: 2021-01-11): Input / output utility functions which do not need the Dataiku API
- [parallelizer](dkulib/parallelizer) (Last update: 2021-06-02): Apply a function to a pandas DataFrame with parallelization, error logging and progress tracking.

## License

This library is distributed under the [Apache License version 2.0](LICENSE).
