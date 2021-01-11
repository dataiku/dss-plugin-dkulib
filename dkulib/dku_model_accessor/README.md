# DKU Model Accessor

## Description
This lib provides tools to interact with dss saved models data (getting the original train/test set for example).

It has a surrogate model and a doctor-like default preprocessor allowing to retrieve feature importance of any non-tree-based models.

It uses an internal api, `dataiku.doctor.posttraining.model_information_handler.PredictionModelInformationHandler` (merci mamène Coni) so beware of future api break.


## Examples


```python
from dku_model_accessor import get_model_handler, ModelAccessor

model_id = 'XQyU0TO0'
model = dataiku.Model(model_id)
model_handler = get_model_handler(model)
model_accessor = ModelAccessor(model_handler)

original_test_set = model_accessor.get_original_test_df()
feature_importance = model_accessor.get_feature_importance() # works for any models
selected_features = model_accessor.get_selected_features()
```

## Projects using the library

Don't hesitate to check these plugins using the library for more examples :

- [dss-plugin-model-drift](https://github.com/dataiku/dss-plugin-model-drift)
- [dss-plugin-model-fairness-report](https://github.com/dataiku/dss-plugin-model-fairness-report)
- [dss-plugin-model-error-analysis](https://github.com/dataiku/dss-plugin-model-error-analysis)

## Version

- Version: 0.1.0
- State: <span style="color:green">Supported</span>

## Credit

Library created by Du Phan.