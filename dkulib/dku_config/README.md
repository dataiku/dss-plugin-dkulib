# DKU Config

## Description

This lib can be used to validate parameters in a custom form. We know that front validation is never enough
and does not work in some cases in DSS (Webapps for example). Thus, you can create an object DkuConfig that
behavies as a dict, and add parameters with checks.

## Examples

Let's say you have built a custom form containing the following fields :

- first_name
- gender
- age
- phone_number
    
You want to validate in backend that the parameters have been filled properly and if not, display an understandable
message. You can then build a DkuConfig object.

```python
from dkulib.dku_config.dku_config import DkuConfig
from dataiku.customrecipe import get_recipe_config

config = get_recipe_config()
dku_config = DkuConfig()

dku_config.add_param(
    name="first_name",
    value=config.get("first_name"),
    required=True
)

dku_config.add_param(
    name="gender",
    value=config.get("gender"),
    checks=[{
        "type": "in",
        "op": ['M', 'F']
    }],
    required=True
)

dku_config.add_param(
    name="age",
    value=config.get("age"),
    checks=[{
        "type": "between",
        "op": (18, 100),
        "err_msg": "You must be over 18 to use the plugin (You specified {value})"
    }],
    required=True
)

dku_config.add_param(
    name="phone_number",
    value=config.get("phone_number"),
    checks=[{
        "type": "match",
        "op": '^(?:(?:\+|00)33[\s.-]{0,3}(?:\(0\)[\s.-]{0,3})?|0)[1-9](?:(?:[\s.-]?\d{2}){4}|\d{2}(?:[\s.-]?\d{3}){2})$'
    }],
    required=True
)

# ...

assert dku_config.age < 100
assert dku_config["age"] < 100

```

## Projects using the library

Don't hesitate to check these plugins using the library for more examples :

- [dss-plugin-deeplearning-image](https://github.com/dataiku/dss-plugin-deeplearning-image)
- [dss-plugin-ml-assisted-labeling](https://github.com/dataiku/dss-plugin-ml-assisted-labeling)

## Version

- Version: 0.1
- State: <span style="color:green">Stable</span>

## Credit

Library created and maintained by Henri Chabert.