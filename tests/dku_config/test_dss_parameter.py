import pytest
import numpy as np
from utils.dku_config.dss_parameter import DSSParameter, DSSParameterError


def test_nominal_case():
    dss_parameter = DSSParameter(
        name='test',
        value=3,
        checks=[{
            "type": "inf",
            "op": 4
        }]
    )

