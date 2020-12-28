import pytest
from dkulib.dku_config.dss_parameter import DSSParameter, DSSParameterError
from dkulib.dku_config.custom_check import CustomCheckError
import logging

LOGGER = logging.getLogger(__name__)


class TestDSSParameter:
    def test_nominal_case(self):
        dss_parameter = DSSParameter(
            name='test',
            value=3,
            checks=[{
                "type": "inf",
                "op": 4
            }],
            required=True
        )
        assert dss_parameter.value == 3
        assert dss_parameter.name == 'test'
        assert len([c for c in dss_parameter.checks if c.type == 'exists'])
        assert len(dss_parameter.checks) == 2

    def test_error(self):
        with pytest.raises(CustomCheckError):
            _ = DSSParameter(
                name='test',
                value=3,
                checks=[{
                    "type": "unknown_type",
                    "op": 4
                }],
                required=True
            )

    def test_success(self, caplog):
        caplog.set_level(logging.INFO)
        dss_parameter = DSSParameter(
            name='test',
            value=3,
            checks=[{
                "type": "inf",
                "op": 4
            }],
            required=True
        )
        assert 'All checks have been successfully done' in caplog.text

    def test_failure(self, caplog):
        caplog.set_level(logging.INFO)
        with pytest.raises(DSSParameterError) as err:
            _ = DSSParameter(
                name='test',
                value=3,
                checks=[{
                    "type": "inf",
                    "op": 2
                }],
                required=True
            )
            assert 'Error for parameter' in err

