import pytest
from dkulib.dku_config.dku_config import DkuConfig
from dkulib.dku_config.dss_parameter import DSSParameterError


class TestDkuConfig:
    def test_init(self):
        dku_config = DkuConfig()
        assert len(dku_config) == 0
        with pytest.raises(KeyError):
            _ = dku_config.abc
        with pytest.raises(KeyError):
            _ = dku_config[5]

    def test_complex_init(self):
        dku_config = DkuConfig(
            arg1={"value": 1},
            arg2={"value": 2})
        assert len(dku_config) == 2
        assert dku_config.arg1 == 1
        with pytest.raises(KeyError):
            _ = dku_config[2]
        with pytest.raises(KeyError):
            _ = dku_config.arg3

    def test_simple_setter(self):
        dku_config = DkuConfig(
            arg1={"value": 1},
            arg2={"value": 2})
        dku_config['arg3'] = 3
        assert len(dku_config) == 3
        assert dku_config['arg3'] == 3
        assert dku_config.arg3 == 3

        dku_config[4] = 'cde'
        assert len(dku_config) == 4
        assert dku_config[4] == 'cde'

    def test_param_setter(self):
        dku_config = DkuConfig()

        dku_config.add_param(
            name='param1',
            value=0,
            required=True
        )
        assert len(dku_config) == 1
        with pytest.raises(DSSParameterError):
            dku_config.add_param(
                name=None,
                value=None,
                required=True
            )
        dku_config.add_param(
            name=None,
            value='abc',
            required=True
        )
        assert len(dku_config) == 2

    def test_param_getter(self):
        dku_config = DkuConfig()

        dku_config.add_param(
            name='param1',
            value=0,
            required=True
        )
        dku_config.add_param(
            name='param2',
            value='abc'
        )
        dku_config.add_param(
            name='param3',
            value='abc',
            checks=[{
                "type": "eq",
                "op": "abc"
            }]
        )
        assert len(dku_config) == 3
        assert dku_config.param3 == 'abc'
        assert type(dku_config['param1']) == int

    def test_local_vars(self):
        dku_config = DkuConfig(
            local_vars={
                "MY_PLUGIN__param1": "value1",
                "MY_PLUGIN__param2": 2,
                "MY_PLUGIN__param3": [1, 2, 3, 4],
            },
            local_prefix="MY_PLUGIN"
        )
        dku_config.add_param(
            name="param1"
        )
        dku_config.add_param(
            name="param2"
        )
        dku_config.add_param(
            name="param3",
            value=[4, 5, 6]
        )
        param1 = dku_config.param1
        param2 = dku_config["param2"]
        param3 = dku_config["param3"]
        assert param1 == "value1"
        assert param2 == 2
        assert param3 == [4, 5, 6]
