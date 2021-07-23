import pytest
import numpy as np
from dku_config.custom_check import CustomCheck, CustomCheckError


class TestCustomCheck:
    def test_init(self):
        custom_check = CustomCheck(
            type='exists'
        )
        assert custom_check.type == 'exists'
        with pytest.raises(CustomCheckError):
            _ = CustomCheck(
                type='unknown_type'
            )

    def test_exists(self):
        custom_check = CustomCheck(
            type='exists'
        )
        assert custom_check.run('test') is None
        with pytest.raises(CustomCheckError):
            _ = custom_check.run('')
        with pytest.raises(CustomCheckError):
            _ = custom_check.run([])
        with pytest.raises(CustomCheckError):
            _ = custom_check.run(None)

    def test_in(self):
        custom_check = CustomCheck(
            type='in',
            op=[1, 2, 3]
        )
        assert custom_check.run(1) is None
        with pytest.raises(CustomCheckError):
            _ = custom_check.run(4)

    def test_not_in(self):
        custom_check = CustomCheck(
            type='not_in',
            op=[1, 2, 3]
        )
        assert custom_check.run(4) is None
        with pytest.raises(CustomCheckError):
            _ = custom_check.run(1)

    def test_eq(self):
        custom_check = CustomCheck(
            type='eq',
            op=5
        )
        assert custom_check.run(5) is None
        with pytest.raises(CustomCheckError):
            _ = custom_check.run('5')
        with pytest.raises(CustomCheckError):
            _ = custom_check.run(-2)
        with pytest.raises(CustomCheckError):
            _ = custom_check.run(-np.Inf)
        with pytest.raises(CustomCheckError):
            _ = custom_check.run(None)

    def test_sup(self):
        custom_check = CustomCheck(
            type='sup',
            op=5
        )
        assert custom_check.run(7) is None
        with pytest.raises(CustomCheckError):
            _ = custom_check.run(3)
        with pytest.raises(CustomCheckError):
            _ = custom_check.run(-2)
        with pytest.raises(CustomCheckError):
            _ = custom_check.run(-np.Inf)
        with pytest.raises(CustomCheckError):
            _ = custom_check.run(5)

    def test_inf(self):
        custom_check = CustomCheck(
            type='inf',
            op=5
        )
        assert custom_check.run(2) is None
        with pytest.raises(CustomCheckError):
            _ = custom_check.run(8)
        with pytest.raises(CustomCheckError):
            _ = custom_check.run(np.Inf)
        with pytest.raises(CustomCheckError):
            _ = custom_check.run(5)

    def test_sup_eq(self):
        custom_check = CustomCheck(
            type='sup_eq',
            op=5
        )
        assert custom_check.run(7) is None
        assert custom_check.run(5) is None
        with pytest.raises(CustomCheckError):
            _ = custom_check.run(3)
        with pytest.raises(CustomCheckError):
            _ = custom_check.run(-2)
        with pytest.raises(CustomCheckError):
            _ = custom_check.run(-np.Inf)

    def test_inf_eq(self):
        custom_check = CustomCheck(
            type='inf_eq',
            op=5
        )
        assert custom_check.run(2) is None
        assert custom_check.run(5) is None
        with pytest.raises(CustomCheckError):
            _ = custom_check.run(8)
        with pytest.raises(CustomCheckError):
            _ = custom_check.run(np.Inf)

    def test_between(self):
        custom_check = CustomCheck(
            type='between',
            op=(3, 8)
        )
        assert custom_check.run(3) is None
        assert custom_check.run(5) is None
        assert custom_check.run(8) is None
        with pytest.raises(CustomCheckError):
            _ = custom_check.run(1)
        with pytest.raises(CustomCheckError):
            _ = custom_check.run(20)
        with pytest.raises(CustomCheckError):
            _ = custom_check.run(-np.Inf)
        with pytest.raises(CustomCheckError):
            _ = custom_check.run(np.Inf)

    def test_between_strict(self):
        custom_check = CustomCheck(
            type='between_strict',
            op=(3, 8)
        )
        assert custom_check.run(5) is None
        with pytest.raises(CustomCheckError):
            _ = custom_check.run(1)
        with pytest.raises(CustomCheckError):
            _ = custom_check.run(20)
        with pytest.raises(CustomCheckError):
            _ = custom_check.run(3)
        with pytest.raises(CustomCheckError):
            _ = custom_check.run(8)
        with pytest.raises(CustomCheckError):
            _ = custom_check.run(-np.Inf)
        with pytest.raises(CustomCheckError):
            _ = custom_check.run(np.Inf)

    def test_is_type(self):
        custom_check_list = CustomCheck(
            type='is_type',
            op=list
        )
        assert custom_check_list.run([1, 2, 3, 4]) is None
        assert custom_check_list.run([]) is None
        with pytest.raises(CustomCheckError):
            _ = custom_check_list.run(None)
        with pytest.raises(CustomCheckError):
            _ = custom_check_list.run("test")
        with pytest.raises(CustomCheckError):
            _ = custom_check_list.run({1, 2, 3})

        custom_check_float = CustomCheck(
            type='is_type',
            op=float
        )
        assert custom_check_float.run(3.4) is None
        with pytest.raises(CustomCheckError):
            _ = custom_check_float.run("test")
        with pytest.raises(CustomCheckError):
            _ = custom_check_float.run(None)
        with pytest.raises(CustomCheckError):
            _ = custom_check_float.run(0)
        with pytest.raises(CustomCheckError):
            _ = custom_check_float.run(4)

    def test_custom(self):
        assert CustomCheck(type='custom', op=1 == 1).run() is None
        assert CustomCheck(type='custom', op=len([1, 2, 3, 4]) == 4).run() is None
        with pytest.raises(CustomCheckError):
            CustomCheck(type='custom', op=3 == 4).run()

    def test_match(self):
        custom_check_match = CustomCheck(
            type='match',
            op=r"^(?:(?:\+|00)33[\s.-]{0,3}(?:\(0\)[\s.-]{0,3})?|0)[1-9](?:(?:[\s.-]?\d{2}){4}|\d{2}(?:[\s.-]?\d{3}){2})$"
        )
        assert custom_check_match.run('0234678956') is None
        with pytest.raises(CustomCheckError):
            _ = custom_check_match.run('abc')

    def test_is_castable(self):
        custom_check_match = CustomCheck(
            type='is_castable',
            op=int
        )
        assert custom_check_match.run(4) is None
        assert custom_check_match.run('4') is None
        assert custom_check_match.run(3.8) is None
        with pytest.raises(CustomCheckError):
            _ = custom_check_match.run('abc')
        with pytest.raises(CustomCheckError):
            _ = custom_check_match.run('3.8')

    def test_is_subset(self):
        custom_check_match = CustomCheck(
            type='is_subset',
            op=[1, 2, 3, "abc"]
        )
        assert custom_check_match.run([1, 2]) is None
        assert custom_check_match.run([1, 1, 2]) is None
        assert custom_check_match.run([]) is None
        assert custom_check_match.run(["abc"]) is None
        with pytest.raises(CustomCheckError):
            _ = custom_check_match.run(["cde"])

        custom_check_match = CustomCheck(
            type='is_subset',
            op=["a", "b", "c"]
        )
        assert custom_check_match.run("ab") is None
        assert custom_check_match.run({"a", "b"}) is None
