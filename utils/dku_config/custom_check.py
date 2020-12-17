import logging
logger = logging.getLogger(__name__)

DEFAULT_ERROR_MESSAGES = {
    'exists': 'This field is required.',
    'in': 'Should be in the following iterable: {op}.',
    'not_in': 'Should not be in the following iterable: {op}.',
    'sup': 'Should be superior to {op} (Currently {value}).',
    'sup_eq': 'Should be superior or equal to {op} (Currently {value}).',
    'inf': 'Should be inferior to {op} (Currently {value}).',
    'inf_eq': 'Should be inferior or equal to {op} (Currently {value}).',
    'between': 'Should be between {op[0]} and {op[1]} (Currently {value}).',
    'between_strict': 'Should be strictly between {op[0]} and {op[1]} (Currently {value}).',
    'is_type': 'Should be of type {op}.',
    'custom': "Unknown error append."
}


class CustomCheckError(Exception):
    pass


class CustomCheck:
    def __init__(self, type, op=None, cond=None, err_msg=''):
        self.type = type
        self.op = op
        self.cond = cond
        self.err_msg = err_msg or self.get_default_err_msg()

    def run(self, value=None):
        result = getattr(self, '_{}'.format(self.type))(value)
        self.handle_return(result, value)

    def handle_return(self, result, value):
        try:
            assert result
        except AssertionError:
            raise CustomCheckError(self.format_err_msg(value))

    def get_default_err_msg(self):
        return DEFAULT_ERROR_MESSAGES[self.type]

    def format_err_msg(self, value):
        formatted_err_msg = self.err_msg.format(value=value, op=self.op)
        return f'{formatted_err_msg}'

    def _exists(self, value):
        return value is not None

    def _in(self, value):
        return value in self.op

    def _not_in(self, value):
        return value not in self.op

    def _sup(self, value):
        return value > float(self.op)

    def _inf(self, value):
        return value < float(self.op)

    def _sup_eq(self, value):
        return value >= float(self.op)

    def _inf_eq(self, value):
        return value <= float(self.op)

    def _between(self, value):
        return float(self.op[0]) <= value <= float(self.op[1])

    def _between_strict(self, value):
        return float(self.op[0]) < value < float(self.op[1])

    def _is_type(self, value):
        return isinstance(value, self.op)

    def _custom(self, value):
        return self.cond