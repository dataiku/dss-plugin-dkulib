from utils.dku_config.custom_check import CustomCheck, CustomCheckError

import logging
logger = logging.getLogger(__name__)


class DSSParameterError(Exception):
    pass


class DSSParameter:
    def __init__(self, name, value, checks=None, required=False):
        if checks is None:
            checks = []
        self.name = name
        self.value = value
        self.checks = [CustomCheck(**check) for check in checks]
        if required:
            self.checks.append(CustomCheck(type='exists'))
        self.run_checks()

    def run_checks(self):
        errors = []
        for check in self.checks:
            try:
                check.run(self.value)
            except CustomCheckError as err:
                errors.append(err)
        if errors:
            self.handle_failure(errors)
        self.handle_success()

    def handle_failure(self, errors):
        raise DSSParameterError(self.format_failure_message(errors))

    def format_failure_message(self, errors):
        return """
        Error in parameter \"{name}\" :
        {errors}
        Please check your settings and fix errors.
        """.format(
            name=self.name,
            errors='\n'.join(["\t- {}".format(e) for e in errors])
        )

    def handle_success(self):
        self.print_success_message()
        return True

    def print_success_message(self):
        logger.info('All checks have been successfully done for {}.'.format(self.name))
