modules = dku_config # Add here new libs separated by a whitespace if needed

test-one:
	@echo "[START] Running unit tests on ${module}..."
	@( \
		rm -rf env; \
		python3 -m venv env/; \
		source env/bin/activate; \
		pip3 install --upgrade pip; \
		pip3 install --no-cache-dir -r dkulib/${module}/requirements.txt -r tests/requirements.txt; \
		export PYTHONPATH="$(PYTHONPATH):$(PWD)"; \
		pytest tests/${module} --alluredir=tests/allure_report; \
		deactivate; \
	)
	@echo "[SUCCESS] Running unit tests on ${module}: Done!"


test-all:
	@echo "[START] Running all unit tests..."
	@for module in $(modules) ; do \
		$(MAKE) module=$${module} test-one; \
	done
	@echo "[SUCCESS] Running all unit tests: Done!"
