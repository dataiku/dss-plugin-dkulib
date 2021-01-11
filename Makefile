modules = dku_config # Add here new libs separated by a whitespace if needed

test-one:
	@echo "[START] Running unit tests on ${module}..."
	@( \
		rm -rf env; \
		python3 -m venv env/; \
		source env/bin/activate; \
		pip3 install --upgrade pip; \
		pip3 install --no-cache-dir -r dkulib/${module}/requirements.txt -r tests/requirements.txt; \
		pytest tests/${module} --alluredir=tests/allure_report; \
		deactivate; \
	)
	@echo "[SUCCESS] Running unit tests: Done!"


test-all:
	@echo "[START] Running unit tests on all modules..."
	for module in $(modules) ; do \
		python3 -m venv env/; \
		source env/bin/activate; \
		pip3 install --upgrade pip; \
		pip3 install --no-cache-dir -r dkulib/$${module}/requirements.txt -r tests/requirements.txt; \
		pytest tests/${module} --alluredir=tests/allure_report; \
		deactivate; \
	done
	@echo "[SUCCESS] Running unit tests: Done!"
