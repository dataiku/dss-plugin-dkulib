dirs = dku_config # Add here new libs separated by a whitespace if needed

test-one:
	@echo "[START] Running unit tests on ${f}..."
	@( \
		rm -rf env; \
		python3 -m venv env/; \
		source env/bin/activate; \
		pip3 install --upgrade pip; \
		pip install --no-cache-dir -r tests/${f}/requirements.txt; \
		pip install --no-cache-dir -r dkulib/${f}/requirements.txt; \
		pytest -o junit_family=xunit2 --junitxml=unit.xml tests/${f} || true; \
		deactivate; \
	)
	@echo "[SUCCESS] Running unit tests: Done!"


test-all:
	@echo "[START] Running unit tests on all..."
	for file in $(dirs) ; do \
		python3 -m venv env/; \
		source env/bin/activate; \
		pip3 install --upgrade pip; \
		pip install --no-cache-dir -r tests/$${file}/requirements.txt; \
		pip install --no-cache-dir -r dkulib/$${file}/requirements.txt; \
		pytest -o junit_family=xunit2 --junitxml=unit.xml tests/${file} || true; \
		deactivate; \
	done
	@echo "[SUCCESS] Running unit tests: Done!"