SYSTEM_PYTHON=python
PYTHON=.venv/bin/python
PIP=.venv/bin/pip

export_requirements:
	poetry export -f requirements.txt --output requirements.txt --without-hashes

clean_requirements:
	rm -f requirements.txt

install_requirements: clean_requirements export_requirements
	$(PIP) install -r requirements.txt

create_venv:
	@if [ ! -d ".venv" ]; then \
		$(SYSTEM_PYTHON) -m venv .venv; \
	fi

install: create_venv install_requirements clean_requirements


