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

HF_REPO := dgcnz/PART

.PHONY: upload_artifact
upload_artifact:
	# pull the second goal (the one after 'upload_artifact') into LOCAL_PATH
	$(eval LOCAL_PATH := $(word 2,$(MAKECMDGOALS)))
	@if [ -z "$(LOCAL_PATH)" ]; then \
	  echo "Usage: make upload_artifact <local-path>        # e.g. artifacts/.../config.yaml" ; \
	  exit 1 ; \
	fi
	@# strip off the first dir (everything up to the first slash)
	@remote=$$(echo $(LOCAL_PATH) | cut -d/ -f2-) ; \
	echo "Uploading '$(LOCAL_PATH)' → '$(remote)' in $(HF_REPO)" ; \
	huggingface-cli upload $(HF_REPO) $(LOCAL_PATH) $$remote

# a catch-all so that 'make' doesn't complain about the
# second goal being an unknown target
.PHONY: %
%: