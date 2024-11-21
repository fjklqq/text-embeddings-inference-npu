SHELL := /bin/bash

gen-venv:
	pip install virtualenv
	python -m venv ./venv
	source venv/bin/activate && pip install torch==2.2.0 && pip install -e . && cd transformers && pip install .


install-all: gen-venv
	pip install -e .

install:
	pip install -e .

