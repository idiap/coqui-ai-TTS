.DEFAULT_GOAL := help
.PHONY: test system-deps style lint install install_dev help docs

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

target_dirs := tests TTS notebooks recipes

test:	## run tests.
	coverage run -m pytest -x -v --durations=0 tests

test_vocoder:	## run vocoder tests.
	coverage run -m pytest -x -v --durations=0 tests/vocoder_tests

test_tts:	## run tts tests.
	coverage run -m pytest -x -v --durations=0 tests/tts_tests

test_aux:	## run aux tests.
	coverage run -m pytest -x -v --durations=0 tests/aux_tests

test_zoo:	## run zoo tests.
	coverage run -m pytest -x -v --durations=0 tests/zoo_tests/test_models.py

test_zoo_big:	## run tests for models that are too big for CI.
	coverage run -m pytest -x -v --durations=0 tests/zoo_tests/test_big_models.py

inference_tests: ## run inference tests.
	coverage run -m pytest -x -v --durations=0 tests/inference_tests

data_tests: ## run data tests.
	coverage run -m pytest -x -v --durations=0 tests/data_tests

test_text: ## run text tests.
	coverage run -m pytest -x -v --durations=0 tests/text_tests

test_failed:  ## only run tests failed the last time.
	coverage run -m pytest -x -v --last-failed tests

style:	## update code style.
	uv run --only-dev ruff format ${target_dirs}

lint:	## run linters.
	uv run --only-dev ruff check ${target_dirs}
	uv run --only-dev ruff format ${target_dirs} --check

system-deps:	## install linux system deps
	sudo apt-get install -y libsndfile1-dev

install:	## install 🐸 TTS
	uv sync --all-extras

install_dev:	## install 🐸 TTS for development.
	uv sync --all-extras
	uv run pre-commit install

docs:	## build the docs
	uv run --group docs $(MAKE) -C docs clean && uv run --group docs $(MAKE) -C docs html
