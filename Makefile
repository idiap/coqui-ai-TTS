.DEFAULT_GOAL := help
.PHONY: test system-deps style lint install install_dev help docs

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

target_dirs := tests TTS notebooks recipes

test_all:	## run tests and don't stop on an error.
	nose2 --with-coverage --coverage TTS tests
	./run_bash_tests.sh

test:	## run tests.
	coverage run -m nose2 -F -v -B tests

test_vocoder:	## run vocoder tests.
	coverage run -m nose2 -F -v -B tests.vocoder_tests

test_tts:	## run tts tests.
	coverage run -m nose2 -F -v -B tests.tts_tests

test_tts2:	## run tts tests.
	coverage run -m nose2 -F -v -B tests.tts_tests2

test_xtts:
	coverage run -m nose2 -F -v -B tests.xtts_tests

test_aux:	## run aux tests.
	coverage run -m nose2 -F -v -B tests.aux_tests
	./run_bash_tests.sh

test_zoo0:	## run zoo tests.
	coverage run -m nose2 -F -v -B tests.zoo_tests.test_models.test_models_offset_0_step_3 \
	tests.zoo_tests.test_models.test_voice_conversion
test_zoo1:	## run zoo tests.
	coverage run -m nose2 -F -v -B tests.zoo_tests.test_models.test_models_offset_1_step_3
test_zoo2:	## run zoo tests.
	coverage run -m nose2 -F -v -B tests.zoo_tests.test_models.test_models_offset_2_step_3

inference_tests: ## run inference tests.
	coverage run -m nose2 -F -v -B tests.inference_tests

data_tests: ## run data tests.
	coverage run -m nose2 -F -v -B tests.data_tests

test_text: ## run text tests.
	coverage run -m nose2 -F -v -B tests.text_tests

test_failed:  ## only run tests failed the last time.
	coverage run -m nose2 -F -v -B tests

style:	## update code style.
	uv run --only-dev black ${target_dirs}

lint:	## run linters.
	uv run --only-dev ruff check ${target_dirs}
	uv run --only-dev black ${target_dirs} --check

system-deps:	## install linux system deps
	sudo apt-get install -y libsndfile1-dev

build-docs: ## build the docs
	cd docs && make clean && make build

install:	## install 🐸 TTS
	uv sync --all-extras

install_dev:	## install 🐸 TTS for development.
	uv sync --all-extras
	uv run pre-commit install

docs:	## build the docs
	$(MAKE) -C docs clean && $(MAKE) -C docs html
