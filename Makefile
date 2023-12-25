.PHONY: init data baseline train deploy prepare-deployment test-endpoint

DEPLOYMENT_DIR = deployment_dir

init:
	curl -sSL https://install.python-poetry.org | python3 -
	poetry install

data:
	poetry run python src/analysis/scraper.py

baseline:
	poetry run python src/train/baseline_model.py

train:
	poetry run python src/train/train.py

prepare-deployment:
	rm -rf $(DEPLOYMENT_DIR) && mkdri $(DEPLOYMENT_DIR)
	poetry export -f requirement.txt --output $(DEPLOYMENT_DIR)/requirement.ext --without-hashes
	cp -r src/predict.py $(DEPLOYMENT_DIR)/main.py
	cp -r src $(DEPLOYMENT_DIR)/src/

deploy: prepare-deployment
	cd $(DEPLOYMENT_DIR) && poetry run cerebrium deploy --api-key $(CEREBRIUM_API_KEY) --hardware CPU eth-price-1-hour-predictor

test-endpoint:
	poetry run python src/train/test_endpoint.py