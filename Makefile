# Convenience targets (macOS / Linux). On Windows, run the python commands
# directly or use `make` via WSL / Git Bash.

.PHONY: install install-gpu test data prepare overfit train train-small \
        generate visualize clean

install:        ## Install CPU dependencies
	pip install -r requirements.txt

install-gpu:    ## Install CUDA build of torch (adjust cu124 to your CUDA)
	pip install torch --index-url https://download.pytorch.org/whl/cu124
	pip install numpy tqdm pyyaml matplotlib pytest jupyter

test:           ## Run the test suite
	pytest -q

data:           ## Download the tiny-shakespeare dataset
	python scripts/download_data.py

prepare:        ## Tokenize raw text into train/val .bin files
	python scripts/prepare_data.py --input data/raw/shakespeare.txt --output data/processed/

overfit:        ## Sanity check: overfit a single batch
	python scripts/overfit_check.py

train:          ## Train the tiny (~1M) model
	python scripts/train.py --config configs/tiny.yaml

train-small:    ## Train the small (~10M) model
	python scripts/train.py --config configs/small.yaml

generate:       ## Generate text (set CKPT=checkpoints/step_XXXXXX.pt)
	python scripts/generate.py --checkpoint $(CKPT) --prompt "ROMEO:" --tokens 300 --temperature 0.8 --top_k 40

visualize:      ## Plot attention heads (set CKPT=checkpoints/step_XXXXXX.pt)
	python scripts/visualize.py --checkpoint $(CKPT) --prompt "To be or not"

clean:          ## Remove generated artifacts (keeps bundled dataset)
	rm -rf checkpoints/*.pt notebooks/*.png
	find . -type d -name __pycache__ -exec rm -rf {} +
