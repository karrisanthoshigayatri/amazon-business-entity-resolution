# Business Entity Resolution Pipeline

This folder contains the challenge pipeline implementation.

## Structure

- `src/` contains the source code
- `requirements.txt` lists the Python dependencies

## Run instructions

1. Create a virtual environment
2. Install requirements
3. Add the training and test TSV files under the dataset folders
4. Run the pipeline

Example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py --mode train
python src/main.py --mode predict
```

## Components

- `config.py` - file paths and configuration
- `data_loader.py` - reading TSV data
- `normalization.py` - text cleanup and normalization
- `blocking.py` - candidate generation logic
- `feature_engineering.py` - pairwise similarity features
- `train_model.py` - model training and validation
- `predict.py` - final inference and output generation
- `submission.py` - TSV output formatting

## Notes

This is a starter implementation for a step-by-step ER workflow. It is designed to be extended with stronger blocking rules, better features, and tuned thresholds as the project develops.
