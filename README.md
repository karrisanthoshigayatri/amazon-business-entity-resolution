# amazon-business-entity-resolution

# Amazon Business Entity Resolution

This repository is being built as a step-by-step implementation of the Business Entity Resolution challenge.

## Goal

Given records from Source 1, Source 2, and Source 3, the task is to find which Source 2 / Source 3 records refer to the same real-world business entity as each Source 1 record.

This project follows the challenge specification and is organized so work can be built incrementally and pushed to the GitHub repo at any time.

## Repository structure

- `code/business_entity_resolution/` - reusable pipeline code for data loading, blocking, matching, and output generation
- `dataset/` - challenge data files (train and test)
- `output/` - final TSV outputs: `matching_results.tsv` and `candidate_pairs.tsv`
- `Documentation_template.md` - challenge methodology write-up template

## Current status

This repo is intentionally scaffolded first so the project can be developed in a structured way:

- pipeline skeleton created
- project folders initialized
- starter code for loading, normalization, blocking, and scoring is in place
- outputs and dataset folders are ready for data to be added

## Next steps

1. Add the dataset files under `dataset/train` and `dataset/test`
2. Run the pipeline on the training data
3. Validate candidate and match outputs
4. Tune blocking and model thresholds using validation splits
5. Generate final test-set predictions
6. Push the final working repo to GitHub

## Notes

The challenge is precision-heavy. The model should prioritize avoiding false merges while keeping recall reasonable.
