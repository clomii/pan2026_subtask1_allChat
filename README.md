# PAN 2026 Subtask 1 Submission

Submission package for PAN 2026 Voight-Kampff Generative AI Detection:
binary AI text detection.

The system reads a PAN/TIRA `dataset.jsonl` file containing `id` and `text`
fields and writes one `prediction.jsonl` file with probability labels in
`[0.0, 1.0]`, where higher means AI-generated.

## Contents

- `predict.py` - TIRA-compatible inference entrypoint.
- `Dockerfile` - builds the submission image.
- `validate_submission.py` - checks prediction format and IDs.
- `models/ngram_pipeline.joblib` - serialized N-gram classifier.
- `deberta_results/checkpoint-2964/` - DeBERTa inference checkpoint.

## Run Locally

Build the Docker image:

```bash
docker build -t pan26-subtask1 .
```

Run prediction:

```bash
docker run --rm \
  -v "$PWD/data/val.jsonl:/input/dataset.jsonl" \
  -v "$PWD/submission_out:/out" \
  pan26-subtask1 /input/dataset.jsonl /out
```

Validate output:

```bash
docker run --rm \
  -v "$PWD/data/val.jsonl:/input/dataset.jsonl" \
  -v "$PWD/submission_out:/out" \
  --entrypoint python3 \
  pan26-subtask1 /app/validate_submission.py /input/dataset.jsonl /out/prediction.jsonl
```

PowerShell users can use `${PWD}` instead of `$PWD`.

## TIRA Command

Use this command for code submission:

```bash
python3 /app/predict.py $inputDataset/dataset.jsonl $outputDir
```

Dry run:

```bash
tira-cli code-submission --dry-run \
  --path . \
  --task generative-ai-authorship-verification-panclef-2026 \
  --dataset DATASET_ID_FROM_TIRA \
  --command 'python3 /app/predict.py $inputDataset/dataset.jsonl $outputDir'
```

Submit:

```bash
tira-cli code-submission \
  --path . \
  --task generative-ai-authorship-verification-panclef-2026 \
  --dataset DATASET_ID_FROM_TIRA \
  --command 'python3 /app/predict.py $inputDataset/dataset.jsonl $outputDir'
```

## Notes

The DeBERTa `model.safetensors` file is tracked with Git LFS. Run
`git lfs pull` after cloning if the model file is missing or appears as a small
pointer file.
