# LLM Normalization for Price Prediction

A thesis project investigating whether LLM-based text normalization of product descriptions improves price prediction accuracy on the Mercari marketplace dataset. Three model architectures are benchmarked on both raw and normalized descriptions.

## Overview

Product listings on secondhand marketplaces often contain inconsistent language — abbreviations, misspellings, seller noise, and non-standard condition descriptions. This project uses **GPT-4o-mini** to normalize those descriptions, then evaluates the effect on price prediction across three architectures:

| Model | Description |
|---|---|
| **Fine-tuned DistilBERT** | All ~66M parameters trained end-to-end |
| **Frozen Linear** | Frozen DistilBERT encoder + single linear regression head (~770 trainable params) |
| **XGBoost** | Gradient boosting on frozen DistilBERT CLS embeddings |

Evaluation metric: **RMSLE** (Root Mean Squared Logarithmic Error)

## Project Structure

```
LLM_Normalization/
├── Data/
│   ├── train_sample.csv              # 85,675 training listings
│   ├── validation_sample.csv         # 18,359 validation listings
│   ├── test_sample.csv               # 18,360 test listings
│   ├── Finetuned_Raw_Results.csv
│   ├── Finetuned_P1_Results.csv
│   ├── Frozen_Linear_Raw_Results.csv
│   ├── Frozen_Linear_P1_Results.csv
│   ├── Frozen_Linear_P1_(Title,Cat,Desc).csv
│   ├── XGBoost_Raw_Results.csv
│   └── XGBoost_P1_Results.csv
└── src/
    ├── DataSplit.py                  # 70/15/15 train/val/test splitting
    ├── distribution.py               # Category distribution analysis
    ├── LLM_Norm_Pilot.py             # Small-scale normalization for manual inspection
    ├── normalize_batch.py            # Async large-scale normalization pipeline
    └── pipelines/
        ├── FinetunedPipeline.ipynb   # Fine-tuned DistilBERT training & evaluation
        ├── FrozenLinearPipeline.ipynb # Frozen encoder + linear head
        └── XGBoostPipeline.ipynb     # XGBoost on CLS embeddings
```

## Data

Each CSV has the following columns:

```
train_id, name, item_condition_id, category_name, brand_name, price, shipping, item_description
```

The dataset is filtered to 23 top-level product categories (Beauty, Apparel, Shoes, Electronics, Home, Toys, etc.) for class balance, then split 70/15/15.

## Setup

**Dependencies** (no requirements.txt — install manually):

```bash
pip install torch transformers pandas numpy scikit-learn xgboost openai tqdm matplotlib
```

The model pipelines are designed for **Google Colab** with a GPU (tested on A100-SXM4-80GB). Running locally requires a CUDA-enabled GPU and adjusting the Google Drive paths in the notebooks.

**OpenAI API key** is required for normalization. Set it in `normalize_batch.py` and `LLM_Norm_Pilot.py` before running.

## Usage

### 1. Data Preparation

```bash
# Inspect category distribution
python src/distribution.py --file Data/train_sample.csv

# Re-create train/val/test splits from a raw Mercari CSV
python src/DataSplit.py
```

### 2. Text Normalization

The normalization prompt instructs GPT-4o-mini to:
- Expand abbreviations and fix spelling
- Standardize item condition descriptions
- Remove seller noise (follower requests, bundle offers, personal appeals)
- Preserve all factual product information

```bash
# Quick pilot on 30 samples for manual inspection
python src/LLM_Norm_Pilot.py --input Data/train_sample.csv --n 30 --output pilot.tsv

# Full async normalization of all three splits (resumable)
python src/normalize_batch.py
# Outputs: train_p1.csv, validation_p1.csv, test_p1.csv
```

`normalize_batch.py` processes all splits concurrently (default: 100 concurrent requests), handles OpenAI rate limits with server-suggested retry delays, and is fully resumable — if interrupted, it picks up from where it left off using a `results.jsonl` checkpoint file.

### 3. Model Training

Run the notebooks in `src/pipelines/` in Jupyter or Google Colab:

1. **`FinetunedPipeline.ipynb`** — Fine-tunes all DistilBERT parameters on the normalized descriptions. Trains for 3 epochs with cosine LR schedule, gradient clipping, and mixed precision (FP16). Saves the best checkpoint by validation RMSLE.

2. **`FrozenLinearPipeline.ipynb`** — Freezes the encoder and trains only the linear regression head. Tests how well pre-trained representations transfer without any task-specific fine-tuning.

3. **`XGBoostPipeline.ipynb`** — Extracts 768-dimensional CLS embeddings from frozen DistilBERT and trains an XGBoost regressor as a classical ML baseline.


## License
MIT

## Authors
Lárus Þóroddsson
Orri Kristjánsson
Steinar Örn Sólmundsson
