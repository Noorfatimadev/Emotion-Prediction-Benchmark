# IN5550 – Oblig 1

Implementation of regression baselines for emotion prediction using:

- **Part 1:** Bag-of-Words (BoW) + MLP  
- **Part 2:** Pre-trained / custom embeddings + MLP

This document explains the purpose of each project file and the generated outputs.

---

## 📁 Project Structure
---

#### `datasets1.py`
Dataset classes are used across training and evaluation.

---

#### `data_loading.py`
Contains the `CollateFunctor`

---

#### `model.py`
Model definitions used in experiments.

---

#### `train_static_embeddings.py`
Trains custom Word2Vec embeddings from a plain-text corpus.

---
#### `create_test_split.py`
Create a test split for the evaluation script

---

## 🧪 Experiment Runner Scripts

#### `run_bow_experiments.py`
Runs the BoW experiment grid:
- BoW variants
- Number of MLP layers

Saves logs and model checkpoints.

---

#### `run_embedding_experiments.py`
Runs embedding experiments across:
- Pooling strategies
- OOV handling
- Embedding source
- Finetuning options
- Network depth

---

## 🔁 Re-running Best Configurations

#### `rerun_best_bow.py`
Re-trains the best BoW configuration using multiple seeds.

---

#### `rerun_best_emb.py`
Same as above, but for embedding-based models.

---

## 📊 Evaluation 

#### `eval_on_test.py`
Loads a trained model checkpoint and evaluates it on test data.

Metrics:
- MSE
- MAE
- RMSE
- Pearson correlation
- Spearman correlation

---

## 📂 Generated Output (Created Automatically)

### `runs_emb`
Trained models and results in JSON format

### `runs_bow`
Trained models and results JSON format

### `embeddings*`
Trained embedding

### `BEST_BOW_3RUNS_DEV_MEANSTD.json`

### `BEST_EMB_3RUNS_DEV_MEANSTD.json`

### `emo_test.tsv`
---


