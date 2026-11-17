"""Fine-tune distilroberta-base on Split A (grouped random) and Split B (temporal).

5 seeds per split. For Split A the seed picks both the random split and the training randomness;
for Split B the split is fixed and the seed only changes training randomness.
Writes results/transformer_runs.csv (one row per run, appended as each run finishes)
and results/transformer_summary.csv.

Run from the repo root:
  python src/train_transformer.py           # full run (GPU strongly recommended)
  python src/train_transformer.py --smoke   # tiny run to check the code works; writes nothing to results/
"""
import argparse
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments, set_seed)

from evaluate import compute_metrics
from splits import RANDOM_SEEDS, load_clean, split_random_grouped, split_temporal

MODEL_NAME = "distilroberta-base"
SEEDS = list(RANDOM_SEEDS)[:5]  # 42-46: the first 5 of the baseline's 10 seeds

# Fixed in advance from common fine-tuning defaults. Never tuned on test scores.
MAX_LEN = 256      # tokens per post; longer posts are cut. 128 would cut 19.5% of posts, 256 cuts 0.5%.
EPOCHS = 3         # full passes over the training set
BATCH_SIZE = 16
LR = 2e-5          # learning rate: how big each weight update is
WEIGHT_DECAY = 0.01
WARMUP_SHARE = 0.1  # first 10% of steps ramp the learning rate up from 0, which keeps early updates gentle

RESULTS = Path(__file__).resolve().parent.parent / "results"


def to_dataset(df: pd.DataFrame, tok) -> Dataset:
    ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))
    # Tokenize = split text into the model's sub-word pieces and map each to an integer id.
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN), batched=True,
                  remove_columns=["text"])


def train_and_score(train: pd.DataFrame, test: pd.DataFrame, seed: int, smoke: bool) -> dict:
    set_seed(seed)  # seeds Python, NumPy and PyTorch together
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Loads the pre-trained body plus a NEW, randomly initialised 2-class output layer.
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    steps_per_epoch = math.ceil(len(train) / BATCH_SIZE)
    with tempfile.TemporaryDirectory() as tmp:  # Trainer needs a folder; we don't keep checkpoints
        args = TrainingArguments(
            output_dir=tmp,
            num_train_epochs=EPOCHS,
            max_steps=5 if smoke else -1,  # -1 = use num_train_epochs
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=64,
            learning_rate=LR,
            weight_decay=WEIGHT_DECAY,
            warmup_steps=int(WARMUP_SHARE * steps_per_epoch * EPOCHS),
            seed=seed,
            data_seed=seed,
            fp16=torch.cuda.is_available(),  # half-precision math on GPU: faster, less memory
            save_strategy="no",
            eval_strategy="no",  # no peeking at test during training
            logging_steps=50,
            report_to="none",
        )
        trainer = Trainer(model=model, args=args, train_dataset=to_dataset(train, tok),
                          data_collator=DataCollatorWithPadding(tok))  # pads each batch to its longest post
        trainer.train()
        logits = trainer.predict(to_dataset(test, tok)).predictions
    preds = np.argmax(logits, axis=-1)  # pick the class with the higher score
    return compute_metrics(test["label"], preds)


def all_runs(df: pd.DataFrame):
    b_train, b_test = split_temporal(df)
    for seed in SEEDS:
        yield ("A_random", seed, *split_random_grouped(df, seed))
        yield ("B_temporal", seed, b_train, b_test)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="tiny data, 5 steps, no files written")
    smoke = parser.parse_args().smoke

    print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    df = load_clean()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    lengths = [len(ids) for ids in tok(df["text"].tolist())["input_ids"]]
    print(f"posts longer than {MAX_LEN} tokens (will be truncated): {np.mean(np.array(lengths) > MAX_LEN):.1%}")

    out = RESULTS / "transformer_runs.csv"
    if not smoke and out.exists():
        out.unlink()  # start fresh so old and new runs never mix
    rows = []
    for split_name, seed, train, test in all_runs(df):
        if smoke:
            train, test = train.head(64), test.head(32)
        metrics = train_and_score(train, test, seed, smoke)
        row = {"model": MODEL_NAME, "split": split_name, "seed": seed,
               "n_train": len(train), "n_test": len(test), **metrics}
        rows.append(row)
        print(f"{split_name} seed {seed}: macro-F1 {metrics['macro_f1']:.4f}  accuracy {metrics['accuracy']:.4f}")
        if smoke:
            break  # one run is enough to prove the code works
        # Append after every run, so a crash halfway keeps the finished runs.
        pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)

    if smoke:
        print("\nsmoke test passed (nothing written)")
        return
    runs = pd.read_csv(out)
    summary = (runs.groupby(["model", "split"], sort=False)
                   .agg(n_runs=("macro_f1", "size"),
                        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
                        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"))
                   .reset_index())
    summary.to_csv(RESULTS / "transformer_summary.csv", index=False)
    print("\n" + summary.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
