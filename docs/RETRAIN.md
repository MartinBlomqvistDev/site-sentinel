# Retraining the dual-target Random Forest

One command. Everything it needs is committed and the venv is built.

```bash
cd C:\DS24\Site_Sentinel
.venv\Scripts\python.exe -m pipeline.04_train_random_forest
```

Expect roughly an hour. Close the browser and any dev servers first: the search
runs four workers and each holds its own copy of the 250k-row frame.

## What it writes

| Path | Contents |
|---|---|
| `data/analysis_results/rf_cv_metrics.csv` | **The point of the run.** One row per target: positives, positive rate, and mean/std for precision, recall and F1. |
| `models/rf_master_predictor_dual_lead_tuned.pkl` | Both fitted models, `{"preventive": ..., "standard": ...}`. |

## The question it answers

`configs/app.yaml` reports one unlabelled precision/recall/F1 triple (0.875 /
0.986 / 0.927) for a model with two targets. Nothing in the repo records which
target it came from, because the original script printed the numbers to a
terminal and saved only the models.

That matters, because the two targets are not equally hard:

- **`Y_standard`** is `ttc <= 2s` at this frame. TTC is an input feature, so
  this target is close to reading a column.
- **`Y_preventive`** is "will a vehicle come within 2 m of a person in the next
  4 s", excluding the current frame. This is the one worth reporting.

When the run finishes, open `rf_cv_metrics.csv` and read the row named
`Preventive (4s)`. That is the number the README and the CV should quote.

## Then update, in this order

1. `configs/app.yaml` — replace the three figures and add the target name beside
   them so this cannot be lost again.
2. `README.md` — the headline sentence, and the model comparison table.
3. The CV bullet, if the figures moved.

## Two things to expect

**The numbers may not be 0.875 / 0.986 / 0.927.** Three real bugs were fixed
between the original run and this one: the preventive target had drifted from a
2 m distance threshold onto a 2 s TTC threshold, the lookahead was including the
current frame, and the label leaked across vehicle-worker pairs at group
boundaries. If the figures move, that is the fixes landing, not a regression.

**The README's "2-5%" imbalance claim is wrong for the training targets.** The
run reports the real rates in its first log line: about 12.4% for `Y_standard`
and about 14.7% for `Y_preventive`. The 2-5% figure describes raw per-frame
event rate in the source data, not what SMOTE is correcting. Worth fixing in the
same pass.

## If it dies

An `ArrayMemoryError` means too many workers for the free memory. Lower
`random_forest.search.n_jobs` in `configs/model_training.yaml` from 4 to 2.
