# Continual-Crisis-NLP

*Implementation & benchmark of **Continual Learning** methods (Regularization & Replay) on two crisis‑management NLP datasets — one **French** and one **English**.*

---

## 1. Project Overview

Crisis-response NLP systems must adapt to **new incoming events** without catastrophically forgetting what was learned before. This repository offers a lightweight, reproducible playground to compare several continual-learning (CL) strategies on two crisis-related corpora: **FrenchCorpus** (FR) and **HumAid** (EN).

### Implemented Methods

* **AGEM** (Gradient Episodic Memory variant)
* **Cumulative** (joint training upper bound)
* **Continual** (naïve sequential fine-tune baseline)
* **EWC** (Elastic Weight Consolidation)
* **MAS** (Memory Aware Synapses)
* **SI**  (Synaptic Intelligence)
* **Vanilla** replay (simple experience replay buffer)
* **NER** utilities / wrapper (sequence‑labeling support; optional)

All methods share a common interface so you can switch approaches with a single CLI flag.

---

## 2. Repository Structure

> Minimal tree (ignoring build / cache / test artifacts)

```text
.
├── .gitignore
├── config.json
├── data
│   ├── processed/
│   ├── raw/
│   └── scripts/
│       ├── dataset.py
│       └── preprocessing.py
├── models
│   ├── model.py
│   └── weights/
├── notebooks/
├── README.md   # (this file)
├── requirements.txt
├── results/
└── src
    ├── approaches/
    │   ├── agem.py
    │   ├── continual.py
    │   ├── cumulative.py
    │   ├── ewc.py
    │   ├── mas.py
    │   ├── ner.py
    │   ├── si.py
    │   └── vanilla.py
    ├── evaluate.py
    └── utils.py
```

---

## 3. Installation

### 3.1 Requirements

* Python >= 3.9 (3.10 recommended)
* pip >= 22
* GPU optional but recommended (CUDA build of PyTorch if available)

### 3.2 Clone & create environment

```bash
git clone https://github.com/plevankiem/Continual-Learning-on-Crisis-Management.git
cd Continual-Learning-on-Crisis-Management

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

**Conda alternative:**

```bash
conda create -n ccl python=3.10 -y
conda activate ccl
pip install -r requirements.txt
```

---

## 4. Data Setup

Place your raw dataset files under:

* `data/raw/FrenchCorpus/`
* `data/raw/HumAid/`

---

## 5. Results Directory Layout

The training / evaluation pipeline expects a results directory per *dataset* and *method*.

Target layout:

```text
results/
├── FrenchCorpus/
│   ├── agem/
│   ├── cumulative/
│   ├── ewc/
│   ├── mas/
│   ├── si/
│   └── vanilla/
└── HumAid/
    ├── agem/
    ├── cumulative/
    ├── ewc/
    ├── mas/
    ├── si/
    └── vanilla/
```

Create everything in one go:

```bash
for DS in FrenchCorpus HumAid; do
  for M in agem cumulative ewc mas si vanilla; do
    mkdir -p "results/${DS}/${M}"
  done
done
```

---

## 6. Running Experiments

The main entrypoint is `src/evaluate.py`, which orchestrates loading data, instantiating the selected CL method, training across tasks in the specified order, and reporting metrics.

### 6.1 Minimal Example

```bash
python src/evaluate.py \
  --method ewc \
  --dataset FrenchCorpus \
  --train_order chronological \
  --n_epochs 10 \
  --lr 3e-4 \
  --batch_size 32 \
  --gpu 0
```

### 6.2 Key Arguments

| Flag            | Type  | Description                   | Example                                               |
| --------------- | ----- | ----------------------------- | ----------------------------------------------------- |
| `--method`      | str   | CL algorithm to use           | `ewc`, `agem`, `si`, `vanilla`, ...                   |
| `--dataset`     | str   | Which dataset                 | `FrenchCorpus`, `HumAid`                              |
| `--train_order` | str   | Task sequence protocol        | `chronological`, `random`, `reverse`, `custom:<path>` |
| `--n_epochs`    | int   | Epochs per task               | `10`                                                  |
| `--lr`          | float | Learning rate                 | `3e-4`                                                |
| `--batch_size`  | int   | Batch size                    | `32`                                                  |
| `--seed`        | int   | Reproducibility seed          | `42`                                                  |
| `--gpu`         | int   | CUDA device id (omit for CPU) | `0`                                                   |

**Method‑specific hyper‑params** (e.g., `--ewc_lambda`, buffer size for replay, etc.) are also exposed. See the corresponding file in `src/approaches/` or run:

```bash
python src/evaluate.py --help
```

for the full list.

### 6.3 Outputs

Each run creates:

* CL metrics (AIS / LAST / FM / BWT)
* overall summary JSON + CSV
* log file with CLI args

Stored under: `results/<Dataset>/<method>/run-<timestamp>/`.

---

## 7. Training‑Order Protocols

Continual learning performance is highly sensitive to **crisis sequence**. Supported options:

* `alphabetical` – alphabetical order with a left shift to reach as many order as the number of crisis.
* `random` – shuffled order (use `--seed`).
* `similarity` – for each starting crisis, the order is set to be decreasing in the similarity scores between crisis.
* `custom:<path>` – newline‑separated list of task IDs.

You can create task order files with helper utilities in `src/utils.py`.

---

## 8. Batch Experiments (Run All Baselines)

Create a simple script (example):

```bash
#!/usr/bin/env bash
METHODS=(agem cumulative ewc mas si vanilla)
DATASETS=(FrenchCorpus HumAid)
for DS in "${DATASETS[@]}"; do
  for M in "${METHODS[@]}"; do
    python src/evaluate.py --method "$M" --dataset "$DS" --train_order chronological --n_epochs 10 --lr 3e-4 --batch_size 32 || exit 1
  done
done
```

Save as `scripts/run_all_baselines.sh` and make executable: `chmod +x scripts/run_all_baselines.sh`.

---

## 9. Extending the Repo

### 9.1 Add a New Method

1. Create `src/approaches/<new_method>.py`.
2. Implement a class extending the shared base (see existing files for a minimal contract: `observe(batch, task_id)`, `end_task(task_id)`, optional regularization hooks).
3. Register the method name in the CLI switch table in `evaluate.py`.

### 9.2 Add a New Dataset

1. Put raw data in `data/raw/<NewDS>/`.
2. Update `data/scripts/dataset.py` to load & split tasks.
3. Add preprocessing rules in `preprocessing.py`.
4. Create corresponding subfolders under `results/<NewDS>/<method>/`.

---

## 10. Troubleshooting & Tips

**Missing results folder?** Ensure the full path exists; use the `mkdir -p` helper above.

**CUDA out of memory?** Lower `--batch_size` or enable gradient accumulation (if available). CPU fallback is possible but slow.

**Long preprocessing time?** Run once; cached artifacts go to `data/processed/`.

**Reproducibility:** Set `--seed` and disable nondeterministic CuDNN if exact repeatability matters.

---

## 11. Citation

```bibtex
@misc{continual-crisis-nlp2025,
  author       = {Your Name},
  title        = {Continual Learning for Crisis Management NLP},
  year         = {2025},
  howpublished = {\url{https://github.com/<your-user>/continual-crisis-nlp}}
}
```
