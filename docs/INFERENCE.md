# Inference Guide

This document covers running inference with RADAR, including a quick demo and evaluation on the MERLIN test set.

> **Prerequisite:** Complete the [Setup](../README.md#setup) section in the main README first.

---

## Table of Contents

- [Inference Demo](#inference-demo)
- [Inference on the MERLIN Test Set](#inference-on-the-merlin-test-set)

---

## Inference Demo

### 1. Download Required Files and Place Them to Target folder

| File                                      | Description                                          | Destination                                                  |
| ----------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------ |
| Pretrained checkpoint                     | RADAR model weights                                  | `radar/RADAR_inference/checkpoint/checkpoint_radar_pretrain.pth`   |
| Demo cases                                | Sample CT examinations                               | `radar/RADAR_inference/demo_cases/`                                |
| `bert-base-chinese`                       | BERT tokenizer and model                             | `radar/RADAR_inference/configs/bert-base-chinese/`                 |
| `RADAR_infer_results_MerlinTestset.csv`   | The inference results of RADAR on Merlin-CT-Test set | `radar/RADAR_inference/RADAR_infer_results_MerlinTestset.csv`      |

All files are available on HuggingFace.

### 2. Run Inference

```bash
cd RADAR_inference
python inference_demo.py
```

The results will be saved as `RADAR_infer_results_demo.csv`, which contains the positive scores for each finding.

---

## Inference on the MERLIN Test Set

### 1. Prepare Data

- Download the MERLIN dataset from the [Stanford AIMI Shared Datasets](https://huggingface.co/datasets/stanfordaimi/merlin).
- Modify `--img_dir` to the MERLIN data path on your local device in `inference_merlin_testset.py`.
- We provide all MERLIN reports in JSON format (`merlin_report.json`), which includes the test split information.
- The labels have also been converted to JSON format for convenience (`merlin_labels.json`).

### 2. Run Inference

Single-GPU:

```bash
cd RADAR_inference
python inference_merlin_testset.py
```

Multi-GPU (recommended):

```bash
cd RADAR_inference
torchrun --nproc_per_node=8 inference_merlin_testset.py
```

The inference results will be saved as a CSV file.

### 3. Evaluate Performance

Compute performance metrics using the inference CSV and the label file:

```bash
python calc_metrics_merlin_testset.py
```

---

[← Back to README](../README.md)
