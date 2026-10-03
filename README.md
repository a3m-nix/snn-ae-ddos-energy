# SNN-AE: Energy-Aware Spiking Autoencoder for DDoS Anomaly Detection in IoT Edge Networks

Code and results for the article "Energy-Aware Spiking Autoencoder for Reconstruction-Based
DDoS Anomaly Detection in IoT Edge Networks" (Scientific Reports, under revision).

## Overview
SNN-AE is a spiking autoencoder with Leaky Integrate-and-Fire neurons, trained on normal
traffic only. Anomalies are detected from the reconstruction error with a p99 threshold of
validation-normal errors. Energy is reported at three tiers (T1, T2, T3) and validated on a
Raspberry Pi 5.

## Repository structure
```text
scripts/            Pipeline scripts S1–S8 (see below)
scripts/rpi/        Raspberry Pi 5 scripts (batched inference, end-to-end pipeline)
notebooks/          Notebook used to produce the figures and tables
00-config/          Experiment configuration (config_revision.json)
01-audit/           Data audit outputs
02-splits/          Flow-level split indices and dataset summaries
03-tuning/          Optuna databases and trial histories (tuning logs)
04-final-eval/      Results per seed and fold; full outputs of the deployed fold (s0_f0)
05-tables/          Detection tables and statistical tests
06-posthoc/         Latent analysis, cross-dataset test, Brian2 check, end-to-end references
07-energy/          Energy per fold, tests, and break-even analysis
08-rpi/             Raspberry Pi 5 results
pilot-spike-aware/  Spike-aware pilot on the holdout set
article-reports/    Figures and tables of the article and supplementary material
```

## Pipeline
```text
S1  s1_sampling_split.py       Balanced sampling, flow-level holdout and CV splits
S2  s2_tuning.py               Hyperparameter tuning on the holdout (Optuna, 50 trials per model)
S3  s3_final_eval.py           Training and evaluation, 3 seeds x 10 folds
S4  s4_tables_stats.py         Tables, statistical tests, bootstrap CIs
S5  s5_posthoc.py              Cross-dataset test and other post-hoc analyses
S6  s6_energy.py               Energy at tiers T1–T3 and break-even spike budget
S7  rpi/s7a_..., rpi/s7b_...   Raspberry Pi 5 validation; s7b also implements the
                               PCAP-to-feature extraction with TShark
S8  s8_latent_analysis.py      Latent-space analysis of SNN-AE
```

## Environment
Python 3.10.12, TensorFlow/Keras 2.12.0, NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.5.2,
SciPy 1.15.1, Optuna 4.8.0, umap-learn 0.5.7.

## Data
The raw Edge-IIoTset and CICIoT2023 datasets are publicly available from their original
sources and are not included here. The balanced samples (sample.parquet) are not included
because of the GitHub file size limit; they can be rebuilt with S1.

## Files not included
Trained models and per-sample scores of all folds other than s0_f0 exceed the size limit
of GitHub. They are available from the corresponding author on request.

## License
MIT License. See LICENSE.
