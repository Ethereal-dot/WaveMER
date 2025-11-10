# Enhanced Handwritten Mathematical Expression Recognition via Wavelet Feature Integration
Official implementation of **WaveMER**, a wavelet-enhanced dual-branch framework for Handwritten Mathematical Expression Recognition (HMER).  
WaveMER integrates spatial features and frequency-domain cues to better capture fine-grained symbols, structures, and layout in handwritten mathematical expressions.

---

## 🌟 Highlights

- Dual-branch encoder: **visual backbone** + **wavelet (DWT) frequency branch**
- Frequency-enhanced residual & attention modules for structure-sensitive modeling
- Designed for **CROHME 2014/2016/2019** and other HMER benchmarks
- Simple training & evaluation pipeline based on PyTorch & PyTorch Lightning

---

## 📂 Project Structure

```text
WaveMER
├── config/             # training & model configs (e.g., config.yaml)
├── data/
│   └── crohme/         # CROHME datasets after preparation
├── eval/               # evaluation scripts (e.g., eval_crohme.sh)
├── wavemer/            # WaveMER model implementations
├── lightning_logs/     # training logs & checkpoints
├── .gitignore
├── requirements.txt
├── train.py
└── README.md

```

## ⚙️ Install Dependencies

```
# clone this repo
git clone https://github.com/Ethereal-dot/WaveMER.git
cd WaveMER

# create and activate conda environment
conda create -y -n WaveMER python=3.7
conda activate WaveMER

# core libraries
conda install pytorch=1.8.1 torchvision=0.2.2 cudatoolkit=11.1 pillow=8.4.0 -c pytorch -c nvidia

# training dependencies
conda install pytorch-lightning=1.4.9 torchmetrics=0.6.0 -c conda-forge

# evaluation dependencies
conda install pandoc=1.19.2.1 -c conda-forge
```

## 🚀 Training

Train WaveMER using the provided configuration file:

```bash
python train.py --config config.yaml
```

## ✅ Evaluation

After training is complete (or if you already have a checkpoint), run evaluation on CROHME with:

```bash
bash eval/eval_crohme.sh 0
```
