# Neural Machine Translation Project (SLAI AP0004)

This repository contains an end-to-end Neural Machine Translation (NMT) project implemented in PyTorch. It includes two main model variants:

- **Seq2Seq RNN-based NMT** with multiple attention mechanisms and decoding strategies
- **Transformer-based NMT** built on `torch.nn.Transformer` with positional encoding and normalization variants

The project supports training from scratch and evaluating trained checkpoints on the test set.

---

## Project Structure

```
.
├── data/
│   ├── ...                     # The training/validation/test datasets
│
├── evaluation_results/
│   ├── ...                     # The evaluation results of scratch RNN and Transformer
│
├── filtered_vocab/
│   ├── ...                     # Filtered vocabulary files and pretrained embedding weights
│
├── models/
│   ├── rnn.py                  # Seq2Seq RNN encoder-decoder with attention
│   └── transformer.py          # Transformer encoder-decoder (absolute/relative PE, LayerNorm/RMSNorm)
│
├── runs/
│   ├── ...                     # The saved checkpoints of scratch RNN and Transformer (with logs and pictures of training processes)
│
├── runs_mt5/
│   ├── ...                     # The saved log and picture of fine-tuned mT5-small
│
├── src/
│   ├── data.py                 # Tokenizer + Dataset implementations, padding and batching logic
│   └── utils.py                # Metrics logger, BLEU/METEOR/ROUGE evaluation, visualization utilities
│
├── filter_vocab_embed.py       # Vocabulary filtering and pretrained embedding extraction script
├── train.py                    # Training script for RNN and Transformer from scratch
├── train_mt5.py                # Fine-tuning script for pretrained google/mt5-small
├── inference.py                # Run inference/evaluation on test set and save outputs
└── README.md
```

---

## Training

### 1) Train Seq2Seq RNN / Transformer from Scratch

Use `train.py` to train either the RNN-based model or the Transformer-based model (depending on the configuration and arguments in the script). This script performs:

- tokenizer / dataset loading
- model initialization (optionally with pretrained embeddings from `filtered_vocab/`)
- training with early stopping and learning-rate decay
- evaluation on the validation set per epoch (BLEU + auxiliary metrics)

Run:

```bash
python train.py --model rnn
python train.py --model transformer
```

> **Note:** Most hyperparameters (model type, attention variant, decoding policy, normalization, etc.) are configured inside `train.py` or passed via arguments depending on your implementation. Please edit or override them according to your experiment needs.

---

### 2) Fine-tune Pretrained mT5-small

Use `train_mt5.py` to fine-tune the pretrained multilingual model `google/mt5-small` for the translation task. This script follows a training pipeline similar to `train.py`, but leverages Hugging Face model loading and pretrained initialization.

Run:

```bash
python train_mt5.py
```

---

## Evaluation / Validation (Inference)

Use `inference.py` to load saved checkpoints and evaluate translation performance on the test set. The script supports:

- computing BLEU / METEOR / ROUGE scores
- saving generated translations for qualitative inspection

Run:

```bash
python inference.py --model-path runs/20251225-101412-transformer-AdamW-lr0.0001-wd0.0001-bs128-epoch100-additive-layer6-emb512-hidden512-absolutePos-layernorm-teacher_forcing-beam/best_checkpoint.pt
python inference.py --model-path runs/20251225-170530-rnn-AdamW-lr0.002-wd0.0001-bs128-epoch100-additive-layer2-emb512-hidden512-relativePos-rmsnorm-teacher_forcing-beam/best_checkpoint.pt
```

The evaluation results and generated outputs will be saved to the designated output directory configured inside the script.

---

## Environment Setup

> - CUDA 12.4
> - Ubuntu 20.04
> - Python 3.10.18
> - PyTorch 2.6.0
> - The complete environment has been written in requirements.txt