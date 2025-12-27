from dataclasses import dataclass
from typing import List, Iterable

import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"


def whitespace_tokenize(text: str) -> List[str]:
    return text.strip().split()


def build_vocab(sentences: Iterable[List[str]], min_freq: int = 1):
    freq = {}
    for sent in sentences:
        for tok in sent:
            freq[tok] = freq.get(tok, 0) + 1
    vocab = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
    vocab += sorted([t for t, c in freq.items() if c >= min_freq])
    stoi = {t: i for i, t in enumerate(vocab)}
    itos = {i: t for t, i in stoi.items()}
    return stoi, itos


def tokens_to_ids(tokens: List[str], stoi):
    return [stoi.get(tok, stoi[UNK_TOKEN]) for tok in tokens]


def ids_to_tokens(ids: List[int], itos):
    return [itos[i] for i in ids]


@dataclass
class Batch:
    src: any
    tgt_input: any
    tgt_output: any


def subsequent_mask(size: int):
    mask = (1 - torch.triu(torch.ones(1, size, size), diagonal=1)).bool()
    return mask


class MetricsLogger:
    """记录训练过程中的各项指标并绘制图表"""
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.bleu_scores = []
        self.meteor_scores = []
        self.rouge1_scores = []
        self.rouge2_scores = []
        self.rougeL_scores = []
        self.epochs = []
    
    def log_epoch(self, epoch, train_loss, val_loss=None, metrics=None):
        """记录一个epoch的数据"""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if metrics is not None:
            self.bleu_scores.append(metrics.get('bleu', 0))
            self.meteor_scores.append(metrics.get('meteor', 0))
            self.rouge1_scores.append(metrics.get('rouge1', 0))
            self.rouge2_scores.append(metrics.get('rouge2', 0))
            self.rougeL_scores.append(metrics.get('rougeL', 0))
    
    def plot_metrics(self, save_path):
        """绘制并保存指标图表"""
        # 创建一个大图，包含多个子图
        plt.figure(figsize=(15, 10))
        
        # 绘制Loss
        plt.subplot(2, 3, 1)
        plt.plot(self.epochs, self.train_losses, 'b-o', label='Train Loss')
        if self.val_losses:
            plt.plot(self.epochs, self.val_losses, 'r-o', label='Val Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 绘制BLEU
        plt.subplot(2, 3, 2)
        if self.bleu_scores:
            plt.plot(self.epochs, self.bleu_scores, 'g-o')
            plt.title('BLEU Score')
            plt.xlabel('Epoch')
            plt.ylabel('BLEU')
            plt.grid(True)
        
        # 绘制METEOR
        plt.subplot(2, 3, 3)
        if self.meteor_scores:
            plt.plot(self.epochs, self.meteor_scores, 'm-o')
            plt.title('METEOR Score')
            plt.xlabel('Epoch')
            plt.ylabel('METEOR')
            plt.grid(True)
        
        # 绘制ROUGE-1
        plt.subplot(2, 3, 4)
        if self.rouge1_scores:
            plt.plot(self.epochs, self.rouge1_scores, 'c-o')
            plt.title('ROUGE-1 Score')
            plt.xlabel('Epoch')
            plt.ylabel('ROUGE-1')
            plt.grid(True)
        
        # 绘制ROUGE-2
        plt.subplot(2, 3, 5)
        if self.rouge2_scores:
            plt.plot(self.epochs, self.rouge2_scores, 'y-o')
            plt.title('ROUGE-2 Score')
            plt.xlabel('Epoch')
            plt.ylabel('ROUGE-2')
            plt.grid(True)
        
        # 绘制ROUGE-L
        plt.subplot(2, 3, 6)
        if self.rougeL_scores:
            plt.plot(self.epochs, self.rougeL_scores, 'k-o')
            plt.title('ROUGE-L Score')
            plt.xlabel('Epoch')
            plt.ylabel('ROUGE-L')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
