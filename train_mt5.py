import argparse
import logging
import time
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import MT5ForConditionalGeneration

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from src.data import (
    ParallelTextDataset,
    InferenceTokenizer,
    FilteredVocabTokenizer,
    HFTokenizerWrapper,
)
from src.utils import MetricsLogger
from train import compute_metrics, batch_to_device

# METEOR needs to download data, skip if it fails
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("downloading nltk punkt")
    nltk.download('punkt', quiet=True)
    print("done")
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("downloading nltk wordnet")
    nltk.download('wordnet', quiet=True)
    print("done")

def parse_args():
    parser = argparse.ArgumentParser(description="Train NMT using HuggingFace mT5-small")
    parser.add_argument("--train-data", type=str, default="data/train_100k.jsonl", help="Path to training set")
    parser.add_argument("--val-data", type=str, default="data/valid.jsonl", help="Path to validation set")
    parser.add_argument("--test-data", type=str, default="data/test.jsonl", help="Path to test set")
    parser.add_argument("--tokenizer-name", type=str, default="google/mt5-small", help="Name of the mT5 tokenizer")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default="runs_mt5")
    parser.add_argument("--decode-strategy", choices=["greedy", "beam"], default="beam")
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3, help="Number of epochs to wait for BLEU improvement before early stopping")
    parser.add_argument("--max-generate-len", type=int, default=128, help="Maximum generation length during validation/testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def run_epoch_mt5(model, dataloader, optimizer, device, pad_token_id):
    model.train()
    total_loss = 0.0
    all_batch = len(dataloader)
    for i,batch in enumerate(dataloader):
        print(f"Batch {i+1}/{all_batch}")
        batch = batch_to_device(batch, device)
        labels = batch.tgt_output.clone()
        labels = labels.masked_fill(labels == pad_token_id, -100)
        outputs = model(
            input_ids=batch.src,
            attention_mask=batch.src != pad_token_id,
            labels=labels,
        )
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_mt5(model, dataloader, device, pad_token_id, max_len, decode_strategy, beam_size):
    model.eval()
    total_loss = 0.0
    predictions: List[str] = []
    references: List[str] = []
    inference_tokenizer = InferenceTokenizer(dataloader.dataset.tokenizer)
    # Since validation/test sets have shuffle=False, order should match dataset.rows
    sample_idx = 0
    with torch.no_grad():
        all_batch = len(dataloader)
        for batch_idx, batch in enumerate(dataloader):
            print(f"Batch {batch_idx+1}/{all_batch}")
            batch = batch_to_device(batch, device)
            labels = batch.tgt_output.clone()
            labels = labels.masked_fill(labels == pad_token_id, -100)
            outputs = model(
                input_ids=batch.src,
                attention_mask=batch.src != pad_token_id,
                labels=labels,
            )
            total_loss += outputs.loss.item()
            print("Generating predictions...")

            for i in range(batch.src.size(0)):
                start_time = time.time()
                print(f"Generating prediction {i+1}")
                src_seq = batch.src[i:i+1]
                attn_mask = src_seq != pad_token_id
                num_beams = beam_size if decode_strategy == "beam" else 1
                generated = model.generate(
                    input_ids=src_seq,
                    attention_mask=attn_mask,
                    num_beams=num_beams,
                    max_new_tokens=max_len,
                    early_stopping=True,
                    decoder_start_token_id=pad_token_id,    
                )
                pred_ids = generated[0].tolist()
                pred_text = inference_tokenizer.decode_tgt(pred_ids)
                predictions.append(pred_text)
                _, ref_text = dataloader.dataset.rows[sample_idx]
                references.append(ref_text)
                sample_idx += 1
                print(f"Prediction {i+1} generated in {time.time() - start_time:.2f} seconds")

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    metrics = compute_metrics(predictions, references)
    return avg_loss, metrics


def train(args):
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    tokenizer = HFTokenizerWrapper(args.tokenizer_name)

    train_dataset = ParallelTextDataset(
        args.train_data,
        tokenizer=tokenizer,
        src_lang="zh",
        tgt_lang="en",
    )
    val_dataset = ParallelTextDataset(
        args.val_data,
        tokenizer=tokenizer,
        src_lang="zh",
        tgt_lang="en",
        vocab_dataset=train_dataset,
    ) if args.val_data else None
    test_dataset = ParallelTextDataset(
        args.test_data,
        tokenizer=tokenizer,
        src_lang="zh",
        tgt_lang="en",
        vocab_dataset=train_dataset,
    ) if args.test_data else None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn_mt5)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=val_dataset.collate_fn_mt5) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn_mt5) if test_dataset else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MT5ForConditionalGeneration.from_pretrained(args.tokenizer_name)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    save_dir = Path(args.output) / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-mt5-small-lr{args.lr}-wd{args.wd}-bs{args.batch_size}-epoch{args.epochs}"
    save_dir.mkdir(parents=True, exist_ok=True)

    log_file = save_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("Arguments: %s", vars(args))
    logger.info("Start mT5-small training")
    
    best_val_bleu = 0.0
    best_val_loss = float('inf')  # 保留用于记录
    epochs_no_improve = 0  # 早停计数器
    plateau_max_num = 3
    plateau_num = 0
    best_epoch = 0
    best_val_metrics = None
    
    # 初始化指标记录器
    metrics_logger = MetricsLogger()
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}: Training...")
        train_loss = run_epoch_mt5(model, train_loader, optimizer, device, tokenizer.pad_token_id)
        logger.info(f"Epoch {epoch+1}: train loss {train_loss:.4f}")

        if val_loader:
            print(f"Epoch {epoch+1}/{args.epochs}: Evaluating on validation set...")
            logger.info(f"Epoch {epoch+1}/{args.epochs}: Computing validation loss and metrics...")
            val_loss, val_metrics = evaluate_mt5(
                model,
                val_loader,
                device,
                tokenizer.pad_token_id,
                args.max_generate_len,
                args.decode_strategy,
                args.beam_size,
            )
            logger.info(f"Epoch {epoch+1}/{args.epochs}:")
            logger.info(f"  Train loss: {train_loss:.4f}")
            logger.info(f"  Validation loss: {val_loss:.4f}")
            logger.info(f"  BLEU: {val_metrics['bleu']:.4f}")
            logger.info(f"  METEOR: {val_metrics['meteor']:.4f}")
            logger.info(f"  ROUGE-1: {val_metrics['rouge1']:.4f}")
            logger.info(f"  ROUGE-2: {val_metrics['rouge2']:.4f}")
            logger.info(f"  ROUGE-L: {val_metrics['rougeL']:.4f}")
            
            print(f"Epoch {epoch+1}/{args.epochs}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
            print(f"  BLEU: {val_metrics['bleu']:.4f}, METEOR: {val_metrics['meteor']:.4f}, ROUGE-L: {val_metrics['rougeL']:.4f}")
            
            # 记录当前epoch的指标
            metrics_logger.log_epoch(epoch + 1, train_loss, val_loss, val_metrics)
            
            # Save best model based on BLEU score
            if val_metrics['bleu'] > best_val_bleu:
                best_val_bleu = val_metrics['bleu']
                best_val_loss = val_loss  # 保存对应的验证损失
                best_epoch = epoch + 1
                best_val_metrics = val_metrics
                epochs_no_improve = 0  # 重置早停计数器
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "tokenizer_name": args.tokenizer_name,
                        "args": vars(args),
                        "epoch": epoch + 1,
                        "val_loss": val_loss,
                        "val_metrics": val_metrics,
                    },
                    save_dir / "best_checkpoint.pt",
                )
                logger.info(f"  -> Best model saved (BLEU: {val_metrics['bleu']:.4f})")
                print(f"  -> Best model saved (BLEU: {val_metrics['bleu']:.4f})")
            else:
                epochs_no_improve += 1
                logger.info(f"  -> No improvement in BLEU for {epochs_no_improve} epoch(s)")
                
                # 学习率调整策略：当epochs_no_improve达到patience时，将学习率减半
                if epochs_no_improve == args.patience:
                    # 早停检查
                    if plateau_num == plateau_max_num:
                        logger.info(f"Early stopping triggered after {args.patience} epochs without BLEU improvement")
                        print(f"Early stopping triggered after {args.patience} epochs without BLEU improvement")
                        break

                    for param_group in optimizer.param_groups:
                        old_lr = param_group['lr']
                        param_group['lr'] = old_lr * 0.5
                    logger.info(f"  -> Learning rate reduced from {old_lr} to {old_lr * 0.5} due to plateau")
                    plateau_num +=1
                    epochs_no_improve = 0  # 重置早停计数器
                
        else:
            # 记录只有训练损失的情况
            metrics_logger.log_epoch(epoch + 1, train_loss)
            logger.info(f"Epoch {epoch+1}/{args.epochs}: Train loss: {train_loss:.4f}")
            print(f"Epoch {epoch+1}/{args.epochs}: train loss {train_loss:.4f}")

    # Load best model and evaluate on test set
    if val_loader and test_loader:
        logger.info("=" * 80)
        logger.info(f"Loading best model (Epoch {best_epoch}, BLEU: {best_val_bleu:.4f}, validation loss: {best_val_loss:.4f})")
        print(f"\nLoading best model from epoch {best_epoch} (BLEU: {best_val_bleu:.4f}, val loss: {best_val_loss:.4f})")
        checkpoint = torch.load(save_dir / "best_checkpoint.pt")
        model.load_state_dict(checkpoint["model_state"])
        
        logger.info("Computing test set loss and metrics...")
        test_loss, test_metrics = evaluate_mt5(
            model,
            test_loader,
            device,
            tokenizer.pad_token_id,
            args.max_generate_len,
            args.decode_strategy,
            args.beam_size,
        )
        
        logger.info("=" * 80)
        logger.info("Test set results:")
        logger.info(f"  Test loss: {test_loss:.4f}")
        logger.info(f"  BLEU: {test_metrics['bleu']:.4f}")
        logger.info(f"  METEOR: {test_metrics['meteor']:.4f}")
        logger.info(f"  ROUGE-1: {test_metrics['rouge1']:.4f}")
        logger.info(f"  ROUGE-2: {test_metrics['rouge2']:.4f}")
        logger.info(f"  ROUGE-L: {test_metrics['rougeL']:.4f}")
        logger.info("=" * 80)
        
        print(f"Test loss: {test_loss:.4f}")
        print(f"  BLEU: {test_metrics['bleu']:.4f}, METEOR: {test_metrics['meteor']:.4f}, ROUGE-L: {test_metrics['rougeL']:.4f}")
    
    # 绘制并保存训练指标图表
    metrics_plot_path = save_dir / "training_metrics.png"
    metrics_logger.plot_metrics(metrics_plot_path)
    logger.info(f"Training metrics plot saved to {metrics_plot_path}")
    print(f"Training metrics plot saved to {metrics_plot_path}")
    
    logger.info("Training completed!")



if __name__ == "__main__":
    train(parse_args())
