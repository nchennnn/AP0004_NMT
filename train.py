import argparse
import logging
import math
import time
import random
import numpy as np
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
from models.rnn import Seq2Seq
from models.transformer import TransformerNMT
from transformers import T5PreTrainedModel

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
    parser = argparse.ArgumentParser(description="Train simple NMT models")
    parser.add_argument("--train-data", type=str, default="data/train_100k.jsonl", help="Path to training jsonl data")
    parser.add_argument("--val-data", type=str, default="data/valid.jsonl", help="Path to validation jsonl data")
    parser.add_argument("--test-data", type=str, default="data/test.jsonl", help="Path to test jsonl data (optional)")
    parser.add_argument("--tokenizer-name", type=str, default="utrobinmv/t5_translate_en_ru_zh_small_1024", help="Name of the T5 tokenizer model")
    parser.add_argument(
        "--tokenizer-mode",
        choices=["filtered", "hf"],
        default="filtered",
        help="filtered: use custom vocab and embedding from filtered_vocab; hf: keep original T5Tokenizer approach",
    )
    parser.add_argument(
        "--filtered-vocab-dir",
        type=str,
        default="filtered_vocab",
        help="Directory containing custom vocab and embedding (includes vocab.json / mapping.json / embedding.pt)",
    )
    parser.add_argument("--model", choices=["rnn", "transformer"], default="rnn")
    parser.add_argument("--attention", choices=["dot", "multiplicative", "additive"], default="dot")
    parser.add_argument("--emb", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--pos-embedding", choices=["absolute", "relative"], default="relative", help="Type of transformer positional embedding")
    parser.add_argument("--norm-type", choices=["layernorm", "rmsnorm"], default="rmsnorm", help="Type of transformer normalization")
    parser.add_argument("--max-pos-len", type=int, default=512, help="Maximum positional embedding length")
    parser.add_argument("--relative-num-buckets", type=int, default=32, help="Number of relative position buckets")
    parser.add_argument("--relative-max-distance", type=int, default=128, help="Maximum distance for relative position encoding")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["Adam", "AdamW"], default="AdamW", help="Optimizer to use")
    parser.add_argument("--output", type=str, default="runs")
    parser.add_argument(
        "--rnn-training-mode",
        choices=["teacher_forcing", "free_running"],
        default="teacher_forcing",
        help="Use Teacher Forcing or Free Running for RNN training",
    )
    parser.add_argument(
        "--decode-strategy",
        choices=["greedy", "beam"],
        default="beam",
        help="Decoding strategy for validation/testing",
    )
    parser.add_argument("--beam-size", type=int, default=4, help="Beam search width")
    parser.add_argument("--patience", type=int, default=3, help="Number of epochs to wait for BLEU improvement before early stopping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def build_tokenizer(args):
    if args.tokenizer_mode == "filtered":
        vocab_dir = Path(args.filtered_vocab_dir)
        return FilteredVocabTokenizer(
            args.tokenizer_name,
            vocab_dir / "vocab.json",
            vocab_dir / "mapping.json",
            vocab_dir / "embedding.pt",
        )
    return HFTokenizerWrapper(args.tokenizer_name)


def build_model(args, tokenizer):
    vocab_size = len(tokenizer.get_vocab())
    embedding_weight = tokenizer.embedding_weight
    padding_idx = tokenizer.pad_token_id
    emb_dim = embedding_weight.size(1) if embedding_weight is not None else args.emb

    if args.model == "rnn":
        return Seq2Seq(
            vocab_size,
            vocab_size,
            emb_size=emb_dim,
            hidden_size=args.hidden,
            num_layers=args.layers,
            attention=args.attention,
            embedding_weight=embedding_weight,
            padding_idx=padding_idx,
            dropout=args.dropout,
        )
    else:
        return TransformerNMT(
            vocab_size,
            vocab_size,
            d_model=emb_dim,
            nhead=args.heads,
            num_layers=args.layers,
            pos_embedding=args.pos_embedding,
            norm_type=args.norm_type,
            max_pos_len=args.max_pos_len,
            relative_num_buckets=args.relative_num_buckets,
            relative_max_distance=args.relative_max_distance,
            embedding_weight=embedding_weight,
            padding_idx=padding_idx,
        )


def run_epoch(model, dataloader, criterion, optimizer=None, device="cpu", args=None):
    total_loss = 0.0
    all_batch = len(dataloader)
    for i,batch in enumerate(dataloader):
        print(f"Batch {i+1}/{all_batch}")
        batch = batch_to_device(batch, device)
        if args and args.model == "rnn" and args.rnn_training_mode == "free_running":
            logits = model.forward_free_running(batch, max_len=batch.tgt_output.size(1))
        else:
            outputs = model(batch)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
        loss = criterion(logits.view(-1, logits.size(-1)), batch.tgt_output.reshape(-1))
        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def batch_to_device(batch, device):
    return batch.__class__(
        src=batch.src.to(device),
        tgt_input=batch.tgt_input.to(device),
        tgt_output=batch.tgt_output.to(device),
    )


def compute_metrics(predictions: List[str], references: List[str]) -> dict:
    """Compute BLEU, METEOR and ROUGE metrics"""
    assert len(predictions) == len(references), "Number of predictions and references must be the same"
    
    # Initialize ROUGE scorer
    rouge_sc = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoothing = SmoothingFunction().method1
    
    bleu_scores = []
    meteor_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, ref in zip(predictions, references):
        # BLEU score (requires tokenization)
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
        bleu_scores.append(bleu)
        
        # METEOR score (requires tokenization)
        try:
            meteor = meteor_score([ref_tokens], pred_tokens)
            meteor_scores.append(meteor)
        except Exception as e:
            # If METEOR calculation fails, use 0.0
            meteor_scores.append(0.0)
        
        # ROUGE scores
        rouge_scores = rouge_sc.score(ref, pred)
        rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
        rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
        rougeL_scores.append(rouge_scores['rougeL'].fmeasure)
    
    return {
        'bleu': sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
        'meteor': sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0,
        'rouge1': sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0,
        'rouge2': sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0,
        'rougeL': sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0,
    }


def evaluate_epoch(model, dataloader, device, criterion, max_len=128, decode_strategy="greedy", beam_size=4):
    """Compute loss and generation metrics in a single pass"""
    model.eval()
    total_loss = 0.0
    predictions = []
    references = []
    
    inference_tokenizer = InferenceTokenizer(dataloader.dataset.tokenizer)
    bos_idx = dataloader.dataset.bos_token_id
    eos_idx = dataloader.dataset.eos_token_id
    
    # Since validation/test sets have shuffle=False, order should match dataset.rows
    sample_idx = 0
    
    with torch.no_grad():
        all_batch = len(dataloader)
        for i,batch in enumerate(dataloader):
            start_time = time.time()
            print(f"Batch {i+1}/{all_batch}")
            batch = batch_to_device(batch, device)
            
            outputs = model(batch)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = criterion(logits.view(-1, logits.size(-1)), batch.tgt_output.reshape(-1))
            total_loss += loss.item()
            print(f"Batch {i+1} loss: {loss.item()} in {time.time() - start_time:.2f} seconds")
            
            # Generate predictions
            for i in range(batch.src.size(0)):
                start_time = time.time()
                print(f"Generating prediction {i+1}")
                src_seq = batch.src[i:i+1]
                if decode_strategy == "beam" and hasattr(model, "beam_search"):
                    pred_ids = model.beam_search(src_seq, beam_size, max_len, bos_idx, eos_idx)
                else:
                    pred_ids = model.greedy_decode(src_seq, max_len, bos_idx, eos_idx)
                pred_text = inference_tokenizer.decode_tgt(pred_ids)
                predictions.append(pred_text)
                
                # Get original reference text from dataset
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
    
    tokenizer = build_tokenizer(args)

    # If using pretrained embedding, ensure emb dimension is synced to args for checkpoint reproducibility
    if tokenizer.embedding_weight is not None:
        args.emb = tokenizer.embedding_weight.size(1)
    # Create three separate dataset objects
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
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=val_dataset.collate_fn) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn) if test_dataset else None

    model = build_model(args, tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.pad_token_id)
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:  # AdamW
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    save_dir = Path(args.output) / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{args.model}-{args.optimizer}-lr{args.lr}-wd{args.wd}-bs{args.batch_size}-epoch{args.epochs}-{args.attention}-layer{args.layers}-emb{args.emb}-hidden{args.hidden}-{args.pos_embedding}Pos-{args.norm_type}-{args.rnn_training_mode}-{args.decode_strategy}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
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
    logger.info("=" * 80)
    logger.info("Starting training")
    logger.info(f"Model type: {args.model}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Weight decay: {args.wd}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info("=" * 80)
    
    best_val_bleu = 0.0
    best_val_loss = float('inf')  # 保留用于记录
    epochs_no_improve = 0  # 早停计数器
    plateau_max_num = 3
    plateau_num = 0
    best_epoch = 0
    best_val_metrics = None
    
    # 初始化指标记录器
    metrics_logger = MetricsLogger()

    # Training loop
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}: Training...")
        model.train()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, args)
        
        if val_loader:
            print(f"Epoch {epoch+1}/{args.epochs}: Evaluating on validation set...")
            logger.info(f"Epoch {epoch+1}/{args.epochs}: Computing validation loss and metrics...")
            val_loss, val_metrics = evaluate_epoch(
                model,
                val_loader,
                device,
                criterion,
                decode_strategy=args.decode_strategy,
                beam_size=args.beam_size,
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
                        "tokenizer_mode": args.tokenizer_mode,
                        "filtered_vocab_dir": args.filtered_vocab_dir,
                        "src_stoi": train_dataset.src_stoi,
                        "tgt_stoi": train_dataset.tgt_stoi,
                        "tgt_itos": train_dataset.tgt_itos,
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
        test_loss, test_metrics = evaluate_epoch(
            model,
            test_loader,
            device,
            criterion,
            decode_strategy=args.decode_strategy,
            beam_size=args.beam_size,
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
