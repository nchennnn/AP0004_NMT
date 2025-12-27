import argparse
import json
import time
import logging
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import (
    ParallelTextDataset,
    InferenceTokenizer,
    FilteredVocabTokenizer,
    HFTokenizerWrapper,
)
from src.utils import Batch, MetricsLogger
from models.rnn import Seq2Seq
from models.transformer import TransformerNMT
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
    parser = argparse.ArgumentParser(description="Evaluate NMT models on test set")
    parser.add_argument("--test-data", type=str, default="data/test.jsonl", help="Path to test jsonl data")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="evaluation_results", help="Directory to save evaluation results")
    parser.add_argument("--max-len", type=int, default=128, help="Maximum generation length")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for evaluation")
    parser.add_argument(
        "--decode-strategy",
        choices=["greedy", "beam"],
        default="beam",
        help="Decoding strategy for evaluation",
    )
    parser.add_argument("--beam-size", type=int, default=4, help="Beam search width")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def load_checkpoint(model_path: str):
    """Load model checkpoint and return model, tokenizer and args"""
    checkpoint = torch.load(model_path, map_location="cpu")
    args_dict = checkpoint["args"]

    # Create a namespace object from the args dict
    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    args = Args(**args_dict)

    # Build tokenizer
    if args.tokenizer_mode == "filtered":
        vocab_dir = Path(args.filtered_vocab_dir)
        tokenizer = FilteredVocabTokenizer(
            args.tokenizer_name,
            vocab_dir / "vocab.json",
            vocab_dir / "mapping.json",
            vocab_dir / "embedding.pt",
        )
    else:
        tokenizer = HFTokenizerWrapper(args.tokenizer_name)

    # Build model
    vocab_size = len(tokenizer.get_vocab())
    embedding_weight = tokenizer.embedding_weight
    padding_idx = tokenizer.pad_token_id
    emb_dim = embedding_weight.size(1) if embedding_weight is not None else args.emb

    if args.model == "rnn":
        model = Seq2Seq(
            vocab_size,
            vocab_size,
            emb_size=emb_dim,
            hidden_size=args.hidden,
            num_layers=args.layers,
            attention=args.attention,
            embedding_weight=embedding_weight,
            padding_idx=padding_idx,
        )
    else:
        model = TransformerNMT(
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

    # Load model state
    model.load_state_dict(checkpoint["model_state"])

    return model, tokenizer, args

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
    return avg_loss, metrics, predictions, references

def save_results(predictions: List[str], references: List[str], metrics: Dict[str, float], output_path: str):
    """Save evaluation results to a JSON file"""
    results = {
        "metrics": metrics,
        "predictions": predictions,
        "references": references,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def main():
    args = parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model, tokenizer, model_args = load_checkpoint(args.model_path)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_type = model_args.model
    decode_strategy = args.decode_strategy
    beam_size = args.beam_size if decode_strategy == "beam" else 0
    # Setup logging
    log_file = output_dir / f"{model_type}_{decode_strategy}_{beam_size}_{timestamp}_evaluation.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Loading model from {args.model_path}")

    # Create test dataset
    test_dataset = ParallelTextDataset(
        args.test_data,
        tokenizer=tokenizer,
        src_lang="zh",
        tgt_lang="en",
    )

    # Create data loader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=test_dataset.collate_fn
    )

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logger.info(f"Evaluating model {model_args.model} on test set...")
    logger.info(f"Decode strategy: {args.decode_strategy}")
    logger.info(f"Beam size: {args.beam_size}")

    # Evaluate model
    loss, metrics, predictions, references = evaluate_epoch(
        model,
        test_loader,
        device,
        nn.CrossEntropyLoss(ignore_index=test_dataset.pad_token_id),
        decode_strategy=args.decode_strategy,
        beam_size=args.beam_size,
    )

    # Log results
    logger.info("=" * 80)
    logger.info("Test set results:")
    logger.info(f"  Test loss: {loss:.4f}")
    logger.info(f"  BLEU: {metrics['bleu']:.4f}")
    logger.info(f"  METEOR: {metrics['meteor']:.4f}")
    logger.info(f"  ROUGE-1: {metrics['rouge1']:.4f}")
    logger.info(f"  ROUGE-2: {metrics['rouge2']:.4f}")
    logger.info(f"  ROUGE-L: {metrics['rougeL']:.4f}")
    logger.info("=" * 80)


    results_file = output_dir / f"{model_type}_{decode_strategy}_{beam_size}_{timestamp}.json"
    save_results(predictions, references, metrics, str(results_file))

    logger.info(f"Results saved to {results_file}")
    logger.info("Evaluation completed!")

if __name__ == "__main__":
    main()
