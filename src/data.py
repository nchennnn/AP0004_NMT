import json
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer

from src.utils import Batch


class BaseTokenizerWrapper:
    """Unified wrapper for easy switching between different tokenizer implementations during training/inference."""

    pad_token_id: int
    eos_token_id: int
    bos_token_id: int

    def get_vocab(self) -> Dict[str, int]:
        raise NotImplementedError

    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        raise NotImplementedError

    def encode_single(self, text: str) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError

    @property
    def embedding_weight(self) -> Optional[torch.Tensor]:
        """Optional pretrained embedding, returns None if not available."""
        return None


class HFTokenizerWrapper(BaseTokenizerWrapper):
    """Maintains original T5Tokenizer behavior as a fallback option."""

    def __init__(self, tokenizer_name: str):
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.bos_token_id = (
            self.tokenizer.bos_token_id
            if hasattr(self.tokenizer, "bos_token_id")
            else self.tokenizer.pad_token_id
        )

    def get_vocab(self) -> Dict[str, int]:
        return self.tokenizer.get_vocab()

    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return encoded["input_ids"]

    def encode_single(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        return encoded["input_ids"]

    def decode(self, ids: List[int]) -> str:
        ids_tensor = torch.tensor([ids], dtype=torch.long)
        return self.tokenizer.decode(ids_tensor[0], skip_special_tokens=True)


class FilteredVocabTokenizer(BaseTokenizerWrapper):
    """
    Uses custom vocab and embedding from the filtered_vocab directory.
    First tokenizes with original T5Tokenizer, then maps to new vocab via old->new mapping.
    """

    def __init__(
        self,
        tokenizer_name: str,
        vocab_path: Path,
        mapping_path: Path,
        embedding_path: Path,
    ):
        self.base_tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        self.vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))

        mapping_raw = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
        self.old_to_new = {int(k): int(v) for k, v in mapping_raw["old_to_new"].items()}
        self.new_to_old = {int(v): int(k) for k, v in mapping_raw["old_to_new"].items()}

        state = torch.load(embedding_path, map_location="cpu")
        self._embedding_weight = state["embedding"].clone()

        self.pad_token_id = self.vocab.get("<pad>", 0)
        self.eos_token_id = self.vocab.get("</s>", 1)
        # Filtered vocab usually doesn't have explicit <s>, use pad as bos placeholder
        self.bos_token_id = self.vocab.get("<s>", self.pad_token_id)
        self.unk_token_id = self.vocab.get("<unk>", self.pad_token_id)

    @property
    def embedding_weight(self) -> Optional[torch.Tensor]:
        return self._embedding_weight

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab

    def _map_ids(self, ids: List[int]) -> List[int]:
        return [self.old_to_new.get(int(t), self.unk_token_id) for t in ids]

    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        encoded = self.base_tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        mapped = [self._map_ids(seq) for seq in encoded["input_ids"].tolist()]
        return torch.tensor(mapped, dtype=torch.long)

    def encode_single(self, text: str) -> torch.Tensor:
        encoded = self.base_tokenizer(text, return_tensors="pt", add_special_tokens=True)
        mapped = self._map_ids(encoded["input_ids"][0].tolist())
        return torch.tensor([mapped], dtype=torch.long)

    def decode(self, ids: List[int]) -> str:
        old_ids = [self.new_to_old.get(int(t), self.base_tokenizer.unk_token_id) for t in ids]
        ids_tensor = torch.tensor([old_ids], dtype=torch.long)
        return self.base_tokenizer.decode(ids_tensor[0], skip_special_tokens=True)


class ParallelTextDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: BaseTokenizerWrapper,
        src_lang: str = "zh",
        tgt_lang: str = "en",
        vocab_dataset: Optional["ParallelTextDataset"] = None,
        preload_cache: bool = True,
    ):
        """
        Create a parallel text dataset using a tokenizer wrapper.
        Args:
            data_path: Path to jsonl file containing parallel text pairs
            tokenizer: Tokenizer wrapper to use
            src_lang: Source language field name in jsonl
            tgt_lang: Target language field name in jsonl
            vocab_dataset: Optional dataset to share tokenizer with (for validation/test sets)
            preload_cache: If True, preload and cache all tokenized data in memory for faster access
        """
        # Load raw text data from jsonl and cache in memory
        self.rows = self._load_jsonl(data_path, src_lang, tgt_lang)

        if vocab_dataset is not None:
            # Use tokenizer from another dataset (for validation/test sets)
            self.tokenizer = vocab_dataset.tokenizer
        else:
            self.tokenizer = tokenizer

        # Store vocabulary mappings for compatibility with existing code
        self.src_stoi = self.tokenizer.get_vocab()
        self.tgt_stoi = self.tokenizer.get_vocab()  # Same vocab for both languages
        self.src_itos = {v: k for k, v in self.src_stoi.items()}
        self.tgt_itos = {v: k for k, v in self.tgt_stoi.items()}

        # Store special token IDs
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        
        # Preload and cache tokenized data for faster access
        self.preload_cache = preload_cache
        if self.preload_cache:
            self._cache_tokenized_data()
    
    def _load_jsonl(self, path: str, src_lang: str, tgt_lang: str) -> List[Tuple[str, str]]:
        """Load parallel text pairs from a jsonl file and cache in memory."""
        rows = []
        for line in Path(path).read_text(encoding="utf-8").strip().splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            src_text = data.get(src_lang, "")
            tgt_text = data.get(tgt_lang, "")
            if src_text and tgt_text:
                rows.append((src_text, tgt_text))
        return rows
    
    def _cache_tokenized_data(self):
        """Preload and tokenize all data into memory cache for faster access."""
        print(f"Preloading and caching tokenized data for {len(self.rows)} samples...")
        self.src_ids_cache = []
        self.tgt_ids_cache = []
        
        # Tokenize each sample individually to avoid unnecessary padding
        # Store as lists of token IDs (unpadded) for efficient memory usage
        for src_text, tgt_text in self.rows:
            # Use encode_single and extract the sequence (remove batch dimension)
            src_ids = self.tokenizer.encode_single(src_text).squeeze(0)  # Shape: (seq_len,)
            tgt_ids = self.tokenizer.encode_single(tgt_text).squeeze(0)  # Shape: (seq_len,)
            
            self.src_ids_cache.append(src_ids.tolist())
            self.tgt_ids_cache.append(tgt_ids.tolist())
        
        print(f"Caching completed. {len(self.src_ids_cache)} samples cached in memory.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        if self.preload_cache and hasattr(self, 'src_ids_cache'):
            # Return cached tokenized data as lists of token IDs
            return (self.src_ids_cache[idx], self.tgt_ids_cache[idx])
        else:
            # Return raw text (will be tokenized in collate_fn)
            src, tgt = self.rows[idx]
            return src, tgt

    def collate_fn(self, batch):
        # Check if batch contains cached token IDs (lists) or raw text strings
        if isinstance(batch[0][0], list):
            # Batch contains cached tokenized data as lists
            src_ids_list = [torch.tensor(item[0], dtype=torch.long) for item in batch]
            tgt_ids_list = [torch.tensor(item[1], dtype=torch.long) for item in batch]
            
            # Pad sequences to same length
            max_src_len = max(seq.size(0) for seq in src_ids_list)
            max_tgt_len = max(seq.size(0) for seq in tgt_ids_list)
            
            src_ids = torch.full((len(batch), max_src_len), self.pad_token_id, dtype=torch.long)
            tgt_ids = torch.full((len(batch), max_tgt_len), self.pad_token_id, dtype=torch.long)
            
            for i, (src_seq, tgt_seq) in enumerate(zip(src_ids_list, tgt_ids_list)):
                src_ids[i, :src_seq.size(0)] = src_seq
                tgt_ids[i, :tgt_seq.size(0)] = tgt_seq
        else:
            # Batch contains raw text strings, need to tokenize
            src_batch, tgt_batch = zip(*batch)
            src_ids = self.tokenizer.encode_batch(list(src_batch))
            tgt_ids = self.tokenizer.encode_batch(list(tgt_batch))

        # Explicitly add <s>/bos for decoder to ensure consistency between training and inference start tokens
        bos_col = torch.full((tgt_ids.size(0), 1), self.bos_token_id, dtype=torch.long)
        tgt_with_bos = torch.cat([bos_col, tgt_ids], dim=1)  # Add <s> before decoder input sequence. Note: T5's decoder input sequence doesn't need to include <s>, and </s> will be automatically added by T5's Tokenizer
        tgt_input = tgt_with_bos[:, :-1]
        tgt_output = tgt_with_bos[:, 1:]

        return Batch(src_ids, tgt_input, tgt_output)

        
    def collate_fn_mt5(self, batch):
        # Check if batch contains cached token IDs (lists) or raw text strings
        if isinstance(batch[0][0], list):
            # Batch contains cached tokenized data as lists
            src_ids_list = [torch.tensor(item[0], dtype=torch.long) for item in batch]
            tgt_ids_list = [torch.tensor(item[1], dtype=torch.long) for item in batch]
            
            # Pad sequences to same length
            max_src_len = max(seq.size(0) for seq in src_ids_list)
            max_tgt_len = max(seq.size(0) for seq in tgt_ids_list)
            
            src_ids = torch.full((len(batch), max_src_len), self.pad_token_id, dtype=torch.long)
            tgt_ids = torch.full((len(batch), max_tgt_len), self.pad_token_id, dtype=torch.long)
            
            for i, (src_seq, tgt_seq) in enumerate(zip(src_ids_list, tgt_ids_list)):
                src_ids[i, :src_seq.size(0)] = src_seq
                tgt_ids[i, :tgt_seq.size(0)] = tgt_seq
        else:
            # Batch contains raw text strings, need to tokenize
            src_batch, tgt_batch = zip(*batch)
            # Add "translate to English: " prefix to Chinese text
            src_batch_with_prefix = [f"translate to English: {src}" for src in src_batch]
            src_ids = self.tokenizer.encode_batch(src_batch_with_prefix)
            tgt_ids = self.tokenizer.encode_batch(list(tgt_batch))

        tgt_input = tgt_ids
        tgt_output = tgt_ids

        return Batch(src_ids, tgt_input, tgt_output)


class InferenceTokenizer:
    def __init__(self, tokenizer: BaseTokenizerWrapper):
        self.tokenizer = tokenizer

    def encode_src(self, text: str):
        return self.tokenizer.encode_single(text)

    def encode_src_mt5(self, text: str):
        # Add "translate to English: " prefix to Chinese text
        prefixed_text = f"translate to English: {text}"
        return self.tokenizer.encode_single(prefixed_text)

    def decode_tgt(self, ids: List[int]):
        return self.tokenizer.decode(ids)
