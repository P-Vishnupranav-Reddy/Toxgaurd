"""Tokenizer wrapper for ToxGuard, reusing IUPACGPT's SentencePiece tokenizer."""

import os
import re
import torch
from transformers import T5Tokenizer


class ToxGuardTokenizer(T5Tokenizer):
    """T5-based SentencePiece tokenizer for IUPAC names.
    
    Identical to the IUPACGPT tokenizer — replaces spaces with underscores
    before tokenization(IUPAC names can contain spaces), and reverses on decode.
    """
    
    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        return re.sub(" ", "_", text), kwargs
    
    def _decode(self, *args, **kwargs):
        text = super()._decode(*args, **kwargs)
        text = re.sub("extra_id_", "extraAidA", text)
        text = re.sub("_", " ", text)
        text = re.sub("extraAidA", "extra_id_", text)
        return text
    
    def sentinels(self, sentinel_ids):
        return self.vocab_size - sentinel_ids - 1
    
    def sentinel_mask(self, ids):
        return ((self.vocab_size - self._extra_ids <= ids) &
                (ids < self.vocab_size))
    
    def _tokenize(self, text, sample=False):
        pieces = super()._tokenize(text)
        # SentencePiece adds a non-printing token at start — remove it
        return pieces[1:]


def get_tokenizer(
    vocab_path: str = None,
    serialized_path: str = None,
    iupacgpt_dir: str = None,
) -> ToxGuardTokenizer:
    """Load the IUPAC SentencePiece tokenizer.
    
    Priority:
      1. serialized_path (torch.load of a saved tokenizer)  
      2. vocab_path (iupac_spm.model file)
      3. iupacgpt_dir (auto-detect from IUPACGPT installation)
    
    Returns:
        ToxGuardTokenizer instance
    """
    if serialized_path and os.path.exists(serialized_path):
        tokenizer = torch.load(serialized_path, map_location="cpu", weights_only=False)
        # Re-wrap as ToxGuardTokenizer if needed
        if not isinstance(tokenizer, ToxGuardTokenizer):
            # The saved tokenizer is a T5IUPACTokenizer; it's compatible
            return tokenizer
        return tokenizer
    
    if vocab_path and os.path.exists(vocab_path):
        return ToxGuardTokenizer(vocab_file=vocab_path)
    
    if iupacgpt_dir:
        # Try to find the vocab file in IUPACGPT directory
        candidates = [
            os.path.join(iupacgpt_dir, "iupac_gpt", "iupac_spm.model"),
            os.path.join(iupacgpt_dir, "iupac_spm.model"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return ToxGuardTokenizer(vocab_file=path)
        
        # Try serialized tokenizer
        serial_candidates = [
            os.path.join(iupacgpt_dir, "iupac_gpt", "real_iupac_tokenizer.pt"),
            os.path.join(iupacgpt_dir, "real_iupac_tokenizer.pt"),
        ]
        for path in serial_candidates:
            if os.path.exists(path):
                return torch.load(path, map_location="cpu", weights_only=False)
    
    raise FileNotFoundError(
        "Could not find IUPAC SentencePiece tokenizer. Provide one of:\n"
        "  - vocab_path: path to iupac_spm.model\n"
        "  - serialized_path: path to real_iupac_tokenizer.pt\n"
        "  - iupacgpt_dir: path to iupac-gpt/ directory"
    )
