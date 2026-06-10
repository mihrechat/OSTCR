

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import spacy
import json
import os
import numpy as np

import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# ── 1. CONFIGURATION ─────────────────────────────────────────────
RAW_JSONL_PATH        = '/root/autodl-tmp/CausalSTGNet/datasets/multi-choice_vqa/STDTraffic/R2_all.jsonl'
VOCAB_PATH      = '/root/autodl-tmp/CausalSTGNet/datasets/multi-choice_vqa/STDTraffic/prior_memory/language/linguistic_stats.json'      

# ── 2. INITIALIZATION ────────────────────────────────────────────

nlp = spacy.load("en_core_web_sm")


VALID_DEPS = {"nsubj", "nsubjpass", "dobj", "pobj", "iobj", "amod", "advmod", "nummod", "prep", "attr", "ROOT", "advcl", "xcomp", "ccomp"}
VQA_BOILERPLATE = {"video", "clip", "image", "picture", "show", "happen"}
PAD_IDX = 0
MAX_TOKENS = 10

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np



class DebertaEncoder:
    """ 
    Simple single-question encoder. No batching needed.
    Returns full token sequence including CLS.
    """
    
    def __init__(self, model_name="microsoft/deberta-v3-large", device="cuda"):
        print(f"Loading {model_name} onto {device}...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    
    def encode(self, questions, max_length=16, add_special_tokens=True, pooling=None):
        """
        Encode question(s) - handles both single string and list of strings.
        
        Args:
            questions: String or List[str] (e.g., "What is...?" or ["Q1?", "Q2?"])
            max_length: Maximum token length
            add_special_tokens: Whether to add [CLS]/[SEP]
            
        Returns:
            tuple: (token_embeddings, attention_mask)
            - Single question: (L, 1024) and (L,)
            - Batch questions: (B, L, 1024) and (B, L)
        """
        is_single = isinstance(questions, str)
        if is_single:
            questions = [questions]
        
        encoded_input = self.tokenizer(
            questions, 
            padding='max_length',
            truncation=True, 
            max_length=max_length, 
            return_tensors='pt',
            add_special_tokens=add_special_tokens
        ).to(self.device)
        
        model_output = self.model(**encoded_input)
        
        token_embeddings = model_output.last_hidden_state.cpu().numpy()  # (B, L, 1024)
        attention_mask = encoded_input['attention_mask'].cpu().numpy()   # (B, L)
        
    
        if pooling == "cls":
            token_embeddings = token_embeddings[:, 0, :]  # (B, 1024)
            if is_single:
                return token_embeddings[0]  # (1024,)
            return token_embeddings  # (B, 1024)
        
        # Return with or without batch dimension
        if is_single:
            return token_embeddings[0], attention_mask[0]  # (L, 1024), (L,)
        return token_embeddings, attention_mask  # (B, L, 1024), (B, L)
 