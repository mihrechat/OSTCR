

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



class DebertaEncoder:
    def __init__(self, model_name="microsoft/deberta-v3-large", device="cuda"):
        print(f"Loading {model_name} onto {device}...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts, batch_size=8):
        """
        Takes a string or list of strings and returns a Numpy array of embeddings.
        DeBERTa-v3-large outputs vectors of dimension: 1024
        """
        if isinstance(texts, str):
            texts = [texts]
            
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            
            # Tokenize and move to GPU
            encoded_input = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors='pt'
            ).to(self.device)

            # Forward pass
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # Perform Mean Pooling (ignoring padding tokens)
            token_embeddings = model_output.last_hidden_state # (Batch, Seq_Len, 1024)
            attention_mask = encoded_input['attention_mask']  # (Batch, Seq_Len)
            
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            sentence_embeddings = sum_embeddings / sum_mask
            
            # Normalize embeddings (crucial for stable dot-products later!)
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            
            
            all_embeddings.append(sentence_embeddings.cpu().numpy())
            
        return np.vstack(all_embeddings)
      

def extract_and_save_question_data2():
    # ── 1. INITIALIZE ENCODER (Runs easily on CPU) ────────────────
    print("Loading Text Encoder...")
    # text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    model_name = "/root/autodl-tmp/models--microsoft--deberta-v3-large/snapshots/64a8c8eab3e352a784c658aef62be1662607476f"
    text_encoder = DebertaEncoder(model_name=model_name, device="cuda")
    
    data_dir = "/root/autodl-tmp/CausalSTGNet/datasets/multi-choice_vqa/STDTraffic/test_dataset"
    video_names = [str(f) for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    seen_keys = set()
    
    with open(VOCAB_PATH, "r") as f:
       vocab = json.load(f)
       qtype_to_idx = {k: v["idx"] for k, v in vocab["qtype"].items()}
       role_to_idx = {k: v["idx"] for k, v in vocab["role_token"].items()}

    with open(RAW_JSONL_PATH, 'r', encoding='utf-8') as f:
        header = json.loads(f.readline())
        vid_id_idx = header.index("vid_filename")        
        q_body_idx = header.index("q_body")
        q_type_idx = header.index("q_type")
        option_indices = [header.index(f"option{i}") for i in range(4)]
        answer_idx = header.index("answer")

        count = 0
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            
            vid_id = str(row[vid_id_idx]) 
            if vid_id.endswith(".mp4"):
                v_id = vid_id[:-4]
                
            if v_id in video_names: 
                if v_id in seen_keys:
                    continue
                else:                    
                    seen_keys.add(v_id)
                    
                video_dir = os.path.join(data_dir, v_id)
                
                q_body = str(row[q_body_idx]).lower().strip()
                q_type = str(row[q_type_idx]).lower().strip()
                answer = int(row[answer_idx]) if "answer" in header else 0 # Best to store answer as integer!
                
                # Make sure we have exactly 4 options
                options = [str(row[i]).lower().strip() for i in option_indices if i < len(row)]
                while len(options) < 4:
                    options.append("") # Pad empty options if a dataset has fewer

                # ── A. ENCODE TEXT TO VECTORS ─────────────────────────
                # q_emb: (1024,)
                # opt_embs: (4, 1024)
                q_emb = text_encoder.encode(q_body)
                opt_embs = text_encoder.encode(options)
                
                # Save embeddings as fast-loading Numpy arrays
                np.save(os.path.join(video_dir, "question_emb.npy"), q_emb.astype(np.float16))
                np.save(os.path.join(video_dir, "options_emb.npy"), opt_embs.astype(np.float16))

                # ── B. PROCESS LINGUISTICS ────────────────────────────
                qtype_idx_val = qtype_to_idx.get(q_type, PAD_IDX)

                doc = nlp(q_body)
                token_idxs = []
                for token in doc:
                    dep = token.dep_
                    if dep not in VALID_DEPS: continue
                    if token.lemma_ in {"be", "do", "have"} or token.lemma_ in VQA_BOILERPLATE: continue

                    role_token = f"{dep} {token.lemma_}"
                    idx = role_to_idx.get(role_token, PAD_IDX)
                    if idx != PAD_IDX:
                        token_idxs.append(idx)

                if len(token_idxs) > MAX_TOKENS:
                    token_idxs = token_idxs[:MAX_TOKENS]
                    
                triplet_mask = [True] * len(token_idxs)
                while len(token_idxs) < MAX_TOKENS:
                    token_idxs.append(PAD_IDX)
                    triplet_mask.append(False)

                # ── C. SAVE METADATA JSON ─────────────────────────────
                question_data = {
                    "qtype_idx": qtype_idx_val,
                    "qtype_name": q_type,
                    "triplet_idxs": token_idxs,
                    "triplet_mask": triplet_mask,
                    "answer": answer  # Keep it simple, just the integer index
                }
                
                with open(os.path.join(video_dir, "question.json"), "w") as qf:
                    json.dump(question_data, qf)
                    
                count += 1

        print(f"Successfully generated offline data for {count} videos!")

if __name__ == "__main__":
 
  extract_and_save_question_data2()
 
