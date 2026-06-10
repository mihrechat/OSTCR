
import spacy
import torch
import json
from collections import defaultdict
from question_extract import DebertaEncoder
import os
from tqdm import tqdm
import csv

# Load NLP and Encoder
nlp = spacy.load("en_core_web_sm")
model_name = "/root/autodl-tmp/models--microsoft--deberta-v3-large/snapshots/64a8c8eab3e352a784c658aef62be1662607476f"
encoder = DebertaEncoder(model_name=model_name)

VALID_DEPS = {
    "nsubj", "nsubjpass", "dobj", "pobj", "iobj", 
    "amod", "advmod", "nummod", 
    "prep", "attr", "ROOT", "advcl", "xcomp", "ccomp"
}
VQA_BOILERPLATE = {"video", "clip", "image", "picture", "show", "happen"}

RAW_JSONL_PATH  = '/root/autodl-tmp/CausalSTGNet/datasets/multi-choice_vqa/STDTraffic/R2_all.jsonl'
trans_VOCAB_PATH      = '/root/autodl-tmp/CausalSTGNet/datasets/multi-choice_vqa/TGIF/trans2/prior_memory/language/linguistic_vocab.json' 
trans_train_data = '/root/autodl-tmp/CausalSTGNet/datasets/multi-choice_vqa/TGIF/trans2/train'
trans_test_data = '/root/autodl-tmp/CausalSTGNet/datasets/multi-choice_vqa/TGIF/trans/test'
train_trans_tgif_csv_path = '/root/autodl-tmp/CausalSTGNet/datasets/multi-choice_vqa/TGIF/trans/prior_memory/language/Train_transition_question.csv'     
test_trans_tgif_csv_path = '/root/autodl-tmp/CausalSTGNet/datasets/multi-choice_vqa/TGIF/trans/prior_memory/language/Test_transition_question.csv'

# ── 2. INITIALIZATION ────────────────────────────────────────────

nlp = spacy.load("en_core_web_sm")


VALID_DEPS = {"nsubj", "nsubjpass", "dobj", "pobj", "iobj", "amod", "advmod", "nummod", "prep", "attr", "ROOT", "advcl", "xcomp", "ccomp"}
VQA_BOILERPLATE = {"video", "clip", "image", "picture", "show", "happen"}
PAD_IDX = 0
MAX_TOKENS = 8


# ["record_id", "vid_id", "vid_filename", "perspective", "q_body", "q_type", "option0", "option1", "option2", "option3", "answer"]
def build_linguistic_priors_offline(jsonl_filepath, ln_output_dir):
    """
    Reads the specific Video-QA JSON list-of-lists format and builds E_z priors.
    """
    print(f"Loading data from {jsonl_filepath}...")
    
    # 1. Initialize Encoder
    print("Loading Text Encoder...")
    model_name = "/root/autodl-tmp/models--microsoft--deberta-v3-large/snapshots/64a8c8eab3e352a784c658aef62be1662607476f"
    text_encoder = DebertaEncoder(model_name=model_name, device="cuda")
    
    with open(jsonl_filepath, 'r', encoding='utf-8') as f:
        # 2. Read the first line to get the headers dynamically
        header_line = f.readline()
        headers = json.loads(header_line)
        
        q_body_idx = headers.index("q_body")
        q_type_idx = headers.index("q_type")
        q_video_idx = headers.index('vid_filename')
        
        # 3. Get training videos
        training_rows = []
        training_dataset_dir = "/root/autodl-tmp/CausalSTGNet/datasets/multi-choice_vqa/STDTraffic/train_dataset"
        training_dataset_video_ids = set(os.listdir(training_dataset_dir))
        
        # 4. Read the rest of the lines
        for line in f:
            if not line.strip(): 
                continue
            row = json.loads(line)
            
            # Safely check if the video is in our training directory (with or without .mp4)
            vid_filename = str(row[q_video_idx])
            v_id = vid_filename[:-4] if vid_filename.endswith(".mp4") else vid_filename
            
            if v_id in training_dataset_video_ids:
                training_rows.append(row)

    print(f"1. Counting Question Types and Role-Tokens for {len(training_rows)} questions...")
    qtype_counts = defaultdict(int)
    role_token_counts = defaultdict(int)
    total_q = len(training_rows)
    total_role_tokens = 0
    
    for row in tqdm(training_rows, desc="== Processing training rows =="):
        q_text = str(row[q_body_idx]).lower().strip()
        q_type = str(row[q_type_idx]).lower().strip()
        
        qtype_counts[q_type] += 1
        
        doc = nlp(q_text)
        for token in doc:
            dep = token.dep_
            if dep not in VALID_DEPS: continue
            if token.lemma_ in {"be", "do", "have"}: continue
            if token.lemma_ in VQA_BOILERPLATE: continue

            role_token = f"{dep} {token.lemma_}"
            role_token_counts[role_token] += 1
            total_role_tokens += 1
    dim = 1024
    os.makedirs(ln_output_dir, exist_ok=True)
    
    # --- Question Type Prior (Macro-level) ---
    qtype_vocab = list(qtype_counts.keys())
    qtype_to_idx = {qt: i for i, qt in enumerate(qtype_vocab)}
    qtype_expected_embs = torch.zeros((len(qtype_vocab), dim))
    qtype_info = {}
    
    for qt, count in qtype_counts.items():
        print(f"qtype: {qt} count: {count}")
        p_t = count / total_q
        qtype_info[qt] = {"idx": qtype_to_idx[qt], "count": count, "prob": p_t}
        
        # Multiply Probability by DeBERTa embedding!
        emb = torch.tensor(text_encoder.encode(qt))
        qtype_expected_embs[qtype_to_idx[qt]] =  emb

    # --- Structural Role-Token Prior (Micro-level) ---
    role_token_vocab = ["<PAD>"] + list(role_token_counts.keys())
    role_token_to_idx = {tr: i for i, tr in enumerate(role_token_vocab)}
    role_token_expected_embs = torch.zeros((len(role_token_vocab), dim))
    role_token_info = {}
    
    for tr, count in role_token_counts.items():
        if tr == "<PAD>": continue
        p_tau = count / total_role_tokens
        role_token_info[tr] = {"idx": role_token_to_idx[tr], "count": count, "prob": p_tau}
    
        # Multiply Probability by DeBERTa embedding!
        emb = torch.tensor(text_encoder.encode(tr))
        role_token_expected_embs[role_token_to_idx[tr]] = emb 

    # Save to disk
    torch.save(qtype_expected_embs, os.path.join(ln_output_dir, "E_qtype_priors.pt"))
    torch.save(role_token_expected_embs, os.path.join(ln_output_dir, "E_role_token_priors.pt"))
    
    with open(os.path.join(ln_output_dir, "linguistic_vocab.json"), "w") as f:
        json.dump({"qtype": qtype_info, "role_token": role_token_info}, f)
    
    print("Done! Saved Hierarchical Linguistic Priors successfully.") 


def build_linguistic_priors_offline_csv_tgif(csv_filepath, ln_output_dir, data_dir):
    """
    Reads the specific Video-QA JSON list-of-lists format and builds E_z priors.
    """
    print(f"Loading data from {csv_filepath}...")
    
    # 1. Initialize Encoder
    print("Loading Text Encoder...")
    model_name = "/root/autodl-tmp/models--microsoft--deberta-v3-large/snapshots/64a8c8eab3e352a784c658aef62be1662607476f"
    text_encoder = DebertaEncoder(model_name=model_name, device="cuda")
    
    with open(csv_filepath, 'r', encoding='utf-8') as f:
        # 2. Read the first line to get the headers dynamically
        reader = csv.DictReader(f, delimiter='\t')
        
        if "question" not in reader.fieldnames or "gif_name" not in reader.fieldnames:
            raise ValueError("Required headers 'question' and 'gif_name' not found in CSV.")
      
        # 3. Get training videos
        training_rows = []
        training_dataset_dir = data_dir
        training_dataset_video_ids = set(os.listdir(training_dataset_dir))
        
        # 4. Read the rest of the lines
        for row in reader:
            if not row:
                continue
            
            vid_filename = row["gif_name"].strip()
            
            if training_dataset_video_ids is not None:
                v_id = os.path.splitext(vid_filename)[0]
                
                if v_id not in training_dataset_video_ids:
                    continue
                
            row["q_type"] = 1
            training_rows.append(row)

    print(f"1. Counting Question Types and Role-Tokens for {len(training_rows)} questions...")
    qtype_counts = defaultdict(int)
    role_token_counts = defaultdict(int)
    total_q = len(training_rows)
    total_role_tokens = 0
    
    for row in tqdm(training_rows, desc="== Processing training rows =="):
        q_text = str(row['question']).lower().strip()
        q_type = str(row["q_type"]).lower().strip()
        
        qtype_counts[q_type] += 1
        
        doc = nlp(q_text)
        for token in doc:
            dep = token.dep_
            if dep not in VALID_DEPS: continue
            if token.lemma_ in {"be", "do", "have"}: continue
            if token.lemma_ in VQA_BOILERPLATE: continue

            role_token = f"{token.lemma_}"
            role_token_counts[role_token] += 1
            total_role_tokens += 1
    
    print("2. Computing Expectations (P * Embedding)...")
    dim = 1024
    os.makedirs(ln_output_dir, exist_ok=True)
    
    # # --- Question Type Prior (Macro-level) ---
    qtype_vocab = list(qtype_counts.keys())
    qtype_to_idx = {qt: i for i, qt in enumerate(qtype_vocab)}
    qtype_expected_embs = torch.zeros((len(qtype_vocab), dim))
    qtype_info = {}
    
    for qt, count in qtype_counts.items():
        print(f"qtype: {qt} count: {count}")
        p_t = count / total_q
        qtype_info[qt] = {"idx": qtype_to_idx[qt], "count": count, "prob": p_t}
        
        emb, mask = text_encoder.encode(qt, add_special_tokens=False, max_length=4)
        print(f'------------question type shape extracted before mask -------: {emb.shape}')
        emb = torch.tensor(emb[mask.astype(bool)], dtype=torch.float32)
        print(f'------------question  type shape extracted after mask -------: {emb.shape}')
        qtype_expected_embs[qtype_to_idx[qt]] =  emb
        

    # --- Structural Role-Token Prior (Micro-level) ---
    role_token_vocab = ["<PAD>"] + list(role_token_counts.keys())
    role_token_to_idx = {tr: i for i, tr in enumerate(role_token_vocab)}
    role_token_expected_embs = torch.zeros((len(role_token_vocab), dim))
    role_token_info = {}
    
    for tr, count in role_token_counts.items():
        if tr == "<PAD>": continue
        p_tau = count / total_role_tokens
        role_token_info[tr] = {"idx": role_token_to_idx[tr], "count": count, "prob": p_tau}
    
        # Multiply Probability by DeBERTa embedding!
        # emb = torch.tensor(text_encoder.encode(tr))
        emb, mask = text_encoder.encode(tr, add_special_tokens=False, max_length=4)  # 
       
        emb = torch.tensor(emb[mask.astype(bool)])
        if emb.shape[0] > 1:
            emb = emb.mean(dim=0)
        else:
            emb = emb[0]
        
        role_token_expected_embs[role_token_to_idx[tr]] = emb 

    # Save to disk
    torch.save(qtype_expected_embs, os.path.join(ln_output_dir, "E_qtype_priors.pt"))
    torch.save(role_token_expected_embs, os.path.join(ln_output_dir, "E_role_token_priors.pt"))
    
    with open(os.path.join(ln_output_dir, "linguistic_vocab.json"), "w") as f:
        json.dump({"qtype": qtype_info, "role_token": role_token_info}, f)
    
    print("Done! Saved Hierarchical Linguistic Priors successfully.") 

def extract_and_save_question_data_csv_tgif():
    
    print("Loading Text Encoder...")
    model_name = "/root/autodl-tmp/models--microsoft--deberta-v3-large/snapshots/64a8c8eab3e352a784c658aef62be1662607476f"
    text_encoder = DebertaEncoder(model_name=model_name, device="cuda")
    
    data_dir = '/root/autodl-tmp/CausalSTGNet/datasets/multi-choice_vqa/TGIF/trans2/train'
    
    
    # Get actual folder names
    video_names = set([str(f) for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    
    video_q_counts = defaultdict(int)
    
    # Load Vocabulary
    with open(trans_VOCAB_PATH, "r") as f:
       vocab = json.load(f)
       qtype_to_idx = {k: v["idx"] for k, v in vocab["qtype"].items()}
       role_to_idx = {k: v["idx"] for k, v in vocab["role_token"].items()}

    # --- CSV READING LOGIC ---
    with open(train_trans_tgif_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        
        # Verify headers
        if "gif_name" not in reader.fieldnames or "question" not in reader.fieldnames:
            raise ValueError("CSV must contain 'gif_name' and 'question' columns.")

        count = 0
        missing = 0
        for row in tqdm(reader, desc="Processing Questions"):
            if not row: continue
            
            vid_filename = str(row["gif_name"]).strip() 
            # Use os.path.splitext to safely remove extensions like .gif, .mp4, etc.
            v_id = os.path.splitext(vid_filename)[0]
                
            if v_id in video_names: 
                video_q_counts[v_id] += 1
                q_idx = video_q_counts[v_id]  
                
                video_dir = os.path.join(data_dir, v_id)
                
                # Map CSV columns to variables
                q_body = str(row["question"]).lower().strip()
                q_type = "1"  # Hardcoded as requested
                answer = int(row["answer"]) if "answer" in row else 0
                
                q_path = os.path.join(video_dir, f"q_{q_idx}_emb.npy")
                q_options_path  = os.path.join(video_dir, f"q_{q_idx}_options_emb.npy")
                q_path_npz = os.path.join(video_dir, f"q_{q_idx}_emb.npz")
                q_options_npz_path = os.path.join(video_dir, f"q_{q_idx}_options_emb.npz")
                # Extract options from a1, a2, a3, a4, a5
                options = []
                for i in range(1, 6):
                    opt_key = f"a{i}"
                    if opt_key in row and row[opt_key].strip():
                        options.append(str(row[opt_key]).lower().strip())
                    else:
                        options.append("<PAD>")
               
                # Encode texts
                if os.path.exists(q_path):
                    os.remove(q_path)
                    os.remove(q_options_path)
                # q_emb = text_encoder.encode(q_body)
                if os.path.exists(q_path_npz) and os.path.exists(q_options_npz_path):
                    continue
                q_emb, q_mask  = text_encoder.encode(q_body)
                missing +=1
                np.savez_compressed(
                    os.path.join(video_dir, f"q_{q_idx}_emb.npz"),
                    emb=q_emb.astype(np.float16),
                    mask=q_mask.astype(bool)
                    )
                opt_embs, opt_mask  = text_encoder.encode(options, max_length=8)
                np.savez_compressed(
                    os.path.join(video_dir, f"q_{q_idx}_options_emb.npz"),
                    emb=opt_embs.astype(np.float16),
                    mask=opt_mask.astype(bool)
                    )
                # np.save(os.path.join(video_dir, f"q_{q_idx}_emb.npy"), q_emb.astype(np.float16))
                # opt_embs = text_encoder.encode(options)
                # np.save(os.path.join(video_dir, f"q_{q_idx}_options_emb.npy"), opt_embs.astype(np.float16))

                qtype_idx_val = qtype_to_idx.get(q_type, PAD_IDX)

                # Extract role tokens using spaCy
                doc = nlp(q_body)
                token_idxs = []
                for token in doc:
                    dep = token.dep_
                    if dep not in VALID_DEPS: continue
                    if token.lemma_ in {"be", "do", "have"} or token.lemma_ in VQA_BOILERPLATE: continue

                    role_token = f"{token.lemma_}"
                    idx = role_to_idx.get(role_token, PAD_IDX)
                    if idx != PAD_IDX:
                        token_idxs.append(idx)

                # Pad or truncate tokens
                if len(token_idxs) > MAX_TOKENS:
                    token_idxs = token_idxs[:MAX_TOKENS]
                    
                triplet_mask = [True] * len(token_idxs)
                while len(token_idxs) < MAX_TOKENS:
                    token_idxs.append(PAD_IDX)
                    triplet_mask.append(False)

                # ── C. SAVE METADATA JSON ─────────────────────────────
                question_data = {
                    "question_id": f"q_{q_idx}",
                    "q_body": q_body,           
                    "qtype_idx": qtype_idx_val,
                    "qtype_name": q_type,
                    "triplet_idxs": token_idxs,
                    "triplet_mask": triplet_mask,
                    "answer": answer 
                }
                
                with open(os.path.join(video_dir, f"q_{q_idx}.json"), "w") as qf:
                    json.dump(question_data, qf)
                    
                count += 1

        print(f"Successfully generated missing: {missing} total: {count} total questions across {len(video_q_counts)} videos!")
import os
import json
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm

def extract_and_save_question_data():
    
    print("Loading Text Encoder...")
    model_name = "/root/autodl-tmp/models--microsoft--deberta-v3-large/snapshots/64a8c8eab3e352a784c658aef62be1662607476f"
    text_encoder = DebertaEncoder(model_name=model_name, device="cuda")
    
    data_dir = '/root/autodl-tmp/CausalSTGNet/datasets/multi-choice_vqa/TGIF/trans2/train'
    
    video_names = set([str(f) for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    
    video_q_counts = defaultdict(int)
    
    # Load Vocabulary
    with open(trans_VOCAB_PATH, "r") as f:
       vocab = json.load(f)
       qtype_to_idx = {k: v["idx"] for k, v in vocab["qtype"].items()}
       role_to_idx = {k: v["idx"] for k, v in vocab["role_token"].items()}

    with open(RAW_JSONL_PATH, 'r', encoding='utf-8') as f:
        header = json.loads(f.readline())
        vid_id_idx = header.index("vid_filename")        
        q_body_idx = header.index("q_body")
        q_type_idx = header.index("q_type")
        answer_idx = header.index("answer") if "answer" in header else -1
        
        option_indices = [i for i, col in enumerate(header) if str(col).startswith("option")]

        count = 0
        for line in tqdm(f, desc="Processing Questions"):
            if not line.strip(): continue
            row = json.loads(line)
            
            vid_filename = str(row[vid_id_idx]) 
            v_id = vid_filename[:-4] if vid_filename.endswith(".mp4") else vid_filename
                
            if v_id in video_names: 
                video_q_counts[v_id] += 1
                q_idx = video_q_counts[v_id]  
                
                video_dir = os.path.join(data_dir, v_id)
                q_path = os.path.join(video_dir, f"q_{q_idx}_emb.npy")
                q_options  = os.path.join(video_dir, f"q_{q_idx}_options_emb.npy")
                
                q_body = str(row[q_body_idx]).lower().strip()
                q_type = str(row[q_type_idx]).lower().strip()
                answer = int(row[answer_idx]) if answer_idx != -1 else 0
                
                # Extract all options found in the header
                options = [str(row[i]).lower().strip() for i in option_indices if i < len(row)]
                
                if not os.path.exists(q_path):
                    q_emb = text_encoder.encode(q_body)
                    np.save(os.path.join(video_dir, f"q_{q_idx}_emb.npy"), q_emb.astype(np.float16))
                    
                if not os.path.exists(q_options):
                    opt_embs = text_encoder.encode(options)
                    np.save(os.path.join(video_dir, f"q_{q_idx}_options_emb.npy"), opt_embs.astype(np.float16))

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
                    "question_id": f"q_{q_idx}",
                    "q_body": q_body,           
                    "qtype_idx": qtype_idx_val,
                    "qtype_name": q_type,
                    "triplet_idxs": token_idxs,
                    "triplet_mask": triplet_mask,
                    "answer": answer 
                }
                
               
                with open(os.path.join(video_dir, f"q_{q_idx}.json"), "w") as qf:
                    json.dump(question_data, qf)
                    
                count += 1

        print(f"Successfully generated {count} total questions across {len(video_q_counts)} videos!")
        
if __name__ == "__main__":
    json_filepath = '/root/autodl-tmp/CausalSTGNet/datasets/multi-choice_vqa/STDTraffic/R2_all.jsonl'
    ln_output_dir = '/root/autodl-tmp/CausalSTGNet/datasets/multi-choice_vqa/TGIF/trans2/prior_memory/language'
    # build_linguistic_priors_offline(json_filepath, ln_out_dir)
    # build_linguistic_priors_offline_csv_tgif(train_trans_tgif_csv_path, ln_output_dir, trans_train_data)
    extract_and_save_question_data_csv_tgif()
    