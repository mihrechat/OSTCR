import os
import csv
import json
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from question_extract import DebertaEncoder
import spacy
import random
import pandas as pd
nlp = spacy.load("en_core_web_sm")
model_name = "/root/autodl-tmp/models--microsoft--deberta-v3-large/snapshots/64a8c8eab3e352a784c658aef62be1662607476f"
encoder = DebertaEncoder(model_name=model_name)
train_question = '/root/autodl-tmp/MSVD-QA/train_qa.json'
valid_qustion  = '/root/autodl-tmp/MSVD-QA/val_qa.json'
test_question = '/root/autodl-tmp/MSVD-QA/test_qa.json'
ln_output_dir  = "/root/autodl-tmp/CausalSTGNet/datasets/open_ended_vqa/MSVD/prior_memory/language"
train_data     = '/root/autodl-tmp/CausalSTGNet/datasets/open_ended_vqa/MSVD/train'
test_data      = '/root/autodl-tmp/CausalSTGNet/datasets/open_ended_vqa/MSVD/test'
val_data       = '/root/autodl-tmp/CausalSTGNet/datasets/open_ended_vqa/MSVD/val'
vocab_path     = '/root/autodl-tmp/CausalSTGNet/datasets/open_ended_vqa/MSVD_vqa/prior_memory/language/vocab.json'
json_filterd_question = '/root/autodl-tmp/CausalSTGNet/datasets/open_ended_vqa/MSVD_vqa/prior_memory/language/filtered_questions.json'
mapping_dir       = '/root/autodl-tmp/CausalSTGNet/datasets/open_ended_vqa/MSVD/prior_memory/youtube_mapping.txt'
data_dir         =  '/root/autodl-tmp/CausalSTGNet/datasets/open_ended_vqa/MSVD/YouTubeClips'
# Define the Question Types
MSVD_QTYPES = {  #change to msrvt question types too to extract question from msvrrt
    'what': 0,
    'who': 1,
    'how': 2,
    'when': 3,
    'where': 4,
    'other': 5
}

VALID_DEPS = {
    "nsubj", "nsubjpass", "dobj", "pobj", "iobj", 
    "amod", "advmod", "nummod", 
    "prep", "attr", "ROOT", "advcl", "xcomp", "ccomp"
}
VQA_BOILERPLATE = {"video", "clip", "image", "picture", "show", "happen"}
VALID_DEPS = {"nsubj", "nsubjpass", "dobj", "pobj", "iobj", "amod", "advmod", "nummod", "prep", "attr", "ROOT", "advcl", "xcomp", "ccomp"}
VQA_BOILERPLATE = {"video", "clip", "image", "picture", "show", "happen"}
PAD_IDX = 0
MAX_TOKENS = 8

def determine_qtype(question_text):
    """Helper to fix the broken 0s in the dataset"""
    q = question_text.lower().strip()
    for qt in ['what', 'who', 'how', 'when', 'where']:
        if q.startswith(qt):
            return qt, MSVD_QTYPES[qt]
    return 'other', MSVD_QTYPES['other']


def build_msvd_vocabs_and_priors(train_json, valid_json, test_json, ln_output_dir):
    print(f"Loading data from {train_json}, {valid_json}, and {test_json}...")
    
    # 1. Get available videos strictly in their respective folders
    training_dataset_video_id    = {os.path.splitext(video)[0] for video in os.listdir(train_data)}
    validation_dataset_video_id  = {os.path.splitext(video)[0] for video in os.listdir(val_data)}
    test_dataset_video_id        = {os.path.splitext(video)[0] for video in os.listdir(test_data)}
    # all_dataset                  = {os.path.splitext(video)[0] for video in os.listdir(data_dir)}

    print(f"Videos -> Train: {len(training_dataset_video_id)}, Val: {len(validation_dataset_video_id)}, Test: {len(test_dataset_video_id)}")

    # 2. Load the mapping from the .txt file (short name -> long ID)
    id_map = {}
    with open(mapping_dir, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                long_id = parts[0]       
                short_name = parts[1]    
                try:
                    numeric_id = int(short_name.replace('vid', ''))
                    id_map[numeric_id] = long_id
                except ValueError:
                    continue

    # 3. Load the JSON question files
    with open(train_json, 'r', encoding='utf-8') as f:
        train_questions = json.load(f)
    with open(valid_json, 'r', encoding='utf-8') as f:
        valid_questions = json.load(f)
    with open(test_json, 'r', encoding='utf-8') as f:
        test_questions = json.load(f)

    # ── 4. Process Questions (FULL DATASET - NO CAPPING) ──
    random.seed(42)
    random.shuffle(train_questions) 
    random.shuffle(valid_questions) 

    filtered_train_rows = []
    filtered_val_rows = []
    filtered_test_rows = []
    
    # Process Train (Strictly inside train_data)
    for q in train_questions:
        video_name = id_map.get(q.get('video_id'))
        if video_name in training_dataset_video_id:
        # if video_name in all_dataset:
            q['video_name'] = video_name 
            filtered_train_rows.append(q)

    # Process Val (Strictly inside val_data)
    for q in valid_questions:
        video_name = id_map.get(q.get('video_id'))
        if video_name in validation_dataset_video_id:  
        # if video_name in all_dataset:
            q['video_name'] = video_name
            filtered_val_rows.append(q)

    # Process Test (Strictly inside test_data)
    for q in test_questions:
        video_name = id_map.get(q.get('video_id'))
        if video_name in test_dataset_video_id:     
        # if video_name in all_dataset:   
            q['video_name'] = video_name
            filtered_test_rows.append(q)

    print(f"Full Train: {len(filtered_train_rows)} questions.")
    print(f"Full Val: {len(filtered_val_rows)} questions.")
    print(f"Full Test: {len(filtered_test_rows)} questions.")

    all_filtered_questions = filtered_train_rows + filtered_val_rows + filtered_test_rows

    # --- Start Linguistic Processing
    qtype_counts = defaultdict(int)
    role_token_counts = defaultdict(int)
    answer_counts = defaultdict(int)
    
    total_q = len(filtered_train_rows)
    total_role_tokens = 0
    
    for row in tqdm(filtered_train_rows, desc="== Processing training questions for Priors =="):
        q_text = str(row['question']).lower().strip()
        ans_text = str(row['answer']).lower().strip()
        
        q_type_str, _ = determine_qtype(q_text)
        qtype_counts[q_type_str] += 1
        answer_counts[ans_text] += 1
        
        doc = nlp(q_text)
        for token in doc:
            dep = token.dep_
            if dep not in VALID_DEPS: continue
            if token.lemma_ in {"be", "do", "have", "a", "an", "the"}: continue

            # role_token = f"{dep} {token.lemma_}"
            role_token = f"{token.lemma_}"
            role_token_counts[role_token] += 1
            total_role_tokens += 1
    
    MIN_FREQ = 2 # keep question with maximum two question at least
    
    # --- Build Answer Vocab ---
    answer_vocab = {"<UNK>": 0}
    answer_stats = {"<UNK>": {"idx": 0, "frequency": 0}}
    current_idx = 1
    
    rare_category_answers = set()
    for row in filtered_train_rows:
        q_text = str(row['question']).lower().strip()
        ans_text = str(row['answer']).lower().strip()
        q_type_str, _ = determine_qtype(q_text)
        
        # If it's a minority category, flag its answer to be saved!
        if q_type_str in ["who","where", "when", "how"]:
            rare_category_answers.add(ans_text)
    
    for ans, count in answer_counts.items():
        if count >= MIN_FREQ or ans in rare_category_answers:
            answer_vocab[ans] = current_idx
            answer_stats[ans] = {"idx": current_idx, "frequency": count}
            current_idx += 1
        
    
    final_train_rows = []
    for row in filtered_train_rows:
        ans_text = str(row['answer']).lower().strip()
        if ans_text in answer_vocab:
            final_train_rows.append(row)
    
    filtered_train_rows = final_train_rows
    # --- Structural Role-Token Prior ---
    role_token_vocab = list(role_token_counts.keys())
    role_token_vocab.insert(0, "<PAD>") 
    role_token_to_idx = {tr: i for i, tr in enumerate(role_token_vocab)}
    role_token_expected_embs = torch.zeros((len(role_token_vocab), 1024))
    role_token_info = {} 
    
    
    
    # --- Question Type Prior ---
    qtype_expected_embs = torch.zeros((len(MSVD_QTYPES), 1024))
    qtype_info = {} 
    for qt_str, count in qtype_counts.items():
        if qt_str not in MSVD_QTYPES: continue
        idx = MSVD_QTYPES[qt_str]
        p_t = count / total_q
        qtype_info[qt_str] = {"idx": idx, "count": count, "prob": p_t}
        emb, mask = encoder.encode(qt_str, add_special_tokens=False, max_length=4) 
        emb = torch.tensor(emb[mask.astype(bool)])
        # emb = torch.tensor(encoder.encode(qt_str))
        qtype_expected_embs[idx] =  emb
        
        
    
    for tr, count in role_token_counts.items():
        if tr == "<PAD>": continue
        p_tau = count / total_role_tokens
        role_token_info[tr] = {"idx": role_token_to_idx[tr], "count": count, "prob": p_tau}
        emb, mask = encoder.encode(tr, add_special_tokens=False, max_length=4)  # 
        emb = torch.tensor(emb[mask.astype(bool)])
        if emb.shape[0] > 1:
            emb = emb.mean(dim=0)
        else:
            emb = emb[0]
        # emb = torch.tensor(encoder.encode(tr))
        role_token_expected_embs[role_token_to_idx[tr]] = emb 

    
        
    # Save outputs
    os.makedirs(ln_output_dir, exist_ok=True)
    torch.save(qtype_expected_embs, os.path.join(ln_output_dir, "E_qtype_priors.pt"))
    torch.save(role_token_expected_embs, os.path.join(ln_output_dir, "E_role_token_priors.pt"))
    
    with open(os.path.join(ln_output_dir, "vocab.json"), "w") as f:
        json.dump({
            "qtype": MSVD_QTYPES, 
            "role_token": role_token_to_idx,
            "answer_vocab": answer_vocab 
        }, f)
        
    with open(os.path.join(ln_output_dir, "linguistic_stats.json"), "w") as f:
        json.dump({
            "qtype": qtype_info, 
            "role_token": role_token_info,
            "answer_stats": answer_stats 
        }, f, indent=2)

    with open(os.path.join(ln_output_dir, "filtered_questions.json"), "w") as f:
        json.dump(all_filtered_questions, f)
    with open(os.path.join(ln_output_dir, "train_split.json"), "w") as f:
        json.dump(filtered_train_rows, f)
    with open(os.path.join(ln_output_dir, "val_split.json"), "w") as f:
        json.dump(filtered_val_rows, f)
    with open(os.path.join(ln_output_dir, "test_split.json"), "w") as f:
        json.dump(filtered_test_rows, f)
    
    print("Done! Saved Priors, Vocab, Stats, and Full Questions successfully.")


def build_msvrtt_vocabs_and_priors(train_csv, test_csv, ln_output_dir):
    print(f"Loading data from {train_csv}, and {test_csv}...")
    
    # 1. Get available videos strictly in their respective folders
    training_dataset_video_id    = {os.path.splitext(video)[0] for video in os.listdir(train_data)}
    # test_dataset_video_id        = {os.path.splitext(video)[0] for video in os.listdir(test_data)}

    print(f"Videos -> Train: {len(training_dataset_video_id)}")

    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)

    # Shuffle
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    filtered_train_rows = []
    filtered_test_rows = []
    
    # Train
    for _, row in train_df.iterrows():

        video_name = row['gif_name']

        if video_name in training_dataset_video_id:

            row_dict = row.to_dict()
            row_dict['video_name'] = video_name
            filtered_train_rows.append(row_dict)
    # Test
    # for _, row in test_df.iterrows():
    #     video_name = row['gif_name']

    #     if video_name in test_dataset_video_id:

    #         row_dict = row.to_dict()
    #         row_dict['video_name'] = video_name
    #         filtered_test_rows.append(row_dict)
    
    print(f"Full Train: {len(filtered_train_rows)} questions.")
    # print(f"Full Test: {len(filtered_test_rows)} questions.")

    all_filtered_questions = filtered_train_rows + filtered_test_rows

    # --- Start Linguistic Processing (STRICTLY ON TRAIN SET ONLY) ---
    qtype_counts = defaultdict(int)
    role_token_counts = defaultdict(int)
    answer_counts = defaultdict(int)
    
    total_q = len(filtered_train_rows)
    total_role_tokens = 0
    
    for row in tqdm(filtered_train_rows, desc="== Processing training questions for Priors =="):
        q_text = str(row['question']).lower().strip()
        ans_text = str(row['answer']).lower().strip()
        
        q_type_str, _ = determine_qtype(q_text)
        qtype_counts[q_type_str] += 1
        answer_counts[ans_text] += 1
        
        doc = nlp(q_text)
        for token in doc:
            dep = token.dep_
            if dep not in VALID_DEPS: continue
            if token.lemma_ in {"be", "do", "have", "a", "an", "the"}: continue

            # role_token = f"{dep} {token.lemma_}"
            role_token = f"{token.lemma_}"
            role_token_counts[role_token] += 1
            total_role_tokens += 1
    
    MIN_FREQ = 2
    
    # --- Build Answer Vocab ---
    answer_vocab = {"<UNK>": 0}
    answer_stats = {"<UNK>": {"idx": 0, "frequency": 0}}
    current_idx = 1
    
    rare_category_answers = set()
    for row in filtered_train_rows:
        q_text = str(row['question']).lower().strip()
        ans_text = str(row['answer']).lower().strip()
        q_type_str, _ = determine_qtype(q_text)
        
        # If it's a minority category, flag its answer to be saved!
        if q_type_str in ["who","where", "when", "how"]:
            rare_category_answers.add(ans_text)
    
    for ans, count in answer_counts.items():
        if count >= MIN_FREQ or ans in rare_category_answers:
            answer_vocab[ans] = current_idx
            answer_stats[ans] = {"idx": current_idx, "frequency": count}
            current_idx += 1
        
    print(f"Generated Vocabulary with {len(answer_vocab)} unique answers (Protected rare categories).")
    
    final_train_rows = []
    for row in filtered_train_rows:
        ans_text = str(row['answer']).lower().strip()
        if ans_text in answer_vocab:
            final_train_rows.append(row)
    print(f"Filtered down to {len(final_train_rows)} training questions with answers in the vocab (min freq {MIN_FREQ}) and removed {len(filtered_train_rows) - len(final_train_rows)}.")
    
    filtered_train_rows = final_train_rows
    # --- Structural Role-Token Prior ---
    role_token_vocab = list(role_token_counts.keys())
    role_token_vocab.insert(0, "<PAD>") 
    role_token_to_idx = {tr: i for i, tr in enumerate(role_token_vocab)}
    role_token_expected_embs = torch.zeros((len(role_token_vocab), 1024))
    role_token_info = {} 
    
    # --- Question Type Prior ---
    qtype_expected_embs = torch.zeros((len(MSVD_QTYPES), 1024))
    qtype_info = {} 
    for qt_str, count in qtype_counts.items():
        if qt_str not in MSVD_QTYPES: continue
        idx = MSVD_QTYPES[qt_str]
        p_t = count / total_q
        qtype_info[qt_str] = {"idx": idx, "count": count, "prob": p_t}
        emb, mask = encoder.encode(qt_str, add_special_tokens=False, max_length=4) 
        print(f'------------question type shape extracted -------: {emb.shape}')
        emb = torch.tensor(emb[mask.astype(bool)])
        print(f'------------question type after bool extracted -------: {emb.shape}')
        # emb = torch.tensor(encoder.encode(qt_str))
        qtype_expected_embs[idx] = emb
        
        
    
    for tr, count in role_token_counts.items():
        if tr == "<PAD>": continue
        p_tau = count / total_role_tokens
        role_token_info[tr] = {"idx": role_token_to_idx[tr], "count": count, "prob": p_tau}
        emb, mask = encoder.encode(tr, add_special_tokens=False, max_length=4)  # 
        # print(f'------------question role type shape extracted before mask -------: {emb.shape}')
        emb = torch.tensor(emb[mask.astype(bool)])
        if emb.shape[0] > 1:
            emb = emb.mean(dim=0)
        else:
            emb = emb[0]
        # print(f'------------question role type shape extracted after mask -------: {emb.shape}')
        # emb = torch.tensor(encoder.encode(tr))
        role_token_expected_embs[role_token_to_idx[tr]] = emb 

    
        
    # Save outputs
    os.makedirs(ln_output_dir, exist_ok=True)
    torch.save(qtype_expected_embs, os.path.join(ln_output_dir, "E_qtype_priors.pt"))
    torch.save(role_token_expected_embs, os.path.join(ln_output_dir, "E_role_token_priors.pt"))
    
    with open(os.path.join(ln_output_dir, "vocab.json"), "w") as f:
        json.dump({
            "qtype": MSVD_QTYPES, 
            "role_token": role_token_to_idx,
            "answer_vocab": answer_vocab 
        }, f)
        
    with open(os.path.join(ln_output_dir, "linguistic_stats.json"), "w") as f:
        json.dump({
            "qtype": qtype_info, 
            "role_token": role_token_info,
            "answer_stats": answer_stats 
        }, f, indent=2)

    with open(os.path.join(ln_output_dir, "filtered_questions.json"), "w") as f:
        json.dump(all_filtered_questions, f)
    with open(os.path.join(ln_output_dir, "train_split.json"), "w") as f:
        json.dump(filtered_train_rows, f)
    with open(os.path.join(ln_output_dir, "val_split.json"), "w") as f:
        json.dump(filtered_val_rows, f)
    with open(os.path.join(ln_output_dir, "test_split.json"), "w") as f:
        json.dump(filtered_test_rows, f)
    
    print("Done! Saved Priors, Vocab, Stats, and Full Questions successfully.")
def extract_and_save_msvd_data(json_filepath, data_dirs, vocab_path):
    """
    data_dirs: a list of the folder paths, e.g., [train_data, val_data, test_data]
    """
    print("Loading Text Encoder...")
    model_name = "/root/autodl-tmp/models--microsoft--deberta-v3-large/snapshots/64a8c8eab3e352a784c658aef62be1662607476f"
    text_encoder = DebertaEncoder(model_name=model_name, device="cuda")
    
    # Map video name to its exact folder path dynamically
    video_to_path = {}
    for d_dir in data_dirs:
        for f in os.listdir(d_dir):
            if os.path.isdir(os.path.join(d_dir, f)):
                video_to_path[f] = os.path.join(d_dir, f)
    
    with open(vocab_path, "r") as f:
       vocab = json.load(f)
       role_to_idx = vocab["role_token"]
       answer_vocab = vocab["answer_vocab"] 

    with open(json_filepath, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    count = 0
    for q in tqdm(questions, desc="Extracting QA features"):
        video_name = q.get('video_name') 
        
        if video_name in video_to_path:
            video_dir = video_to_path[video_name] 
            
            q_id = q['id'] 
            q_text = q['question'].lower().strip()
            ans_text = q['answer'].lower().strip()
            
            answer_idx = answer_vocab.get(ans_text, 0) 
            q_type_str, qtype_idx_val = determine_qtype(q_text)

            # ── A. ENCODE QUESTION TEXT ──
            q_emb, q_mask  = text_encoder.encode(q_text)
            
            
            np.savez_compressed(
            os.path.join(video_dir, f"q_{q['id']}_emb.npz"),
            emb=q_emb.astype(np.float16),
            mask=q_mask.astype(bool)
            )
            # np.save(os.path.join(video_dir, f"q_{q_id}_emb.npy"), q_emb.astype(np.float16))

            # ── B. PROCESS LINGUISTICS ──
            doc = nlp(q_text)
            token_idxs = []
            for token in doc:
                dep = token.dep_
                if dep not in VALID_DEPS: continue
                if token.lemma_ in {"be", "do", "have"}: continue

                role_token = f"{token.lemma_}"
                idx = role_to_idx.get(role_token, PAD_IDX)
                if idx != PAD_IDX:
                    token_idxs.append(idx)

            if len(token_idxs) > MAX_TOKENS:
                token_idxs = token_idxs[:MAX_TOKENS]
                
            triplet_mask = [True] * len(token_idxs)
            while len(token_idxs) < MAX_TOKENS:
                token_idxs.append(PAD_IDX)
                triplet_mask.append(False)

            # ── C. SAVE METADATA JSON ──
            question_data = {
                "question_id": q_id,
                "qtype_idx": qtype_idx_val,
                "qtype_name": q_type_str,
                "triplet_idxs": token_idxs,
                "triplet_mask": triplet_mask,
                "answer": answer_idx 
            }
            
            with open(os.path.join(video_dir, f"q_{q_id}.json"), "w") as qf:
                json.dump(question_data, qf)
                
            count += 1

    print(f"Successfully generated offline data for {count} MSVD questions!")
    
def force_extract_missing():
    print("Loading Text Encoder...")
    model_name = "/root/autodl-tmp/models--microsoft--deberta-v3-large/snapshots/64a8c8eab3e352a784c658aef62be1662607476f"
    text_encoder = DebertaEncoder(model_name=model_name, device="cuda")
    
    ln_output_dir = '/root/autodl-tmp/CausalSTGNet/datasets/open_ended_vqa/MSVD_2/prior_memory/language'
    vocab_path = os.path.join(ln_output_dir, "vocab.json")
    
    with open(vocab_path, "r") as f:
       vocab = json.load(f)
       role_to_idx = vocab["role_token"]
       answer_vocab = vocab["answer_vocab"] 

    splits = {
        "train": '/root/autodl-tmp/CausalSTGNet/datasets/open_ended_vqa/MSVD_2/train',
        "val":   '/root/autodl-tmp/CausalSTGNet/datasets/open_ended_vqa/MSVD_2/val',
        "test":  '/root/autodl-tmp/CausalSTGNet/datasets/open_ended_vqa/MSVD_2/test'
    }

    count_emb_generated = 0
    count_json_updated = 0

    for split_name, root_dir in splits.items():
        split_json = os.path.join(ln_output_dir, f"{split_name}_split.json")
        with open(split_json, 'r', encoding='utf-8') as f:
            questions = json.load(f)
            
        for q in tqdm(questions, desc=f"Checking {split_name.upper()} split"):
            video_name = q['video_name']
            q_id = q['id']
            video_dir = os.path.join(root_dir, video_name)
            
            q_json_path = os.path.join(video_dir, f"q_{q_id}.json")
            q_emb_path  = os.path.join(video_dir, f"q_{q_id}_emb.npy")
            q_emb_npz_path  = os.path.join(video_dir, f"q_{q_id}_emb.npz")
            
            # Make sure the video folder actually exists
            if not os.path.exists(video_dir):
                print(f"CRITICAL ERROR: The video folder {video_dir} does not exist!")
                continue
                
            q_text = str(q['question']).lower().strip()
            ans_text = str(q['answer']).lower().strip()
            
            answer_idx = answer_vocab.get(ans_text, 0) 
            q_type_str, qtype_idx_val = determine_qtype(q_text)

            
            if os.path.exists(q_emb_path):
                print(f"file {q_emb_path} exists and removed")
                os.remove(q_emb_path)
            if os.path.exists(q_emb_npz_path):
                print(f"file {q_id} exists in {split_name} skipped----------------")
                continue
                
            q_emb, q_mask  = text_encoder.encode(q_text)
            print(f"------------ shape of question: {q_emb.shape}")
            np.savez_compressed(os.path.join(video_dir, f"q_{q_id}_emb.npz"),
                                emb = q_emb.astype(np.float16),
                                mask = q_mask.astype(bool))
            
            # q_emb = text_encoder.encode(q_text)
            # np.save(q_emb_path, q_emb.astype(np.float16))
            count_emb_generated += 1

            # ── 2. JSON METADATA: ALWAYS overwrite to fix vocab indices! ──
            doc = nlp(q_text)
            token_idxs = []
            for token in doc:
                dep = token.dep_
                if dep not in VALID_DEPS: continue
                if token.lemma_ in {"be", "do", "have"}: continue

                # role_token = f"{dep} {token.lemma_}"
                role_token = f"{token.lemma_}"
                idx = role_to_idx.get(role_token, PAD_IDX)
                if idx != PAD_IDX:
                    token_idxs.append(idx)

            if len(token_idxs) > MAX_TOKENS:
                token_idxs = token_idxs[:MAX_TOKENS]
                
            triplet_mask = [True] * len(token_idxs)
            while len(token_idxs) < MAX_TOKENS:
                token_idxs.append(PAD_IDX)
                triplet_mask.append(False)

            question_data = {
                "question_id": q_id,
                "qtype_idx": qtype_idx_val,
                "qtype_name": q_type_str,
                "triplet_idxs": token_idxs,
                "triplet_mask": triplet_mask,
                "answer": answer_idx   
            }
            
            with open(q_json_path, "w") as qf:
                json.dump(question_data, qf)
                
            count_json_updated += 1

    print(f"\nDone! Generated {count_emb_generated} heavy text embeddings.")
    print(f"Updated {count_json_updated} JSON files with the new vocab indices.")
if __name__ == "__main__":
    
#    build_msvd_vocabs_and_priors(train_question, valid_qustion, test_question, ln_output_dir)
    force_extract_missing()

