
import sys
import os
import json
import pandas as pd
from collections import defaultdict
import spacy
from common import *
import torch
import torch.nn as nn
import torch
import os
from .Language_Encoder.question_extract import DebertaEncoder
from tqdm import tqdm
import random


# Run `python -m spacy download en_core_web_sm` before using
nlp = spacy.load("en_core_web_sm")


CAUSAL_RELS = {"UsedFor", "CapableOf", "Causes", "ReceivesAction"}
PRIOR_TEMPLATES = [
    "A {cls} is typically known for {rel}.",
    "Common for a {cls}: to {rel}.",
    "People expect a {cls} to be used for {rel}."
]

vg_templates = [
    "A {subj} {verb}s a {obj}.",
    "People observe {subj}s and {obj}s when someone {verb}s.",
    "Often, a {obj} is affected when a {subj} {verb}s.",
]



TRIPLET_TEMPLATES = {
    # For ConceptNet-style relations
    "UsedFor": [
        "A {subj} is used for a {target}, has relation with a {obj}.",
        "A {subj} can {rel} {obj}.",
        "One may use a {subj} to {target}, has relation with a {obj}.",
        "To {target}, people often need a {subj}.",
        "Use a {subj}  for {target}."
    ],
    "CapableOf": [
        "A {subj} is capable of {target}, has relation a {obj}.",
        "{subj}, it's possible to {target}.",
        "People use a {subj} to {target}."
    ],
    "AtLocation": [
        "A {subj} can be found at {target}, along with a {obj}.",
        "A typical {obj} is seen where a {subj} is located: {target}.",
        "You'd expect a {subj} and a {obj} together around {target}."
    ],
    "HasProperty":[
        "A {subj} is {target}, has relation with {obj}.",
        "A {subj} is {target}."
        
    ],
    "ReceivesAction":[
        "A {subj} {target}",
        "A {subj} {target} can have relation with {obj}."
        
    ],
    # Fallback for generalized/unseen relations/verbs
    "Default": [
        "A {subj} and {obj} interact through {target}.",
        "a {subj}, a {target}, and a {obj} are involved.",
        "A scene with a {subj} can involve {target}, sometimes with a {obj}."
    ]
}

def is_obj_in_target(obj, target):
    return obj.lower() in target.lower().split()

def extract_action_verbs(phrase: str):
    """Extracts base-form action verbs from VG predicates."""
    doc = nlp(phrase.lower())
    return [token.lemma_ for token in doc if token.pos_ == "VERB" and token.lemma_.isalpha()]

def clean_uri(uri: str) -> str:
    if not isinstance(uri, str) or not uri.startswith("/c/en/"): return ""
    return uri.split("/")[3].replace("_", " ").lower().strip()


# Define Causal Action Relations for ConceptNet Filter

alias_map = load_object_alias(OBJECT_ALIAS_PATH) # Assumed defined elsewhere

def build_hybrid_knowledge_banks(conceptnet_csv, relationship_path, scene_dir, save_dir):
    print("1. Loading ConceptNet...")
    
    cn_df = pd.read_csv(conceptnet_csv, sep="\t", header=None, names=["uri", "rel", "start", "end", "meta"], on_bad_lines="skip")
    
    # Filter for English and our specific causal relations
    cn_df = cn_df[cn_df["start"].str.startswith("/c/en/")]
    
    vg_pair_dict = defaultdict(lambda: defaultdict(set))
    vg_prior_counts = defaultdict(lambda: defaultdict(int))
    cn_dict = defaultdict(lambda: defaultdict(list))
    
    for _, row in tqdm(cn_df.iterrows(), desc="Parsing ConceptNet"):
        rel = row["rel"].replace("/r/", "")
        if rel not in ACTION_RELS:
            continue  # Drop noisy non-causal relations!
            
        subj = clean_uri(row["start"]) 
        if subj in COCO_CLASSES or subj in manual_map or subj in alias_map:
            if subj in COCO_CLASSES:
                subj = subj
            elif subj in manual_map:
                subj = manual_map.get(subj, subj) 
            else:
                subj = alias_map.get(subj, subj) 

            try:
               meta = json.loads(row["meta"])
               weight = float(meta.get("weight", 1.0))
            except (json.JSONDecodeError, ValueError, TypeError):
               weight = 1.0
               
            target = clean_uri(row["end"])
            if target: 
              # Store as a dict so we can sort by weight
              cn_dict[subj][rel].append({"target": target, "weight": weight})

    # Sort ConceptNet targets by weight to keep the strongest causal priors
    for subj in cn_dict:
        for rel in cn_dict[subj]:
            cn_dict[subj][rel] = sorted(cn_dict[subj][rel], key=lambda x: x["weight"], reverse=True)


    print("2. Loading Visual Genome (Fallback & Priors)...")
    with open(relationship_path, "r") as f:
        vg_data = json.load(f) # List of VG images
  
    for img in tqdm(vg_data, desc="Processing VG Relationships"):
        for rel in img.get("relationships", []):
            subj = rel.get("subject", {}).get("name", "").lower()
            obj = rel.get("object", {}).get("name", "").lower()
            predicate = rel.get("predicate", "").lower()
            
            sub = alias_map.get(subj, subj)
            obj = alias_map.get(obj, obj)
            if subj in COCO_CLASSES and obj in COCO_CLASSES:
                verbs = extract_action_verbs(predicate) # Assumed defined elsewhere
                for v in verbs:
                    if v not in ACTION_VERBS: # Assumed defined elsewhere
                       continue
                    # For Triplet Fallback: specific (subj, obj) relations
                    vg_pair_dict[subj][obj].add(v)
                    # For Prior Bank: general class affordances
                    vg_prior_counts[subj][v] += 1             
        
    # print("3. Scanning Training Dataset for Object Pairs...")
    class_frequencies = defaultdict(int)
    observed_pairs = set()
    scene_root_dir = scene_dir
    scene_graphs = [f for f in os.listdir(scene_root_dir) if f.endswith(".json")]
  
    
    for scene_graph_path in tqdm(scene_graphs, desc="Loading Scene Graphs"):
        with open(os.path.join(scene_root_dir , scene_graph_path), 'r') as f:
            dataset_scene_graphs = json.load(f)

        for graph in dataset_scene_graphs['frames']:
            objects = graph["objects"] 
            for obj in objects: # each object is a dict with "class_name", "id", "bbox", etc.
                class_frequencies[obj['class_name']] += 1
            # Get all permutations of edges
            for i in range(len(objects)):
                for j in range(len(objects)):
                    # add paris in observed_pairs only if they are different objects to avoid self-loops (which are less informative for affordances)
                    if i != j:
                            observed_pairs.add((objects[i]['class_name'], objects[j]['class_name']))
                    # if i != j: observed_pairs.add((objects[i], objects[j]))

    
    print("4. Building Hybrid Triplet Bank (Frontdoor M)...")
    triplet_bank = {}
    MAX_TRIPLETS_PER_PAIR = 10  # Enforce a strict maximum!
    
    # from collections import Counter
    # import torch

    # class_list = list(COCO_CLASSES)  
    # label_to_idx = {cls: i for i, cls in enumerate(class_list)}
    # counts = Counter()

    # for (subj, obj) in observed_pairs:
    #     counts[label_to_idx[subj]] += 1
    #     counts[label_to_idx[obj]] += 1

    # # If you want pair counts:
    # # pair_counts = Counter()
    # # for (subj, obj) in observed_pairs:
    # #     pair_key = f"{subj}____{obj}"
    # #     pair_counts[pair_key] += 1

    # # Normalize to probabilities or embed as needed:
    # prior_array = torch.zeros(len(class_list))
    # for idx in range(len(class_list)):
    #     prior_array[idx] = counts[idx]
    # emp_prior = prior_array / (prior_array.sum() + 1e-8)  # To get p(z)
    # torch.save(emp_prior, f"{save_dir}/E_z_priors.pt")
    
    for (subj, obj) in observed_pairs:
        pair_key = f"{subj}____{obj}"
        triplets = []
        
        # ── STRATEGY A: Try ConceptNet First (High Quality) ──
        # Gather ALL causal relations for the subject with their weights
        all_cn_targets = []
        for rel, targets in cn_dict.get(subj, {}).items():
            for t_dict in targets:
                all_cn_targets.append({
                    "rel": rel,
                    "target": t_dict["target"],
                    "weight": t_dict["weight"]
                })
                
        # Sort globally by weight and take only the absolute Top K
        all_cn_targets = sorted(all_cn_targets, key=lambda x: x["weight"], reverse=True)
        
        # for item in all_cn_targets[:MAX_TRIPLETS_PER_PAIR]: 
        #     triplets.append(f"a {subj} is {item['rel']} {item['target']} involving a {obj}")
        for item in all_cn_targets[:MAX_TRIPLETS_PER_PAIR]:
            rel = item['rel']
            template_list = TRIPLET_TEMPLATES.get(rel, TRIPLET_TEMPLATES['Default'])
            template = random.choice(template_list)
            triplet = template.format(
                subj=subj,
                obj=obj,
                rel=rel,
                target=item['target']
                )
            triplets.append(triplet)
        # ── STRATEGY B: Visual Genome Fallback ──
        # If we have fewer than 3 triplets, we inject VG actions to help out
        
        if len(triplets) < 3 and obj in vg_pair_dict.get(subj, {}):
            for verb in list(vg_pair_dict[subj][obj])[:5]:
                sentence = random.choice(vg_templates).format(subj=subj, obj=obj, verb=verb)
                if sentence not in triplets:
                    triplets.append(sentence)
            # for verb in list(vg_pair_dict[subj][obj])[:5]:
            #     sentence = f"a {subj} {verb}s a {obj}"
            #     if sentence not in triplets: # Prevent duplicates without using set()
            #         triplets.append(sentence)
                
        # ── STRATEGY C: Structural Fallback ──
        if not triplets:
            fallback_verbs = CONCEPTNET_FALLBACK.get(subj, {}) 
            if fallback_verbs:
                for verb, _ in list(fallback_verbs.items())[:5]:
                    triplets.append(f"a {subj} {verb}s a {obj}")
            else:
                triplets.append(f"a {subj} interacts with a {obj}")
            
        # STRICTLY CAP AT MAX_TRIPLETS and preserve the sorted order!
        triplet_bank[pair_key] = triplets[:MAX_TRIPLETS_PER_PAIR]
    
    print("5. Building Hybrid Prior Bank (Backdoor E_z)...")
    prior_bank = {}
    MAX_PRIORS_PER_CLASS = 10  # Richer context for the node prior
    
    # We build a prior for EVERY class in your vocabulary
    for cls in COCO_CLASSES:
        affordances = []
        
        # ── 1. Combine ConceptNet targets (Sorted globally by weight) ──
        all_cn_targets = []
        for rel, targets in cn_dict.get(cls, {}).items():
            for t_dict in targets:
                if rel not in CAUSAL_RELS:
                    continue
                if is_obj_in_target(obj, t_dict['target']):
                    all_cn_targets.append({
                        "target": t_dict["target"],
                        "weight": t_dict["weight"],
                        "rel": rel
                    })
                
        all_cn_targets = sorted(all_cn_targets, key=lambda x: x["weight"], reverse=True)
        for item in all_cn_targets:
            rel = item['rel']
            template = random.choice(PRIOR_TEMPLATES)
            affordance = template.format(cls=cls, rel=rel, target=item['target'])
            affordances.append(affordance)
        # for item in all_cn_targets:
        #     affordances.append(f"a {cls} is {item['rel']} {item['target']}")
            
        # ── 2. Combine top VG action verbs ──
        # Use .get() just in case the class never appeared in VG at all
        sorted_vg_verbs = sorted(vg_prior_counts.get(cls, {}).items(), key=lambda x: x[1], reverse=True)
        for verb, count in sorted_vg_verbs:
            sentence = f"a {cls} typically {verb}s"
            if sentence not in affordances:
                affordances.append(sentence)
                
        # ── 3. STRATEGIC FALLBACK for completely empty classes ──
        if not affordances:
            fallback_verbs = CONCEPTNET_FALLBACK.get(cls, {})
            if fallback_verbs:
                # If we have dictionary fallbacks, use them!
                for verb, _ in list(fallback_verbs.items())[:5]:
                    template = random.choice(PRIOR_TEMPLATES)
                    affordance = template.format(cls=cls, rel=verb, target=item['target'])
                    affordances.append(affordance)
                    # affordances.append(f"a {cls} is capable of {verb}ing")
            else:
                # Ultimate generic fallback to guarantee at least one valid sentence
                affordances.append(f"a {cls} is a visual object present in the scene")
        
        # STRICTLY CAP AT MAX_PRIORS (and remember, no P(z) here!)
        prior_bank[cls] = {
            "sentences": affordances[:MAX_PRIORS_PER_CLASS]
        }
        
    with open(os.path.join(save_dir, "hybrid_triplet_bank.json"), "w") as f:
        json.dump(triplet_bank, f, indent=2)
    with open(os.path.join(save_dir, "hybrid_prior_bank.json"), "w") as f:
        json.dump(prior_bank, f, indent=2)
    
    print("Saved Hybrid JSON banks successfully!")

    
# =====================================================================
# RUN ONCE OFFLINE: Convert JSON text to embedded Tensors
# =====================================================================
# e.g., YOLO_CLASS_VOCAB = {"person": 0, "bicycle": 1, "car": 2, ...}

def encode_banks_to_tensors(triplet_json_path, prior_json_path, bank_path):
    encoder = DebertaEncoder(debrata_path)
    
    with open(triplet_json_path) as f:
        triplet_dict = json.load(f)
    with open(prior_json_path) as f:
        prior_dict = json.load(f)

    # Calculate num_classes based on the maximum YOLO ID + 1 (usually 80)
    num_classes = max(YOLO_CLASS_VOCAB.values()) + 1 
    dim = 1024  # DeBERTa embedding dimension
    expected_priors = torch.zeros((num_classes, dim))
    
    # 1. Encode Expected Priors (E_z)
    for cls, data in tqdm(prior_dict.items(), desc="Encoding E_z Priors"):
        # Skip if the class isn't in our strict YOLO vocabulary
        if cls not in YOLO_CLASS_VOCAB or cls not in manual_map or cls not in alias_map:
            continue
        if cls in YOLO_CLASS_VOCAB:
            cls = cls
        elif cls in manual_map:
            cls = manual_map.get(cls, cls)
        else:
            cls = alias_map.get(cls, cls)
            
        cls_id = YOLO_CLASS_VOCAB[cls]
        sentences = data["sentences"]
        
        if sentences:
            # embeddings = torch.tensor(encoder.encode(sentences))
            embeddings  = torch.tensor(encoder.encode(sentences, max_length=12, pooling="cls"))
            expected_priors[cls_id] = embeddings.mean(dim=0) 
    torch.save(expected_priors, f"{bank_path}/E_z_priors.pt")

    # 2. Encode Triplet Bank (M)
    triplet_tensors = {}
    for pair_key, sentences in tqdm(triplet_dict.items(), desc="Encoding M Triplets"):
        if not sentences: 
            continue
            
        subj, obj = pair_key.split("____")
        
        # Only process if BOTH subject and object are valid YOLO classes
        if subj in COCO_CLASSES or subj in manual_map or subj in alias_map:
            if subj in COCO_CLASSES:
                subj = subj
            elif subj in manual_map:
                subj = manual_map.get(subj, subj) 
            else:
                subj = alias_map.get(subj, subj)
                
        if obj in COCO_CLASSES or obj in manual_map or obj in alias_map:
            if obj in COCO_CLASSES:
                obj = obj
            elif obj in manual_map:
                obj = manual_map.get(obj, obj)
            else:
                obj = alias_map.get(obj, obj)

        if subj in YOLO_CLASS_VOCAB and obj in YOLO_CLASS_VOCAB:
            subj_id = YOLO_CLASS_VOCAB[subj]
            obj_id = YOLO_CLASS_VOCAB[obj]
            
            # emb = torch.tensor(encoder.encode(sentences)) # (num_sentences, dim)
            emb = torch.tensor(encoder.encode(sentences, max_length=16, pooling="cls"))
            triplet_tensors[(subj_id, obj_id)] = emb
            print(f'---------triplets shape N: {emb.shape}=======')
            
    torch.save(triplet_tensors, f"{bank_path}/M_triplets.pt")
    print(f"Successfully encoded tensors mapped strictly to {num_classes} YOLO classes!")

# =====================================================================
# PYTORCH MODULES
# =====================================================================


       
if __name__ == "__main__":
    save_dir = '/root/autodl-tmp/CausalSTGNet/datasets/multi-choice_vqa/TGIF/trans/prior_memory/visual'
    scene_dir = '/root/autodl-tmp/CausalSTGNet/datasets/multi-choice_vqa/TGIF/trans/train_masks'
    msvd_visual_prior_path = '/root/autodl-tmp/CausalSTGNet/datasets/multi-choice_vqa/TGIF/trans/prior_memory/visual/hybrid_prior_bank.json'
    msvd_visual_triplet_path = '/root/autodl-tmp/CausalSTGNet/datasets/multi-choice_vqa/TGIF/trans/prior_memory/visual/hybrid_triplet_bank.json'
    
    # build_hybrid_knowledge_banks(concept_csv_path, relationship_path, scene_dir, save_dir)
    encode_banks_to_tensors(msvd_visual_triplet_path, msvd_visual_prior_path, save_dir)
    


    
