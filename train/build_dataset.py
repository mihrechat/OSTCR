import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
from torch_geometric.data import Data, Batch as PyGBatch
import shutil



YOLO_CLASS_VOCAB = {
    "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "airplane": 4,
    "bus": 5, "train": 6, "truck": 7, "boat": 8, "traffic light": 9,
    "fire hydrant": 10, "stop sign": 11, "parking meter": 12, "bench": 13,
    "bird": 14, "cat": 15, "dog": 16, "horse": 17, "sheep": 18, "cow": 19,
    "elephant": 20, "bear": 21, "zebra": 22, "giraffe": 23, "backpack": 24,
    "umbrella": 25, "handbag": 26, "tie": 27, "suitcase": 28, "frisbee": 29,
    "skis": 30, "snowboard": 31, "sports ball": 32, "kite": 33,
    "baseball bat": 34, "baseball glove": 35, "skateboard": 36,
    "surfboard": 37, "tennis racket": 38, "bottle": 39, "wine glass": 40,
    "cup": 41, "fork": 42, "knife": 43, "spoon": 44, "bowl": 45,
    "banana": 46, "apple": 47, "sandwich": 48, "orange": 49, "broccoli": 50,
    "carrot": 51, "hot dog": 52, "pizza": 53, "donut": 54, "cake": 55,
    "chair": 56, "couch": 57, "potted plant": 58, "bed": 59,
    "dining table": 60, "toilet": 61, "tv": 62, "laptop": 63, "mouse": 64,
    "remote": 65, "keyboard": 66, "cell phone": 67, "microwave": 68,
    "oven": 69, "toaster": 70, "sink": 71, "refrigerator": 72, "book": 73,
    "clock": 74, "vase": 75, "scissors": 76, "teddy bear": 77,
    "hair drier": 78, "toothbrush": 79,
}

# ==================================================================
# Predicate vocab sizes (must match RELTR_PREDICATES + temporal)
# ==================================================================
# NUM_SPATIAL_RELS  = 51   # 0=background ... 50=with (RelTR 0-indexed)
# NUM_TEMPORAL_RELS = 1    # 51="follows"
# NUM_PREDICATES    = 52   # total for nn.Embedding
# NUM_YOLO_CLASSES = 80


# ==================================================================
# Dataset
# ==================================================================

class STGraphData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key in ('s_idx', 't_idx'):
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ('s_idx', 't_idx'):
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)
    
class STGraphDataset(Dataset):
    def __init__(self, cfg):
        """
        root_dir/
            video_001/
                node_raw.npy          (N_total, 1280)  float16
                node_proj.npy         (N_total, 2048)  float16
                node_metadata.json 
            video_002/
                ...
        """
        if cfg.model.test: 
            self.root_dir           = cfg.data.test_root
        else:
            self.root_dir           = cfg.data.test_root
        self.NUM_YOLO_CLASSES   = cfg.model.num_classes
        self.NUM_PREDICATES     = cfg.model.num_preds
        self.YOLO_CLASS_VOCAB   = YOLO_CLASS_VOCAB
        
        self.required_files = [
        "node_layer1.npy",
        "node_layer2.npy",
        "node_layer3.npy", 
        "node_layer4.npy",
        # "node_proj.npy",
        "node_metadata.json",
        "question_emb.npy",
        "options_emb.npy",
        "question.json"
        ]
        
        
        # all_dirs = sorted([
        #     os.path.join(self.root_dir, d)
        #     for d in os.listdir(self.root_dir)
        #     if os.path.isdir(os.path.join(self.root_dir, d))
        # ])
        
        all_dirs = sorted([
        os.path.join(self.root_dir, d)
        for d in os.listdir(self.root_dir)
        if os.path.isdir(os.path.join(self.root_dir, d))
        ])
        
        self.video_dirs = []
        self.invalid_dirs = []
        for video_dir in all_dirs:
            if self._has_required_files(video_dir):
                self.video_dirs.append(video_dir)
            else:
                self.invalid_dirs.append(video_dir)
                missing = self._get_missing_files(video_dir)
                print(f"Warning: Skipping {video_dir} - missing files: {missing}")
        
        print(f"Found {len(self.video_dirs)} valid video directories")
        if self.invalid_dirs:
            print(f"Skipped {len(self.invalid_dirs)} invalid directories")

    def _has_required_files(self, video_dir: str) -> bool:
        """Check if all required files exist in the directory."""
        return all(
            os.path.exists(os.path.join(video_dir, file))
            for file in self.required_files
        )

    def _get_missing_files(self, video_dir: str) -> list:
        """Return list of missing required files."""
        return [
            file for file in self.required_files
            if not os.path.exists(os.path.join(video_dir, file))
        ]
   
    def __len__(self):
        return len(self.video_dirs)
    

    def __getitem__(self, idx):
        return self._load_video(self.video_dirs[idx])
    def _load_video(self, video_dir: str) -> Data:

        # ── 1: load node features ──────────────────────────────────
        node_raw  = np.load(os.path.join(video_dir, "node_layer1.npy") ).astype(np.float32)
        # node_proj = np.load(os.path.join(video_dir, "node_proj.npy")).astype(np.float32)
        
        #Load the new text embeddings!
        q_emb     = np.load(os.path.join(video_dir, "question_emb.npy")).astype(np.float32)
        opt_embs  = np.load(os.path.join(video_dir, "options_emb.npy")).astype(np.float32)
        
        
        with open(os.path.join(video_dir, "node_metadata.json")) as f:
            meta = json.load(f)
        with open(os.path.join(video_dir, 'question.json')) as f:
            question = json.load(f)

        # ── 2: node identity ───────────────────────────────────────
        node_frame_index  = meta["node_frame_index"]          # [frame_id,  ...]
        node_kf_list_idx  = meta["node_keyframe_list_idx"]    # [kf_idx,    ...]
        node_is_keyframe  = meta["node_is_keyframe"]          # [bool,      ...]
        node_obj_ids_flat = meta["node_obj_ids_flat"]         # [tracked_id,...]
        keyframe_indices  = meta["keyframe_indices"]          # [frame_id,  ...]
        num_keyframes     = int(meta["num_keyframes"])

        # ── 3: node attributes ─────────────────────────────────────
        node_bbox  = np.array(meta["node_bbox"],  dtype=np.float32)   # (N, 4) + [H, W]
        node_conf  = np.array(meta["node_conf"],  dtype=np.float32)   # (N,)
        
        # ── node class — fixed COCO vocab, no dynamic encoding needed ──
        node_class_ids = torch.tensor(
            [self.YOLO_CLASS_VOCAB.get(c, 0) for c in meta["node_class"]],
            dtype=torch.long
        )

        # ── 4: spatial edges ───────────────────────────────────────
        s_data  = meta.get("spatial_relations",  {})
        s_edges = s_data.get("edges",  [])
        s_attrs = s_data.get("attrs",  [])
        s_scores= s_data.get("scores", [])

        if len(s_edges) > 0:
            s_idx   = torch.tensor(s_edges,  dtype=torch.long).T    # (2, E_s)
            s_attr  = torch.tensor(s_attrs,  dtype=torch.long)      # (E_s,) int
            s_score = torch.tensor(s_scores, dtype=torch.float32)   # (E_s,)
        else:
            s_idx   = torch.zeros(2, 0, dtype=torch.long)
            s_attr  = torch.zeros(0,    dtype=torch.long)
            s_score = torch.zeros(0,    dtype=torch.float32)

        # ── 5: temporal edges ──────────────────────────────────────
        t_data  = meta.get("temporal_relations", {})
        t_edges = t_data.get("edges", [])
        t_attrs = t_data.get("attrs", [51])
        # t_attrs = torch.tensors(2)

        if len(t_edges) > 0:
            t_idx  = torch.tensor(t_edges, dtype=torch.long).T      # (2, E_t)
            # t_attrs may only have 1 entry — expand to match edge count
            if len(t_attrs) == 1:
                t_attrs = t_attrs * len(t_edges)
            t_attr = torch.tensor(t_attrs, dtype=torch.long)        # (E_t,)
        else:
            t_idx  = torch.zeros(2, 0, dtype=torch.long)
            t_attr = torch.zeros(0,    dtype=torch.long)
            
        
        # ── 6: pack into PyG Data ──────────────────────────────────
        data = STGraphData(
            node_raw          = torch.tensor(node_raw,  dtype=torch.float32),
            # node_proj         = torch.tensor(node_proj, dtype=torch.float32),
            node_bbox         = torch.tensor(node_bbox, dtype=torch.float32),
            node_conf         = torch.tensor(node_conf, dtype=torch.float32),
            node_class        = node_class_ids,
            node_obj_ids_flat = torch.tensor(node_obj_ids_flat, dtype=torch.long),
            node_frame_index  = torch.tensor(node_frame_index,  dtype=torch.long),
            node_kf_list_idx  = torch.tensor(node_kf_list_idx,  dtype=torch.long),
            node_is_keyframe  = torch.tensor(node_is_keyframe,  dtype=torch.bool),
            s_idx             = s_idx,
            s_attr            = s_attr,
            s_score           = s_score,
            t_idx             = t_idx,
            t_attr            = t_attr,
            num_keyframes     = torch.tensor(num_keyframes, dtype=torch.long),
            num_nodes         = torch.tensor(node_raw.shape[0], dtype=torch.long),
            question_emb             = torch.tensor(q_emb, dtype=torch.float32),
            options_emb          = torch.tensor(opt_embs, dtype=torch.float32),
            qtype_idx         = torch.tensor([question["qtype_idx"]], dtype=torch.long),
            triplet_idxs      = torch.tensor([question["triplet_idxs"]], dtype=torch.long),
            triplet_mask      = torch.tensor([question["triplet_mask"]], dtype=torch.bool),
            answers           = torch.tensor([question["answer"]], dtype=torch.long),
        )

        return data

class MSVDDataset(Dataset):
    def __init__(self, cfg, split="train"):
        """
        Args:
            cfg: configuration object
            split (str): "train", "val", or "test". Dictates which JSON split to load and which folder to read from.
        """
        self.split = split
        
        # ── 1. Determine root directory based on the exact split ──
        if split == "test": 
            self.root_dir = cfg.data.test_root
        elif split == "val":
            self.root_dir = cfg.data.val_root 
        else:
            self.root_dir = cfg.data.train_root 

        self.NUM_YOLO_CLASSES   = cfg.model.num_classes
        self.NUM_PREDICATES     = cfg.model.num_preds
        self.YOLO_CLASS_VOCAB   = YOLO_CLASS_VOCAB
        self.layer              = cfg.model.node_feature_layer
        
        # We only require the SHARED video files to validate a directory
        self.shared_video_files = [
            "node_layer1.npy",
            "node_layer2.npy",
            "node_layer3.npy",
            "node_layer4.npy",
            "motion_features.npy",
            "node_metadata.json",
        ]
        
        self.qa_pairs = [] 
        self.invalid_dirs = set() # Use a set so we don't print warnings multiple times per video
        
        # ── 2. Load the exact question list for this split ──
        split_file = os.path.join(cfg.data.question_dir, f"{split}_split.json")
        print(f"Loading {split.upper()} split from {split_file}...")
        
        with open(split_file, "r", encoding="utf-8") as f:
            questions = json.load(f)
            
        # ── 3. Build the dataset using ONLY the allowed questions ──
        for q in questions:
            video_name = q['video_name']
            q_id = q['id']
            
            # Now safely looks inside /train, /val, or /test based on the split
            video_dir = os.path.join(self.root_dir, video_name)
            
            # Check if the shared video features exist
            if not self._has_shared_files(video_dir):
                if video_dir not in self.invalid_dirs:
                    self.invalid_dirs.add(video_dir)
                    missing = self._get_missing_files(video_dir)
                    print(f"Warning: Skipping {video_dir} - missing core files: {missing}")
                continue
                
            # Define specific offline question paths
            q_json_path = os.path.join(video_dir, f"q_{q_id}.json")
            q_emb_path  = os.path.join(video_dir, f"q_{q_id}_emb.npy")
            
            # Ensure the offline extraction was successful for this specific question
            if os.path.exists(q_json_path) and os.path.exists(q_emb_path):
                self.qa_pairs.append({
                    "video_dir": video_dir,
                    "q_json_path": q_json_path,
                    "q_emb_path": q_emb_path,
                })
            else:
                # Useful for catching offline extraction bugs
                print(f"Warning: Missing extracted features for question ID {q_id} in {video_name}")
        
        print(f"[{split.upper()}] Successfully loaded {len(self.qa_pairs)} valid question-answer pairs.")
        if self.invalid_dirs:
            print(f"[{split.upper()}] Skipped {len(self.invalid_dirs)} invalid video directories.")

    def _has_shared_files(self, video_dir: str) -> bool:
        """Check if all required shared video files exist in the directory."""
        return all(
            os.path.exists(os.path.join(video_dir, file))
            for file in self.shared_video_files
        )

    def _get_missing_files(self, video_dir: str) -> list:
        """Return list of missing required files."""
        return [
            file for file in self.shared_video_files
            if not os.path.exists(os.path.join(video_dir, file))
        ]
   
    def __len__(self):
        # Return the number of specific QUESTIONS
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        # Pass the dictionary containing the specific paths
        return self._load_sample(self.qa_pairs[idx])
    
    def _load_sample(self, sample_info: dict): # -> Data:
        video_dir = sample_info["video_dir"]
        q_json_path = sample_info["q_json_path"]
        q_emb_path = sample_info["q_emb_path"]

        # ── 1: load shared node features (Same for every question in this video) ──
        node_raw  = np.load(os.path.join(video_dir, self.layer + ".npy")).astype(np.float32)
        # node_proj = np.load(os.path.join(video_dir, "node_proj.npy")).astype(np.float32)
        
        with open(os.path.join(video_dir, "node_metadata.json")) as f:
            meta = json.load(f)
        
        with open(q_json_path) as f:
            question = json.load(f)
        
        q_emb     = np.load(q_emb_path).astype(np.float32)
        
        # ── 2: node identity ───────────────────────────────────────
        node_frame_index  = meta["node_frame_index"]          # [frame_id,  ...]
        node_kf_list_idx  = meta["node_keyframe_list_idx"]    # [kf_idx,    ...]
        node_is_keyframe  = meta["node_is_keyframe"]          # [bool,      ...]
        node_obj_ids_flat = meta["node_obj_ids_flat"]         # [tracked_id,...]
        # keyframe_indices  = meta["keyframe_indices"]          # [frame_id,  ...]
        num_keyframes     = int(meta["num_keyframes"])

        # ── 3: node attributes ─────────────────────────────────────
        node_bbox  = np.array(meta["node_bbox"],  dtype=np.float32)   # (N, 4) + [H, W]
        node_conf  = np.array(meta["node_conf"],  dtype=np.float32)   # (N,)
        
        # ── node class — fixed COCO vocab, no dynamic encoding needed ──
        node_class_ids = torch.tensor(
            [self.YOLO_CLASS_VOCAB.get(c, 0) for c in meta["node_class"]],
            dtype=torch.long
        )

        # ── 4: spatial edges ───────────────────────────────────────
        s_data  = meta.get("spatial_relations",  {})
        s_edges = s_data.get("edges",  [])
        s_attrs = s_data.get("attrs",  [])
        s_scores= s_data.get("scores", [])

        if len(s_edges) > 0:
            s_idx   = torch.tensor(s_edges,  dtype=torch.long).T    # (2, E_s)
            s_attr  = torch.tensor(s_attrs,  dtype=torch.long)      # (E_s,) int
            s_score = torch.tensor(s_scores, dtype=torch.float32)   # (E_s,)
        else:
            s_idx   = torch.zeros(2, 0, dtype=torch.long)
            s_attr  = torch.zeros(0,    dtype=torch.long)
            s_score = torch.zeros(0,    dtype=torch.float32)

        # ── 5: temporal edges ──────────────────────────────────────
        t_data  = meta.get("temporal_relations", {})
        t_edges = t_data.get("edges", [])
        t_attrs = t_data.get("attrs", [51])
        # t_attrs = torch.tensors(2)

        if len(t_edges) > 0:
            t_idx  = torch.tensor(t_edges, dtype=torch.long).T      # (2, E_t)
            # t_attrs may only have 1 entry — expand to match edge count
            if len(t_attrs) == 1:
                t_attrs = t_attrs * len(t_edges)
            t_attr = torch.tensor(t_attrs, dtype=torch.long)        # (E_t,)
        else:
            t_idx  = torch.zeros(2, 0, dtype=torch.long)
            t_attr = torch.zeros(0,    dtype=torch.long)
            
        
        # ── 6: pack into PyG Data ──────────────────────────────────
        data = STGraphData(
            node_raw          = torch.tensor(node_raw,  dtype=torch.float32),
            # node_proj         = torch.tensor(node_proj, dtype=torch.float32),
            node_bbox         = torch.tensor(node_bbox, dtype=torch.float32),
            node_conf         = torch.tensor(node_conf, dtype=torch.float32),
            node_class        = node_class_ids,
            node_obj_ids_flat = torch.tensor(node_obj_ids_flat, dtype=torch.long),
            node_frame_index  = torch.tensor(node_frame_index,  dtype=torch.long),
            node_kf_list_idx  = torch.tensor(node_kf_list_idx,  dtype=torch.long),
            node_is_keyframe  = torch.tensor(node_is_keyframe,  dtype=torch.bool),
            s_idx             = s_idx,
            s_attr            = s_attr,
            s_score           = s_score,
            t_idx             = t_idx,
            t_attr            = t_attr,
            num_keyframes     = torch.tensor(num_keyframes, dtype=torch.long),
            num_nodes         = torch.tensor(node_raw.shape[0], dtype=torch.long),
            question_emb      = torch.tensor(q_emb, dtype=torch.float32),   
            qtype_idx         = torch.tensor([question["qtype_idx"]], dtype=torch.long),
            triplet_idxs      = torch.tensor([question["triplet_idxs"]], dtype=torch.long),
            triplet_mask      = torch.tensor([question["triplet_mask"]], dtype=torch.bool),
            answers           = torch.tensor([question["answer"]], dtype=torch.long),
        )

        return data

    

class OpenEndedDataMotion(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key in ('s_idx', 't_idx'):
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ('s_idx', 't_idx'):
            return 1
        
        # Tell PyG to STACK these attributes into a new Batch dimension
        # Instead of flattening them, motion_feat will become (B, T, Dim)
        if key in ('motion_feat'):
            return None 
        
        if key in ('question_emb', "question_mask"):
            return None
            
        return super().__cat_dim__(key, value, *args, **kwargs)


class MSVDDatasetMotion(Dataset):
    def __init__(self, cfg, split="train"):
        """
        Args:
            cfg: configuration object
            split (str): "train", "val", or "test". Dictates which JSON split to load and which folder to read from.
        """
        self.split = split
        
        # ── 1. Determine root directory based on the exact split ──
        if split == "test": 
            self.root_dir = cfg.data.test_root
        elif split == "val":
            self.root_dir = cfg.data.val_root 
        else:
            self.root_dir = cfg.data.train_root 

        self.NUM_YOLO_CLASSES   = cfg.model.num_classes
        self.NUM_PREDICATES     = cfg.model.num_preds
        self.YOLO_CLASS_VOCAB   = YOLO_CLASS_VOCAB # Make sure this is imported/defined
        self.feature_layer      = cfg.data.feature_layer
        
        
        self.shared_video_files = [
            "node_layer1.npy",
            "node_layer2.npy",
            "node_layer3.npy",
            "node_layer4.npy",
            "motion_features.npy",    
            "node_metadata.json",
        
        ]
        
        self.qa_pairs = [] 
        self.invalid_dirs = set() 
        
        # ── 2. Load the exact question list for this split ──
        # split_file = os.path.join(cfg.data.question_dir, f"{split}_split.json")
        # print(f"Loading {split.upper()} split from {split_file}...")
        
        # with open(split_file, "r", encoding="utf-8") as f:
        #     questions = json.load(f)
            
        # ── 3. Build the dataset using ONLY the allowed questions ──
         # ── 2. Scan directories directly for extracted questions ──
        print(f"Scanning {split.upper()} directory: {self.root_dir}...")
        video_folders = [f.path for f in os.scandir(self.root_dir) if f.is_dir()]
        for video_dir in video_folders:
            video_name = os.path.basename(video_dir)
            # video_id = os.path.basename(video_dir)
        # for q in questions:
        #     video_name = q['video_name']
            # q_id = q['id']
            # video_dir = os.path.join(self.root_dir, video_name)
            
            # Remove empty directories immediately
            if self._is_directory_empty(video_dir):
                # # print(f"Removing empty directory: {video_name}")
                shutil.rmtree(video_dir)
                continue
            
            if not self._has_shared_files(video_dir):
                if video_dir not in self.invalid_dirs:
                    self.invalid_dirs.add(video_dir)
                    missing = self._get_missing_files(video_dir)
                    shutil.rmtree(video_dir)
                    print(f"Warning: Skipping {video_name} - missing core files: {missing}")
                continue
                
            # Find all question JSONs in this video folder (e.g., q_1.json, q_2.json)
            q_jsons = glob.glob(os.path.join(video_dir, "q_*.json"))
            if not q_jsons:
                print(f"Removing directory with no questions: {video_name}")
                shutil.rmtree(video_dir)
                continue
        
            
            for q_json_path in q_jsons:
                # Extract the base name (e.g., "q_1")
                q_base = os.path.splitext(os.path.basename(q_json_path))[0]
                
                q_emb_path  = os.path.join(video_dir, f"{q_base}_emb.npz")
                
                # q_json_path = os.path.join(video_dir, f"q_{q_id}.json")
                # q_emb_path  = os.path.join(video_dir, f"q_{q_id}_emb.npy")
                # q_emb_path  = os.path.join(video_dir, f"q_{q_id}_emb.npz")
                
                if os.path.exists(q_emb_path):
                    self.qa_pairs.append({
                        "video_dir": video_dir,
                        "q_json_path": q_json_path,
                        "q_emb_path": q_emb_path,
                    })
                else:
                    print(f"Warning: Missing extracted features for question ID {q_base} in {video_name}")
        
        print(f"[{split.upper()}] Successfully loaded {len(self.qa_pairs)} valid question-answer pairs.")

    def _has_shared_files(self, video_dir: str) -> bool:
        return all(os.path.exists(os.path.join(video_dir, file)) for file in self.shared_video_files)

    def _get_missing_files(self, video_dir: str) -> list:
        return [file for file in self.shared_video_files if not os.path.exists(os.path.join(video_dir, file))]
   
    def __len__(self):
        return len(self.qa_pairs)
    
    # def __getitem__(self, idx):
    #     return self._load_sample(self.qa_pairs[idx])
    
    def __getitem__(self, idx):

        tries = 0

        while tries < 20:

            try:

                sample = self._load_sample(self.qa_pairs[idx])

                return sample

            except Exception as e:
                #show what exactly is going wrong with and which sample is corrupted and show directory and question id for debugging
                print(f"[Error loading sample] idx={idx} from {self.qa_pairs[idx]['video_dir']}", end=" - ", flush=True)
                #remove the corrupted file(qa_pair) from the dataset to avoid future errors
                # self.qa_pairs.pop(idx)
                # os.remove(self.qa_pairs[idx]['q_json_path'])
                # os.remove(self.qa_pairs[idx]['q_emb_path'])
                
                print(f"[Dataset Error] idx={idx}", end=" - ", flush=True, )
                print(e)

                idx = random.randint(0, len(self.qa_pairs) - 1)

                tries += 1

        raise RuntimeError("Too many corrupted samples.")
    
    def _is_directory_empty(self, video_dir):
        """Check if directory is empty or only contains empty directories"""
        try:
            return not any(os.scandir(video_dir))
        except:
            return True
    
    def _validate_sample(
        self,
        node_raw,
        motion_raw,
        node_bbox,
        node_conf,
        node_class_ids,
        s_idx,
        t_idx,
        q_emb,
        q_mask
    ):

 

        N = node_raw.shape[0]

        if N == 0:
            return False

        if torch.isnan(torch.tensor(node_raw)).any():
            return False

        if torch.isinf(torch.tensor(node_raw)).any():
            return False

   

        if q_emb.shape[0] != q_mask.shape[0]:
            return False

        if q_emb.shape[0] == 0:
            return False

    

        if motion_raw.shape[0] == 0:
            return False

  =

        if s_idx.numel() > 0:

            if s_idx.dim() != 2 or s_idx.shape[0] != 2:
                return False

            if s_idx.max() >= N:
                return False

            if s_idx.min() < 0:
                return False

    

        if t_idx.numel() > 0:

            if t_idx.dim() != 2 or t_idx.shape[0] != 2:
                return False

            if t_idx.max() >= N:
                return False

            if t_idx.min() < 0:
                return False

    

        if len(node_bbox) != N:
            return False

        if len(node_conf) != N:
            return False

        if len(node_class_ids) != N:
            return False

        return True
    
    
    def _load_sample(self, sample_info: dict): # -> Data:
        video_dir = sample_info["video_dir"]
        q_json_path = sample_info["q_json_path"]
        q_emb_path = sample_info["q_emb_path"]

        # ── 1: load shared node features (Same for every question in this video) ──
        node_raw  = np.load(os.path.join(video_dir, f"{self.feature_layer}")).astype(np.float32)
        motion_raw = np.load(os.path.join(video_dir, "motion_features.npy")).astype(np.float32)
        
        with open(os.path.join(video_dir, "node_metadata.json")) as f:
            meta = json.load(f)
        
        with open(q_json_path) as f:
            question = json.load(f)
        
        # q_emb     = np.load(q_emb_path).astype(np.float32)
        data = np.load(q_emb_path)
        q_emb = data['emb'].astype(np.float32)
        q_mask = data['mask']
        
        # ── 2: node identity ───────────────────────────────────────
        node_frame_index  = meta["node_frame_index"]          # [frame_id,  ...]
        node_obj_ids_flat = meta["node_obj_ids_flat"]         # [tracked_id,...]
 

        # ── 3: node attributes ─────────────────────────────────────
        node_bbox  = np.array(meta["node_bbox"],  dtype=np.float32)   # (N, 4) + [H, W]
        node_conf  = np.array(meta["node_conf"],  dtype=np.float32)   # (N,)
        
        # ── node class — fixed COCO vocab, no dynamic encoding needed ──
        node_class_ids = torch.tensor(
            [self.YOLO_CLASS_VOCAB.get(c, 0) for c in meta["node_class"]],
            dtype=torch.long
        )

        # ── 4: spatial edges ───────────────────────────────────────
        s_data  = meta.get("spatial_relations",  {})
        s_edges = s_data.get("edges",  [])
        s_attrs = s_data.get("attrs",  [])
        s_scores= s_data.get("scores", [])

        if len(s_edges) > 0:
            s_idx   = torch.tensor(s_edges,  dtype=torch.long).T    # (2, E_s)
            s_attr  = torch.tensor(s_attrs,  dtype=torch.long)      # (E_s,) int
            s_score = torch.tensor(s_scores, dtype=torch.float32)   # (E_s,)
        else:
            s_idx   = torch.zeros(2, 0, dtype=torch.long)
            s_attr  = torch.zeros(0,    dtype=torch.long)
            s_score = torch.zeros(0,    dtype=torch.float32)

        # ── 5: temporal edges ──────────────────────────────────────
        t_data  = meta.get("temporal_relations", {})
        t_edges = t_data.get("edges", [])
        t_attrs = t_data.get("attrs", [51])
        # t_attrs = torch.tensors(2)

        if len(t_edges) > 0:
            t_idx  = torch.tensor(t_edges, dtype=torch.long).T      # (2, E_t)
            # t_attrs may only have 1 entry — expand to match edge count
            if len(t_attrs) == 1:
                t_attrs = t_attrs * len(t_edges)
            t_attr = torch.tensor(t_attrs, dtype=torch.long)        # (E_t,)
        else:
            t_idx  = torch.zeros(2, 0, dtype=torch.long)
            t_attr = torch.zeros(0,    dtype=torch.long)
            
        
        valid = self._validate_sample(
            node_raw=node_raw,
            motion_raw=motion_raw,
            node_bbox=node_bbox,
            node_conf=node_conf,
            node_class_ids=node_class_ids,
            s_idx=s_idx,
            t_idx=t_idx,
            q_emb=q_emb,
            q_mask=q_mask
        )

        if not valid:
            raise ValueError(f"Invalid sample detected: {video_dir}, question ID {question['id']}, video name {question['video_name']}, node count {node_raw.shape[0]}, motion shape {motion_raw.shape}, question emb shape {q_emb.shape}, question mask shape {q_mask.shape}, s_idx shape {s_idx.shape}, t_idx shape {t_idx.shape}")
        
        # ── 6: pack into PyG Data ──────────────────────────────────
        data = OpenEndedDataMotion(
            node_raw          = torch.tensor(node_raw,  dtype=torch.float32),
            motion_feat       = torch.tensor(motion_raw, dtype=torch.float32),
            node_bbox         = torch.tensor(node_bbox, dtype=torch.float32),
            node_conf         = torch.tensor(node_conf, dtype=torch.float32),
            node_class        = node_class_ids,
            node_obj_ids_flat = torch.tensor(node_obj_ids_flat, dtype=torch.long),
            node_frame_index  = torch.tensor(node_frame_index,  dtype=torch.long),
            s_idx             = s_idx,
            s_attr            = s_attr,
            s_score           = s_score,
            t_idx             = t_idx,
            t_attr            = t_attr,
            # num_nodes         = torch.tensor(node_raw.shape[0], dtype=torch.long),
            num_nodes         = node_raw.shape[0],
            question_emb      = torch.tensor(q_emb, dtype=torch.float32), 
            question_mask     = torch.tensor(q_mask, dtype=torch.bool),
            qtype_idx         = torch.tensor([question["qtype_idx"]], dtype=torch.long),
            triplet_idxs        = torch.tensor([question["triplet_idxs"]], dtype=torch.long),
            triplet_mask      = torch.tensor([question["triplet_mask"]], dtype=torch.bool),
            answers           = torch.tensor(question["answer"], dtype=torch.long),
        )

        return data

import os
import glob
import json
import torch
import numpy as np
from torch_geometric.data import Data, Dataset

class MultiChoiceDataMotion(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key in ('s_idx', 't_idx'):
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ('s_idx', 't_idx'):
            return 1
        
        # Tell PyG to STACK these attributes into a new Batch dimension
        # motion_feat -> (B, T, Dim)
        # options_emb -> (B, option, max_len, Dim)
        if key in ('motion_feat', 'num_temporal_slots', 'question_emb', 'question_mask'):
            return None 
        if key in ('options_emb', 'option_mask'):
            return None
        
        
            
        return super().__cat_dim__(key, value, *args, **kwargs)


class STUDTrafficDatasetMotion(Dataset):
    def __init__(self, cfg, split="train"):
        """
        Args:
            cfg: configuration object
            split (str): "train", "val", or "test". Dictates which folder to read from.
        """
        self.split = split
        
        # ── 1. Determine root directory based on the exact split ──
        if split == "test": 
            self.root_dir = cfg.data.test_root
        elif split == "val":
            self.root_dir = cfg.data.val_root 
        else:
            self.root_dir = cfg.data.train_root 

        self.NUM_YOLO_CLASSES   = cfg.model.num_classes
        self.NUM_PREDICATES     = cfg.model.num_preds
        self.YOLO_CLASS_VOCAB   = YOLO_CLASS_VOCAB # Make sure this is imported/defined
        
        self.shared_video_files = [
            "node_raw.npy",
            "motion_raw.npy",
            "node_metadata.json",
            "node_proj.npy"
        ]
        
        self.qa_pairs = [] 
        self.invalid_dirs = set() 
        
        # ── 2. Scan directories directly for extracted questions ──
        print(f"Scanning {split.upper()} directory: {self.root_dir}...")
        video_folders = [f.path for f in os.scandir(self.root_dir) if f.is_dir()]
        
        for video_dir in video_folders:
            video_name = os.path.basename(video_dir)
            
            if not self._has_shared_files(video_dir):
                if video_dir not in self.invalid_dirs:
                    self.invalid_dirs.add(video_dir)
                    missing = self._get_missing_files(video_dir)
                    print(f"Warning: Skipping {video_name} - missing core files: {missing}")
                continue
                
            # Find all question JSONs in this video folder (e.g., q_1.json, q_2.json)
            q_jsons = glob.glob(os.path.join(video_dir, "q_*.json"))
            
            for q_json_path in q_jsons:
                # Extract the base name (e.g., "q_1")
                q_base = os.path.splitext(os.path.basename(q_json_path))[0]
                
                q_emb_path  = os.path.join(video_dir, f"{q_base}_emb.npz")
                opt_emb_path = os.path.join(video_dir, f"{q_base}_options_emb.npz")
                
                if os.path.exists(q_emb_path) and os.path.exists(opt_emb_path):
                    self.qa_pairs.append({
                        "video_dir": video_dir,
                        "q_json_path": q_json_path,
                        "q_emb_path": q_emb_path,
                        "opt_emb_path": opt_emb_path
                    })
                else:
                    print(f"Warning: Missing extracted embeddings for {q_base} in {video_name}")
        
        print(f"[{split.upper()}] Successfully loaded {len(self.qa_pairs)} valid question-answer pairs.")

    def _has_shared_files(self, video_dir: str) -> bool:
        return all(os.path.exists(os.path.join(video_dir, file)) for file in self.shared_video_files)

    def _get_missing_files(self, video_dir: str) -> list:
        return [file for file in self.shared_video_files if not os.path.exists(os.path.join(video_dir, file))]
   
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        return self._load_sample(self.qa_pairs[idx])
    
    def _load_sample(self, sample_info: dict): 
        video_dir = sample_info["video_dir"]
        q_json_path = sample_info["q_json_path"]
        q_emb_path = sample_info["q_emb_path"]
        opt_emb_path = sample_info["opt_emb_path"]  # <--- NEW

        # ── 1: load shared node features (Same for every question in this video) ──
        node_raw  = np.load(os.path.join(video_dir, "node_raw.npy")).astype(np.float32)
        node_proj = np.load(os.path.join(video_dir, "node_proj.npy")).astype(np.float32)
        motion_raw = np.load(os.path.join(video_dir, "motion_raw.npy")).astype(np.float32)
        
        MAX_T = 120 
        T = motion_raw.shape[0]
        valid_T = min(T, MAX_T)
        
        with open(os.path.join(video_dir, "node_metadata.json")) as f:
            meta = json.load(f)
        
        with open(q_json_path) as f:
            question = json.load(f)
        
        # q_emb   = np.load(q_emb_path).astype(np.float32)
        # opt_emb = np.load(opt_emb_path).astype(np.float32)
        data = np.load(q_emb_path)
        q_emb = data['emb'].astype(np.float32)
        q_mask = data['mask']  
        option_data = np.load(opt_emb_path)
        opt_emb    = option_data['emb'].astype(np.float32)
        opt_mask   = option_data['mask']
        
        padded_motion = np.zeros((MAX_T, motion_raw.shape[1]), dtype=np.float32)
        padded_motion[:valid_T] = motion_raw[:valid_T] # Copy the real frames in
        
        # ── 2: node identity ───────────────────────────────────────
        node_frame_index  = meta["node_frame_index"]          # [frame_id,  ...]
        node_kf_list_idx  = meta["node_keyframe_list_idx"]    # [kf_idx,    ...]
        node_is_keyframe  = meta["node_is_keyframe"]          # [bool,      ...]
        node_obj_ids_flat = meta["node_obj_ids_flat"]         # [tracked_id,...]
        # keyframe_indices  = meta["keyframe_indices"]          # [frame_id,  ...]
        num_keyframes     = int(meta["num_keyframes"])

        # ── 3: node attributes ─────────────────────────────────────
        node_bbox  = np.array(meta["node_bbox"],  dtype=np.float32)   # (N, 4) + [H, W]
        node_conf  = np.array(meta["node_conf"],  dtype=np.float32)   # (N,)
        
        # ── node class — fixed COCO vocab, no dynamic encoding needed ──
        node_class_ids = torch.tensor(
            [self.YOLO_CLASS_VOCAB.get(c, 0) for c in meta["node_class"]],
            dtype=torch.long
        )

        # ── 4: spatial edges ───────────────────────────────────────
        s_data  = meta.get("spatial_relations",  {})
        s_edges = s_data.get("edges",  [])
        s_attrs = s_data.get("attrs",  [])
        s_scores= s_data.get("scores", [])

        if len(s_edges) > 0:
            s_idx   = torch.tensor(s_edges,  dtype=torch.long).T    # (2, E_s)
            s_attr  = torch.tensor(s_attrs,  dtype=torch.long)      # (E_s,) int
            s_score = torch.tensor(s_scores, dtype=torch.float32)   # (E_s,)
        else:
            s_idx   = torch.zeros(2, 0, dtype=torch.long)
            s_attr  = torch.zeros(0,    dtype=torch.long)
            s_score = torch.zeros(0,    dtype=torch.float32)

        # ── 5: temporal edges ──────────────────────────────────────
        t_data  = meta.get("temporal_relations", {})
        t_edges = t_data.get("edges", [])
        t_attrs = t_data.get("attrs", [51])
        # t_attrs = torch.tensors(2)

        if len(t_edges) > 0:
            t_idx  = torch.tensor(t_edges, dtype=torch.long).T      # (2, E_t)
            # t_attrs may only have 1 entry — expand to match edge count
            if len(t_attrs) == 1:
                t_attrs = t_attrs * len(t_edges)
            t_attr = torch.tensor(t_attrs, dtype=torch.long)        # (E_t,)
        else:
            t_idx  = torch.zeros(2, 0, dtype=torch.long)
            t_attr = torch.zeros(0,    dtype=torch.long)
            
        
        
    
        
        # ── 6: pack into PyG Data ──────────────────────────────────
        data = MultiChoiceDataMotion(
            node_raw          = torch.tensor(node_raw,  dtype=torch.float32),
            node_proj         = torch.tensor(node_proj, dtype=torch.float32),
            motion_feat       = torch.tensor(padded_motion, dtype=torch.float32),
            num_temporal_slots= torch.tensor([valid_T], dtype=torch.long),
            node_bbox         = torch.tensor(node_bbox, dtype=torch.float32),
            node_conf         = torch.tensor(node_conf, dtype=torch.float32),
            node_class        = node_class_ids,
            node_obj_ids_flat = torch.tensor(node_obj_ids_flat, dtype=torch.long),
            node_frame_index  = torch.tensor(node_frame_index,  dtype=torch.long),
            node_kf_list_idx  = torch.tensor(node_kf_list_idx,  dtype=torch.long),
            node_is_keyframe  = torch.tensor(node_is_keyframe,  dtype=torch.bool),
            s_idx             = s_idx,
            s_attr            = s_attr,
            s_score           = s_score,
            t_idx             = t_idx,
            t_attr            = t_attr,
            num_keyframes     = torch.tensor(num_keyframes, dtype=torch.long),
            num_nodes         = torch.tensor(node_raw.shape[0], dtype=torch.long),
            question_emb      = torch.tensor(q_emb, dtype=torch.float32),  
            question_mask    = torch.tensor(q_mask, dtype=torch.bool) ,
            options_emb       = torch.tensor(opt_emb, dtype=torch.float32),
            option_mask       = torch.tensor(opt_mask, dtype=torch.bool),
            qtype_idx         = torch.tensor([question["qtype_idx"]], dtype=torch.long),
            triplet_idxs      = torch.tensor([question["triplet_idxs"]], dtype=torch.long),
            triplet_mask      = torch.tensor([question["triplet_mask"]], dtype=torch.bool),
            answers           = torch.tensor([question["answer"]], dtype=torch.long),
        )

        return data
    
    
class TGIFDatasetMotion(Dataset):
    def __init__(self, cfg, split="train", allowed_video_ids=None):
        """
        Args:
            cfg: configuration object
            split (str): "train", "val", or "test". Dictates which folder to read from.
        """
        self.split = split
        self.allowed_video_ids = allowed_video_ids
        
        # ── 1. Determine root directory based on the exact split ──
        if split == "test": 
            self.root_dir = cfg.data.test_root
        elif split == "val":
            self.root_dir = cfg.data.val_root 
        else:
            self.root_dir = cfg.data.train_root 

        self.NUM_YOLO_CLASSES   = cfg.model.num_classes
        self.NUM_PREDICATES     = cfg.model.num_preds
        self.YOLO_CLASS_VOCAB   = YOLO_CLASS_VOCAB 
        self.feature_layer      = cfg.data.feature_layer
        
        self.shared_video_files = [
            "node_layer1.npy",
            "node_layer2.npy",
            "node_layer3.npy",
            "node_layer4.npy",
            "motion_features.npy",    
            "node_metadata.json",
        
        ]
        
        
        self.qa_pairs = [] 
        self.invalid_dirs = set() 
        
        # ── 2. Scan directories directly for extracted questions ──
        print(f"Scanning {split.upper()} directory: {self.root_dir}...")
        video_folders = [f.path for f in os.scandir(self.root_dir) if f.is_dir()]
        
        for video_dir in video_folders:
            video_name = os.path.basename(video_dir)
            video_id = os.path.basename(video_dir)
            
            if (
                self.allowed_video_ids is not None
                and video_id not in self.allowed_video_ids
            ):
                continue
            
            # Remove empty directories immediately
            if self._is_directory_empty(video_dir):
                print(f"Removing empty directory: {video_name}")
                shutil.rmtree(video_dir)
                continue
            
            # # Remove directories missing core files
            # if not self._has_shared_files(video_dir):
            #     print(f"Removing incomplete directory: {video_name}")
            #     missing = self._get_missing_files(video_dir)
            #     print(f"  Missing: {missing}")
            #     shutil.rmtree(video_dir)
            #     continue
            
            if not self._has_shared_files(video_dir):
                if video_dir not in self.invalid_dirs:
                    self.invalid_dirs.add(video_dir)
                    missing = self._get_missing_files(video_dir)
                    shutil.rmtree(video_dir)
                    print(f"Warning: Skipping {video_name} - missing core files: {missing}")
                continue
                
            # Find all question JSONs in this video folder (e.g., q_1.json, q_2.json)
            q_jsons = glob.glob(os.path.join(video_dir, "q_*.json"))
            
            
            if not q_jsons:
                print(f"Removing directory with no questions: {video_name}")
                shutil.rmtree(video_dir)
                continue
            
            for q_json_path in q_jsons:
                # Extract the base name (e.g., "q_1")
                q_base = os.path.splitext(os.path.basename(q_json_path))[0]
                
                q_emb_path  = os.path.join(video_dir, f"{q_base}_emb.npz")
                opt_emb_path = os.path.join(video_dir, f"{q_base}_options_emb.npz")
                
                if os.path.exists(q_emb_path) and os.path.exists(opt_emb_path):
                    self.qa_pairs.append({
                        "video_id": video_id,
                        "video_dir": video_dir,
                        "q_json_path": q_json_path,
                        "q_emb_path": q_emb_path,
                        "opt_emb_path": opt_emb_path
                    })
                else:
                    print(f"Warning: Missing extracted embeddings for {q_base} in {video_name}")
        
        print(f"[{split.upper()}] Successfully loaded {len(self.qa_pairs)} valid question-answer pairs.")
       

    def _has_shared_files(self, video_dir: str) -> bool:
        return all(os.path.exists(os.path.join(video_dir, file)) for file in self.shared_video_files)

    def _get_missing_files(self, video_dir: str) -> list:
        return [file for file in self.shared_video_files if not os.path.exists(os.path.join(video_dir, file))]

    def __len__(self):
        return len(self.qa_pairs)
    
    # def __getitem__(self, idx):
    #     return self._load_sample(self.qa_pairs[idx])
    
    def __getitem__(self, idx):

        tries = 0

        while tries < 20:

            try:

                sample = self._load_sample(self.qa_pairs[idx])

                return sample

            except Exception as e:

                print(f"[Dataset Error] idx={idx}")
                print(e)

                idx = random.randint(0, len(self.qa_pairs) - 1)

                tries += 1

        raise RuntimeError("Too many corrupted samples.")

    def _is_directory_empty(self, video_dir):
        """Check if directory is empty or only contains empty directories"""
        try:
            return not any(os.scandir(video_dir))
        except:
            return True
    def _validate_sample(
        self,
        node_raw,
        motion_raw,
        opt_emb ,
        node_bbox,
        node_conf,
        node_class_ids,
        s_idx,
        t_idx,
        q_emb,
        q_mask
    ):

        # =====================================================
        # Basic node checks
        # =====================================================

        N = node_raw.shape[0]

        if N == 0:
            return False

        if torch.isnan(torch.tensor(node_raw)).any():
            return False

        if torch.isinf(torch.tensor(node_raw)).any():
            return False

        # =====================================================
        # Question checks
        # =====================================================

        if q_emb.shape[0] != q_mask.shape[0]:
            return False

        if q_emb.shape[0] == 0:
            return False

        # =====================================================
        # Motion checks
        # =====================================================

        if motion_raw.shape[0] == 0:
            return False
        if opt_emb.shape[0] == 0:
            return False

        # =====================================================
        # Spatial edge checks
        # =====================================================

        if s_idx.numel() > 0:

            if s_idx.dim() != 2 or s_idx.shape[0] != 2:
                return False

            if s_idx.max() >= N:
                return False

            if s_idx.min() < 0:
                return False

        # =====================================================
        # Temporal edge checks
        # =====================================================

        if t_idx.numel() > 0:

            if t_idx.dim() != 2 or t_idx.shape[0] != 2:
                return False

            if t_idx.max() >= N:
                return False

            if t_idx.min() < 0:
                return False

        # =====================================================
        # Class / bbox consistency
        # =====================================================

        if len(node_bbox) != N:
            return False

        if len(node_conf) != N:
            return False

        if len(node_class_ids) != N:
            return False

        return True
    
    def _load_sample(self, sample_info: dict): 
        video_dir = sample_info["video_dir"]
        q_json_path = sample_info["q_json_path"]
        q_emb_path = sample_info["q_emb_path"]
        opt_emb_path = sample_info["opt_emb_path"]  # <--- NEW

        # ── 1: load shared node features (Same for every question in this video) ──
        node_raw  = np.load(os.path.join(video_dir, f"{self.feature_layer}")).astype(np.float32)
        # node_proj = np.load(os.path.join(video_dir, "node_proj.npy")).astype(np.float32)
        motion_raw = np.load(os.path.join(video_dir, "motion_features.npy")).astype(np.float32)
        
        
        with open(os.path.join(video_dir, "node_metadata.json")) as f:
            meta = json.load(f)
        
        with open(q_json_path) as f:
            question = json.load(f)
        
        data = np.load(q_emb_path)
        q_emb = data['emb'].astype(np.float32)
        q_mask = data['mask']  
        option_data = np.load(opt_emb_path)
        opt_emb    = option_data['emb'].astype(np.float32)
        opt_mask   = option_data['mask']
        
        
        # ── 2: node identity ───────────────────────────────────────
        node_frame_index  = meta["node_frame_index"]     
        node_obj_ids_flat = meta["node_obj_ids_flat"]         


        # ── 3: node attributes ─────────────────────────────────────
        node_bbox  = np.array(meta["node_bbox"],  dtype=np.float32)   # (N, 4) + [H, W]
        node_conf  = np.array(meta["node_conf"],  dtype=np.float32)   # (N,)
        
        # ── node class — fixed COCO vocab, no dynamic encoding needed ──
        node_class_ids = torch.tensor(
            [self.YOLO_CLASS_VOCAB.get(c, 0) for c in meta["node_class"]],
            dtype=torch.long
        )

        # ── 4: spatial edges ───────────────────────────────────────
        s_data  = meta.get("spatial_relations",  {})
        s_edges = s_data.get("edges",  [])
        s_attrs = s_data.get("attrs",  [])
        s_scores= s_data.get("scores", [])

        if len(s_edges) > 0:
            s_idx   = torch.tensor(s_edges,  dtype=torch.long).T    # (2, E_s)
            s_attr  = torch.tensor(s_attrs,  dtype=torch.long)      # (E_s,) int
            s_score = torch.tensor(s_scores, dtype=torch.float32)   # (E_s,)
        else:
            s_idx   = torch.zeros(2, 0, dtype=torch.long)
            s_attr  = torch.zeros(0,    dtype=torch.long)
            s_score = torch.zeros(0,    dtype=torch.float32)

        # ── 5: temporal edges ──────────────────────────────────────
        t_data  = meta.get("temporal_relations", {})
        t_edges = t_data.get("edges", [])
        t_attrs = t_data.get("attrs", [51])
        # t_attrs = torch.tensors(2)

        if len(t_edges) > 0:
            t_idx  = torch.tensor(t_edges, dtype=torch.long).T      # (2, E_t)
            # t_attrs may only have 1 entry — expand to match edge count
            if len(t_attrs) == 1:
                t_attrs = t_attrs * len(t_edges)
            t_attr = torch.tensor(t_attrs, dtype=torch.long)        # (E_t,)
        else:
            t_idx  = torch.zeros(2, 0, dtype=torch.long)
            t_attr = torch.zeros(0,    dtype=torch.long)
            
        valid = self._validate_sample(
            node_raw=node_raw,
            motion_raw=motion_raw,
            opt_emb=opt_emb,
            node_bbox=node_bbox,
            node_conf=node_conf,
            node_class_ids=node_class_ids,
            s_idx=s_idx,
            t_idx=t_idx,
            q_emb=q_emb,
            q_mask=q_mask
        )
        
        if not valid:
            raise ValueError(f"Invalid sample detected: {video_dir}")
        
        # ── 6: pack into PyG Data ──────────────────────────────────
        data = MultiChoiceDataMotion(
            node_raw          = torch.tensor(node_raw,  dtype=torch.float32),
            motion_feat       = torch.tensor(motion_raw, dtype=torch.float32),
            node_bbox         = torch.tensor(node_bbox, dtype=torch.float32),
            node_conf         = torch.tensor(node_conf, dtype=torch.float32),
            node_class        = node_class_ids,
            node_obj_ids_flat = torch.tensor(node_obj_ids_flat, dtype=torch.long),
            node_frame_index  = torch.tensor(node_frame_index,  dtype=torch.long),
            s_idx             = s_idx,
            s_attr            = s_attr,
            s_score           = s_score,
            t_idx             = t_idx,
            t_attr            = t_attr,
            num_nodes         = torch.tensor(node_raw.shape[0], dtype=torch.long),
            question_emb      = torch.tensor(q_emb, dtype=torch.float32),  
            question_mask    = torch.tensor(q_mask, dtype=torch.bool) ,
            options_emb       = torch.tensor(opt_emb, dtype=torch.float32),
            option_mask       = torch.tensor(opt_mask, dtype=torch.bool),
            qtype_idx         = torch.tensor([question["qtype_idx"]], dtype=torch.long),
            triplet_idxs      = torch.tensor([question["triplet_idxs"]], dtype=torch.long),
            triplet_mask      = torch.tensor([question["triplet_mask"]], dtype=torch.bool),
            answers           = torch.tensor([question["answer"]], dtype=torch.long),
        )

        return data
        
    
    
    
    # @staticmethod
    # def _flatten(nested, dtype=None):
    #     flat = [x for sub in nested
    #             for x in (sub if isinstance(sub, list) else [sub])]
    #     return np.array(flat, dtype=dtype) if dtype else flat

    def get_vocab_sizes(self) -> dict:
        return {
            "num_classes":    self.NUM_YOLO_CLASSES,   # 80
            "num_predicates": self.NUM_PREDICATES,     # 52 (0-50 RelTR + 51 temporal)
        }


# ==================================================================
# Collate
# ==================================================================
def stgraph_collate(data_list):
    return PyGBatch.from_data_list(data_list)
 

def build_dataloader(cfg, split = "train", shuffle=True
) -> DataLoader:
    # if cfg.data.dataset == "STDTraffic":
    # dataset = STUDTrafficDatasetMotion(cfg, "train")
    dataset   = MSVDDatasetMotion(cfg, split)
    # dataset     = TGIFDatasetMotion(cfg, split)
    # else:
    #     print(f"Unsupported dataset: {cfg.dataset}")
    #     dataset = None

    # if dataset is None:
    #     raise ValueError(f"Unsupported dataset: {cfg.dataset}")

    loader  = DataLoader(
        dataset,
        batch_size   = cfg.train.batch_size,
        shuffle      = shuffle,
        num_workers  = cfg.data.num_workers,
        collate_fn   = stgraph_collate,
    )
    print(f"Dataset : {len(dataset)} videos")
    print(f"Loader  : {len(loader)} batches @ batch_size={cfg.train.batch_size}")

    return loader


    

# if __name__ == "__main__":
#     from data_class_open_ended import get_args
    
#     cfg = get_args
#     loader = build_dataloader(
#     cfg=cfg, 
# )

#     for batch in loader:
#         print(batch["question_emb"].shape)
#         print(batch["question_emb"])
#         break
        