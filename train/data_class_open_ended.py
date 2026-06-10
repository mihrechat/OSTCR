
import argparse
import json
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

# ==================================================================
# 1. Backbone Registry
# ==================================================================
@dataclass
class BackboneConfig:
    name:     str
    raw_dim:  int
    proj_dim: int

BACKBONE_REGISTRY: Dict[str, BackboneConfig] = {
    "swin-b-layer1":   BackboneConfig("swin-B-layer1", raw_dim=128, proj_dim=128),
    "qwen2.5-vl-3b":   BackboneConfig("qwen2.5-vl-3b", raw_dim=1280, proj_dim=2048),
    "swin-b-layer1":   BackboneConfig("swin-B-layer1", raw_dim=128, proj_dim=128),
    "swin-b-layer2":   BackboneConfig("swin-B-layer2", raw_dim=256, proj_dim=256),
    "swin-b-layer3":   BackboneConfig("swin-B-layer3", raw_dim=512, proj_dim=128),
    "swin-b-layer4":   BackboneConfig("swin-B-layer4", raw_dim=1024, proj_dim=128),
        }

def get_backbone_config(name: str) -> BackboneConfig:
    key = name.lower().strip()
    if key not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone '{name}'. Available: {list(BACKBONE_REGISTRY.keys())}")
    return BACKBONE_REGISTRY[key]

# ==================================================================
# 2. Modular Sub-Configs
# ==================================================================
@dataclass
class SystemConfig:
    save_dir:    str  = '/root/autodl-tmp/CausalSTGNet/train/msvd_motion_configs/msvd_motion_layer3_configs/casual_1/layer2_feature/no_hstr'
    log_dir:     str  = '/root/autodl-tmp/CausalSTGNet/train/msvd_logs'
    seed:        int  = 42
    resume_from: Optional[str] = None
    use_amp:     bool = True

# @dataclass
# class DataConfig:
#     train_root:   str   = '/root/autodl-tmp/CausalSTGNet/datasets/open_ended_vqa/MSVD_vqa/train'
#     test_root:   str    =  '/root/autodl-tmp/CausalSTGNet/datasets/open_ended_vqa/MSVD_vqa/test'
#     val_root:    str    = '/root/autodl-tmp/CausalSTGNet/datasets/open_ended_vqa/MSVD_vqa/val'
#     dataset:     str    = "MSVD"
#     question_dir: str   = '/root/autodl-tmp/CausalSTGNet/datasets/open_ended_vqa/MSVD_vqa/prior_memory/language'
#     num_workers: int    = 4
    
@dataclass
class DataConfig:
    base_root:   str  = '/root/autodl-tmp/CausalSTGNet/datasets'
    task_type:   str  = "open_ended_vqa"
    dataset:     str  = "MSVD_2" 
    language:    str  = "prior_memory/language"
    visual:      str  = 'prior_memory/visual'
    num_workers: int  = 4
    val_split:   float  = 0.15
    feature_layer: str = "node_layer2.npy" 
    

    @property
    def train_root(self) -> str:
        if os.path.exists(os.path.join(self.base_root, self.task_type, self.dataset, "train")):
          return os.path.join(self.base_root, self.task_type, self.dataset, "train")
        else:
         raise FileNotFoundError("----------- Please make sure the <<train>> path exists ")

    @property
    def test_root(self) -> str:
       if os.path.exists(os.path.join(self.base_root, self.task_type, self.dataset, "test")):
         return os.path.join(self.base_root, self.task_type, self.dataset, "test")
       else:
        raise FileNotFoundError("----------- Please make sure the <<test>> path exists ")

    @property
    def val_root(self) -> str:
        if os.path.exists(os.path.join(self.base_root, self.task_type, self.dataset, "val")):
          return os.path.join(self.base_root, self.task_type, self.dataset, "val")
        else:
         raise FileNotFoundError("----------- Please make sure the <<val>> path exists ")
    @property
    def question_dir(self) -> str:
        if os.path.exists(os.path.join(self.base_root, self.task_type, self.dataset, self.language)):
           return os.path.join(self.base_root, self.task_type, self.dataset, self.language)
        else:
         raise FileNotFoundError("----------- Please make sure the <<language>> path exists ")
       
    @property
    def qtype_pt_path(self) -> str:
        if os.path.exists(os.path.join(self.base_root, self.task_type, self.dataset, self.language, "E_qtype_priors.pt")):
           return os.path.join(self.base_root, self.task_type, self.dataset, self.language, "E_qtype_priors.pt")
        else:
         raise FileNotFoundError("----------- Please make sure the <<qtype prior pt>> path exists ")
          
    @property
    def qrole_pt_path(self) -> str:
        if os.path.exists(os.path.join(self.base_root, self.task_type, self.dataset, self.language, "E_role_token_priors.pt")):
           return os.path.join(self.base_root, self.task_type, self.dataset, self.language, "E_role_token_priors.pt")
        else:
         raise FileNotFoundError("----------- Please make sure the <<E_role prior pt>> path exists ")
         
    @property
    def prior_pt_path(self) -> str:
       if os.path.exists(os.path.join(self.base_root, self.task_type, self.dataset, self.visual, "E_z_priors.pt")):
         return os.path.join(self.base_root, self.task_type, self.dataset, self.visual, "E_z_priors.pt")
       else:
         raise FileNotFoundError("----------- Please make sure the <<E_z prior pt>> path exists ")
      
      
    @property
    def triplet_pt_path(self) -> str:
       if os.path.exists(os.path.join(self.base_root, self.task_type, self.dataset, self.visual, "M_triplets.pt")):
         return os.path.join(self.base_root, self.task_type, self.dataset, self.visual, "M_triplets.pt")
       else:
         raise FileNotFoundError("----------- Please make sure the <<M_triplet pt>> path exists ")
    @property
    def num_qtypes(self):
        if "TGIF" in self.dataset:
            qtypes = 1
            return qtypes
        if "MSVD" in self.dataset:
            qtypes = 5
            return qtypes
        if "STUD" in self.dataset:
            qtypes = 6
            return qtypes

import os
from dataclasses import dataclass



@dataclass
class ModelConfig:
    backbone:           str   = 'swin-b-layer2'
    dim:                int   = 256
    motion_dim:         int   = 1536
    num_heads:          int   = 8
    edge_dim:           int   = 64
    num_layers:         int   = 4
    graph_layer:        int   = 2
    num_anchors:        int   = 8
    dropout:            float = 0.2
    num_classes:        int   = 80
    num_tokens:          int  = 48
    causal_num_tokens:   int  = 8
    num_preds:          int   = 52
    msvd_vocab_size:    int   = 945
    
    # Causal & Text Params
    text_dim:           int   = 1024
    concept_dim:        int   = 1024
    mediator_dim:       int   = 512
    num_prototypes:     int   = 32
    prototype_momentum: float = 0.998

    
    #Prior paths
    # prior_pt_path:      str   = '/root/autodl-tmp/CausalSTGNet/datasets/open_ended_vqa/MSVD/prior_memory/visual/E_z_priors.pt'
    # triplet_pt_path:    str   = '/root/autodl-tmp/CausalSTGNet/datasets/open_ended_vqa/MSVD/prior_memory/visual/M_triplets.pt'
    # qtype_pt_path:      str   = '/root/autodl-tmp/CausalSTGNet/datasets/open_ended_vqa/MSVD/prior_memory/language/E_qtype_priors.pt'
    # qrole_pt_path:      str   = '/root/autodl-tmp/CausalSTGNet/datasets/open_ended_vqa/MSVD/prior_memory/language/E_role_token_priors.pt'

    # Auto-resolved (Do not set via CLI)
    raw_dim:            int   = field(init=False)
    proj_dim:           int   = field(init=False)
    
    def __post_init__(self):
        bb = get_backbone_config(self.backbone)
        self.raw_dim  = bb.raw_dim
        self.proj_dim = bb.proj_dim

@dataclass
class TrainConfig:
    epochs:           int   = 30
    batch_size:       int   = 32
    grad_accum_steps: int   = 1
    max_grad_norm:    float = 1.0
    lr:               float = 1e-4
    encoder_lr:        float = 5e-5
    fusion_lr:         float = 2e-4
    classifier_lr:     float = 2e-4
    no_decay_lr:       float = 2e-4
    weight_decay:     float = 1e-4
    q_drop:           float = 0.2
    final_drop:       float = 0.2
    h_drop:           float = 0.2
    m_drop:           float = 0.2
    v_drop:           float = 0.2
    c_drop:           float = 0.2
    m_gated:          float = 0.9
    ecl_drop:         float = 0.2
    q_cross:          bool = False
    q_deconf:         bool = True
    
    
    
    # Loss Weights
    lambda_kl:        float = 1.0
    lambda_mse:       float = 1.0
    lambda_vqa:       float = 1.0 
    lambda_entropy:   float = 0.01
    lambda_gate:      float = 0.005
    aux_loss:         float  = 0.3
    tau:              float = 0.1
    fallback_temp:    float = 0.07
    losses: list = field(default_factory=lambda: ["vqa"])
    
    # Early Stopping
    patience:         int   = 2
    warmup:           float = 0.05
    warmup_epochs:    int = 2
    # early_stop_patience: int = 6
    min_delta:        float = 1e-5
    div_factor:       float = 0.7
    save_every:       int   = 10


# ==================================================================
# 3. Master Experiment Config
# ==================================================================
@dataclass
class ExperimentConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    data:   DataConfig   = field(default_factory=DataConfig)
    model:  ModelConfig  = field(default_factory=ModelConfig)
    train:  TrainConfig  = field(default_factory=TrainConfig)

    @classmethod
    def from_json(cls, path: str) -> "ExperimentConfig":
        """Loads heavily nested or flat JSON dynamically."""
        with open(path) as f:
            raw_data = json.load(f)
            
        cfg = cls()
        # Parse nested dicts directly to sub-configs
        for sub_name in ["system", "data", "model", "train"]:
            sub_config = getattr(cfg, sub_name)
            sub_data = raw_data.get(sub_name, {})
            
            valid_keys = {f.name for f in dataclasses.fields(sub_config) if f.init}
            for k, v in sub_data.items():
                if k in valid_keys:
                    setattr(sub_config, k, v)
                    
        cfg.model.__post_init__() # Resolve backbone dims
        return cfg

    def save_json(self, path: str):
        """Saves a beautiful nested JSON structure."""
        out = {}
        for sub_name in ["system", "data", "model", "train"]:
            sub_config = getattr(self, sub_name)
            d = {k: v for k, v in sub_config.__dict__.items() if k not in ("raw_dim", "proj_dim")}
            out[sub_name] = d
            
        with open(path, "w") as f:
            json.dump(out, f, indent=4)

# ==================================================================
# 4. Automated CLI Parser (Zero Boilerplate)
# ==================================================================
def str2bool(v):
    if isinstance(v, bool): return v
    return v.lower() in ('yes', 'true', 't', '1')

def get_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Train STGraph Causal Model")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config file")

    # Dynamically inject all dataclass fields into ArgParse!
    config_classes = [SystemConfig, DataConfig, ModelConfig, TrainConfig]
    
    for cls_type in config_classes:
        group = parser.add_argument_group(cls_type.__name__)
        for f in dataclasses.fields(cls_type):
            if f.init: # Skip auto-resolved fields like raw_dim
                arg_type = type(f.default) if f.default is not None else str
                _type = str2bool if arg_type is bool else arg_type
                group.add_argument(f"--{f.name}", type=_type, default=None, help=f"Override {f.name}")

    args, unknown = parser.parse_known_args()

    # 1. Start from pure defaults or JSON
    cfg = ExperimentConfig.from_json(args.config) if args.config else ExperimentConfig()

    # 2. Apply ONLY the CLI overrides dynamically
    cli_overrides = {k: v for k, v in vars(args).items() if v is not None and k != "config"}

    for sub_name in ["system", "data", "model", "train"]:
        sub_config = getattr(cfg, sub_name)
        valid_keys = {f.name for f in dataclasses.fields(sub_config)}
        for k, v in cli_overrides.items():
            if k in valid_keys:
                setattr(sub_config, k, v)

    # 3. Finalize
    cfg.model.__post_init__()
    return cfg

