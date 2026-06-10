
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_add
import torch.nn.functional as F
from .ST_block import SpatioTemporalBlock
from .node_attribute_encoder import NodeAttributeEncoder
import os
from languageEncoder.linguistic_prior import HierarchicalLinguisticPriorBank,ConceptNetExpectedPriorBank
from models.multiEncoder.cross_encoder_updated import GraphSTCausalMotion_Transformer, MultiChoiceClassifier, MultiChoices


class ConceptNetTripletBank(nn.Module):
    """
    O(1) Lookup Table for Precomputed ConceptNet Triplets.
    Outputs the candidate causal mechanisms (M) for Cross-Attention.
    """
    def __init__(self, triplet_pt_path="M_triplets.pt", max_triplets=6):
        super().__init__()
        self.max_triplets = max_triplets
        
        triplet_dict = torch.load(triplet_pt_path)
        sample_tensor = next(iter(triplet_dict.values()))
        dim = sample_tensor.shape[1]
        num_classes = 80  # Match your COCO/Visual Genome classes
        
        lookup_table = torch.zeros((num_classes, num_classes, max_triplets, dim))
        
        for (src_idx, dst_idx), embs in triplet_dict.items():
            num_avail = min(len(embs), max_triplets)
            lookup_table[src_idx, dst_idx, :num_avail] = embs[:num_avail]
            
        # Registered as a buffer so it moves to GPU but does not receive gradients
        self.register_buffer("lookup_table", lookup_table)

    def forward(self, src_cls_ids: torch.Tensor, dst_cls_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src_cls_ids: (E,) tensor of source node classes
            dst_cls_ids: (E,) tensor of target node classes
        Returns:
            (E, max_triplets, dim) tensor of retrieved textual affordances
        """
        return self.lookup_table[src_cls_ids, dst_cls_ids]

class STGraphTransformerNet(nn.Module):
    """
    PRODUCTION-READY: Complete causal video QA with all fixes.
    Now supports both free-form and multi-choice!
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dropout = cfg.model.dropout
        self.dim = cfg.model.dim
        self.raw_dim = cfg.model.raw_dim
        self.edge_dim = cfg.model.edge_dim
        self.final_drop = cfg.train.final_drop
        
        # ── Stage 1: Encoders
        self.node_encoder = NodeAttributeEncoder(
            self.raw_dim, self.dim, cfg.model.num_classes
        )
        self.edge_embed = nn.Embedding(cfg.model.num_preds, self.edge_dim)
        
        self.visual_question  = GraphSTCausalMotion_Transformer(cfg.model.motion_dim, self.dim, cfg.model.concept_dim, cfg.model.text_dim, cfg.model.dim, cfg.model.num_heads, cfg.model.num_layers, cfg.model.num_tokens)
        
   
        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(
                self.dim, cfg.model.num_heads, self.edge_dim, cfg.model.num_anchors
            )
            for _ in range(cfg.model.graph_layer)  
        ])
        self.node_feat_norm = nn.LayerNorm(self.dim)
        
        self.linguistic_prior_bank = HierarchicalLinguisticPriorBank(
            qtype_pt_path=cfg.data.qtype_pt_path, 
            qrole_pt_path=cfg.data.qrole_pt_path
        )
      
        
        self.conceptnet_prior = ConceptNetExpectedPriorBank(
            prior_pt_path=cfg.data.prior_pt_path,
        )
        
        self.conceptnet_triplets = ConceptNetTripletBank(cfg.data.triplet_pt_path)
        
        self.gate_graph = nn.Parameter(torch.zeros(self.dim))
        # self.final_mlp = MultiChoiceClassifier(self.dim, text_dim= self.cfg.model.text_dim ,num_heads=cfg.model.num_heads, dropout=0.1, final_dropout=self.final_drop)
        self.final_mlp = MultiChoices(self.dim)
        
        # ── Temperature & Gates
        self.logit_temp = nn.Parameter(
            torch.clamp(torch.tensor(1.0, dtype=torch.float32), min=0.1, max=2.0)
        )
    
    def compute_graph_relational(
            self,
            cls_id: torch.Tensor,
            batch_idx: torch.Tensor,
            B: int
        ) -> tuple:
            """
            Compute causal signals with proper error handling.
            """
            device = cls_id.device
            
            E_z_nodes = self.conceptnet_prior(cls_id)  # (N, concept_dim)
            E_z = global_mean_pool(E_z_nodes, batch_idx)  # (B, concept_dim)
            return E_z
    
    
    def _compute_triplet_mediator(
        self,
        s_idx,             # [2,E]
        cls_id,            # [N]
        batch_idx,         # [N]
        E_z,               # [B,d]
        B,
        max_triplets=6
        ):

        device = E_z.device
        d = E_z.size(-1)

        
        if s_idx.numel() == 0 or s_idx.size(1) == 0:

            K = self.conceptnet_triplets.max_triplets   # or fixed K

            pooled_triplets = E_z.unsqueeze(1).repeat(1, K, 1)

            triplet_mask = torch.ones(
                (B, K),
                dtype=torch.bool,
                device=device
            )

            return pooled_triplets, triplet_mask

       

        src, dst = s_idx[0], s_idx[1]

        edge_batch = batch_idx[src]

        

        # [E,K,d]
        E_T = self.conceptnet_triplets(
            cls_id[src],
            cls_id[dst]
        )

        E, K, d = E_T.shape

        

        E_T = E_T.reshape(E * K, d)

        edge_batch = edge_batch.repeat_interleave(K)

        

        valid_mask = (
            E_T.abs().sum(dim=-1) > 1e-6
        )

        E_T = E_T[valid_mask]

        edge_batch = edge_batch[valid_mask]

       

        if E_T.size(0) == 0:

            triplet_feat = E_z.unsqueeze(1)

            triplet_mask = torch.ones(
                (B, 1),
                dtype=torch.bool,
                device=device
            )

            return triplet_feat, triplet_mask

        

        pooled_triplets = []
        pooled_masks = []

        for b in range(B):

            t_b = E_T[edge_batch == b]

           
            
            

            if t_b.size(0) == 0:

                
                K = self.conceptnet_triplets.max_triplets

                pooled_triplets.append(
                    E_z[b].unsqueeze(0).repeat(K, 1)
                )

                pooled_masks.append(
                    torch.ones(K, dtype=torch.bool, device=device)
                )

                continue

            

            # simple norm-based importance
            scores = t_b.norm(dim=-1)

            Kb = min(max_triplets, t_b.size(0))

            topk_idx = scores.topk(Kb).indices

            selected = t_b[topk_idx]


            if Kb < max_triplets:

                pad = torch.zeros(
                    max_triplets - Kb,
                    d,
                    device=device,
                    dtype=t_b.dtype
                )

                selected = torch.cat(
                    [selected, pad],
                    dim=0
                )

            mask = torch.zeros(
                max_triplets,
                dtype=torch.bool,
                device=device
            )

            mask[:Kb] = True

            pooled_triplets.append(selected)
            pooled_masks.append(mask)
   
        pooled_triplets = torch.stack(pooled_triplets, dim=0)
        triplet_mask = torch.stack(pooled_masks, dim=0)

        return pooled_triplets, triplet_mask
    

    def forward(self, data) -> dict:
        """
        PRODUCTION-READY: Complete forward pass.
       
        """
        # ── Unpack
        node_raw = data["node_raw"]
        bbox = data["node_bbox"]
        cls_id = data["node_class"]
        conf = data["node_conf"]
        node_obj_ids_flat = data["node_obj_ids_flat"]
        batch_idx = data["batch"]
        
        motion_feat = data["motion_feat"]
        
        s_idx = data["s_idx"]
        s_attr_ids = data["s_attr"]
        t_idx = data["t_idx"]
        t_attr_ids = data["t_attr"]
        question_emb = data["question_emb"]
        qtype_idx = data["qtype_idx"].squeeze(-1)
        question_mask = data["question_mask"]
        options_emb = data["options_emb"]
        option_mask = data["option_mask"]
        
        B = question_emb.size(0)
      
        E_cl, ecl_mask = self.linguistic_prior_bank(
            qtype_idx,
            data["triplet_idxs"],
            data["triplet_mask"]
        )
        motion_mask = None
        
        # ── Node Encoding & ST-Graph
        h = self.node_encoder(
            node_raw, bbox, cls_id, conf,
        )  # (N, dim)
        
        if self.training:
            h = F.dropout(h, p=self.dropout, training=True)
        
        for block in self.blocks:
            h_residual = h
            h, s_attr, t_attr = block(
                h,
                s_idx, self.edge_embed(s_attr_ids),
                t_idx, self.edge_embed(t_attr_ids)
            )

            alpha = torch.sigmoid(self.gate_graph)  # [dim]

            h = h_residual + alpha * (h - h_residual)

        node_feat = h
        
        
       
        
        E_z = self._compute_causal_signals(
            cls_id=cls_id,
            batch_idx=batch_idx,
            B=B
        )
        
        casual_feature, triplet_mask = self._compute_triplet_mediator(
            s_idx=s_idx,
            cls_id=cls_id,
            batch_idx=batch_idx,
            E_z=E_z,
            B=B
        )   
        
        question_tokens, answer_tokens, ecl_question_tokens,feature_fused,answer_feature_fused,feature_causal_fused,causal_fused, question_masks, ecl_masks,answer_masks  = self.visual_question(
                node_feat = node_feat,
                motion_video_feat = motion_feat,
                casual_embedding = None,
                question_embedding = question_emb,
                e_cl_embedding = E_cl,
                answer_embedding = options_emb,
                motion_mask = None,
                triplet_mask = triplet_mask,
                question_mask = question_mask,
                ecl_mask = ecl_mask,
                answer_mask = option_mask,
                batch_idx = batch_idx )
     
        logits = self.final_mlp(question_tokens,answer_tokens,ecl_question_tokens,feature_fused, answer_feature_fused, feature_causal_fused, causal_fused,question_masks, ecl_masks, answer_masks )
        
        return {
            "causal_logits": logits,
        }
        

        
        
        
        
        
        
        
      
        