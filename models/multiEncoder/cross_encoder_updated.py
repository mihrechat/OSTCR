import torch
import torch.nn as nn
from torch_scatter import scatter_add
from torch_geometric.utils import softmax as pyg_softmax
import torch.nn.functional as F
import math
from models.languageEncoder.linguistic_prior import HybridLanguageEncoder



class Question_guided_feature_aggregation(nn.Module):

    def __init__(self, module_dim):

        super().__init__()

        self.question_proj = nn.Linear(
            module_dim,
            module_dim,
            bias=False
        )

        self.visual_proj = nn.Linear(
            module_dim,
            module_dim,
            bias=False
        )

        self.cat = nn.Linear(
            module_dim * 2,
            module_dim
        )

        self.attn = nn.Linear(
            module_dim,
            1
        )

        self.activation = nn.ELU()

        self.dropout = nn.Dropout(0.15)

    def forward(
        self,
        question,
        visual_feat,
    ):
        visual_feat = self.dropout(
            visual_feat
        )

        quest_proj = self.question_proj(
            question
        )  
        vis_proj = self.visual_proj(
            visual_feat
        )  

        v_q = torch.cat([
            vis_proj,
            quest_proj * vis_proj
        ], dim=-1)

        visual_question = self.cat(v_q)

        visual_question = self.activation(v_q)

        attn = self.attn(visual_question)

        attn = F.softmax(
            attn,
            dim=1
        )

        visual_aggregated = attn * visual_feat

        return visual_aggregated
    
class OpenEndedClassification(nn.Module):

    def __init__(self, module_dim, num_answers=1000):

        super().__init__()

        self.classifier2 = nn.Sequential(
            nn.Dropout(0.15),

            nn.Linear(module_dim * 2, module_dim),

            nn.ELU(),

            nn.BatchNorm1d(module_dim),

            nn.Dropout(0.15),

            nn.Linear(module_dim, num_answers)
        )
        
        self.classifier3 = nn.Sequential(
            nn.Dropout(0.15),

            nn.Linear(module_dim * 3, module_dim),

            nn.ELU(),

            nn.BatchNorm1d(module_dim),

            nn.Dropout(0.15),

            nn.Linear(module_dim, num_answers)
        )
        
        self.classifier4 = nn.Sequential(
            nn.Dropout(0.15),

            nn.Linear(module_dim * 4, module_dim),

            nn.ELU(),

            nn.BatchNorm1d(module_dim),

            nn.Dropout(0.15),

            nn.Linear(module_dim, num_answers)
        )
        
        self.classifier5 = nn.Sequential(
            nn.Dropout(0.15),

            nn.Linear(module_dim * 5, module_dim),

            nn.ELU(),

            nn.BatchNorm1d(module_dim),

            nn.Dropout(0.15),

            nn.Linear(module_dim, num_answers)
        )

        for m in self.modules():

            if isinstance(m, nn.Linear):

                nn.init.xavier_normal_(m.weight)

                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):

                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        question_embedding,
        ecl_embedding,
        visual_embedding,
        visual_causal_embedding,
        causal_embedding,
        question_mask,
        ecl_mask
    ):

        # -----------------------------------------------------
        # Visual global pooling
        # -----------------------------------------------------
        
        visual_causal_global = None
        causal_global = None
        ecl_masks = None
        ecl_global = None
        out = None
        

        visual_global = visual_embedding.sum(dim=1)
        if visual_causal_embedding is not None:
            visual_causal_global = visual_causal_embedding.sum(dim=1)
        if causal_embedding is not None:
            causal_global = causal_embedding.sum(dim=1)
        

        # -----------------------------------------------------
        # Masked question pooling
        # -----------------------------------------------------

        question_mask = question_mask.float()
        if ecl_mask is not None:
            ecl_masks      =  ecl_mask.float()

        question_global = (
            question_embedding *
            question_mask.unsqueeze(-1)
        ).sum(dim=1)

        question_global = (
            question_global /
            question_mask.sum(
                dim=1,
                keepdim=True
            ).clamp(min=1.0)
        )
        
        if ecl_embedding is not None:
            ecl_global = (
                ecl_embedding * ecl_masks.unsqueeze(-1)
            ).sum(dim=1)
            ecl_global = (ecl_global / ecl_masks.sum(dim=1, keepdim=True).clamp(min=1.0))

        # -----------------------------------------------------
        # Final representation
        # -----------------------------------------------------

            
        if  visual_causal_global is not None and causal_global is not None: # use 5

            out = torch.cat([
                visual_global,
                question_global,
                visual_causal_global,
                ecl_global,
                causal_global
            ], dim=-1)
        if ecl_global is None and causal_global is not None and visual_causal_global is None: #use 3
            
            out = torch.cat([
                visual_global,
                question_global,
                causal_global
                
            ], dim=-1)
        if visual_causal_global is not None and causal_global is None:  
            out = torch.cat([
                visual_global,
                question_global,
                visual_causal_global,
                ecl_global
            ], dim=-1)
        if visual_causal_global is None and causal_global is None: 
            out = torch.cat([
                visual_global,
                question_global
            ], dim=-1)
            
        if out.size(1) == 5 * question_global.size(-1):
            logits = self.classifier5(out)
        elif out.size(1) == 4 * question_global.size(-1):
            logits = self.classifier4(out)
        elif out.size(1) == 3 * question_global.size(-1):
            logits = self.classifier3(out)
        else:
                logits = self.classifier2(out)

        return logits
    
    
    
class MultiChoices(nn.Module):
    def __init__(self, model_dim, drorate=0.0, activation='elu'):
        super(MultiChoices, self).__init__()
        if activation=='relu':
            self.activ=nn.ReLU()
        if activation=='prelu':
            self.activ=nn.PReLU()
        if activation=='elu':
            self.activ=nn.ELU()
        if activation=='gelu':
            self.activ=nn.GELU()

        # self.question_proj = nn.Linear(2*model_dim, model_dim)
        # self.ans_candidates_proj = nn.Linear(2*model_dim, model_dim)
        self.classifier4 = nn.Sequential(nn.Dropout(drorate),
                                        nn.Linear(model_dim * 4, model_dim),
                                        self.activ,
                                        nn.BatchNorm1d(model_dim),
                                        nn.Dropout(drorate),
                                        nn.Linear(model_dim, 1))
        self.classifier6 = nn.Sequential(nn.Dropout(drorate),
                                        nn.Linear(model_dim * 6, model_dim),
                                        self.activ,
                                        nn.BatchNorm1d(model_dim),
                                        nn.Dropout(drorate),
                                        nn.Linear(model_dim, 1))
        self.classifier7 = nn.Sequential(nn.Dropout(drorate),
                                        nn.Linear(model_dim * 7, model_dim),
                                        self.activ,
                                        nn.BatchNorm1d(model_dim),
                                        nn.Dropout(drorate),
                                        nn.Linear(model_dim, 1))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        ##question based######
        question_embedding, #B, L,dim
        answer_embedding,   #Bxc L, dim
        ecl_embedding,
        #visual based########
        visual_embedding,   #B, N, dim
        visual_answer_embedding, #BxC, L, dim
        visual_causal_embedding, #B, C, dim
        causal_embedding,
        question_mask,
        ecl_mask,
        answer_mask
        ):
        
        BC = answer_embedding.size(0)
        B = question_embedding.size(0)
        C = BC // B
        
        causal_global = None
        visual_causal_global = None
        ecl_masks = None
        ecl_global = None
        causal_expand = None
        ecl_expand = None
        
        question_mask = question_mask.float()
    
        if ecl_mask is not None:
            ecl_masks      = ecl_mask.float()
        answer_mask_expanded = answer_mask.float()
        
        # -----------------------------------------------------
        # Visual global pooling
        # -----------------------------------------------------

        visual_global = visual_embedding.sum(dim=1) 
        if visual_causal_embedding is not None:
            visual_causal_global = visual_causal_embedding.sum(dim=1)
        if causal_embedding is not None: 
            causal_global = causal_embedding.sum(dim=1)
        visual_answer_global = visual_answer_embedding.sum(dim=1)
        
        
        # -----------------------------------------------------
        # Masked question pooling
        # -----------------------------------------------------
        
        question_global = (
            question_embedding *
            question_mask.unsqueeze(-1)
        ).sum(dim=1)

        question_global = (
            question_global /
            question_mask.sum(
                dim=1,
                keepdim=True
            ).clamp(min=1.0)
        )
        
        if ecl_embedding is not None:
            ecl_global = (
                ecl_embedding * ecl_masks.unsqueeze(-1)
            ).sum(dim=1)
            ecl_global = (ecl_global / ecl_masks.sum(dim=1, keepdim=True).clamp(min=1.0))
        
        

        answer_visual_global = (
            answer_embedding *
            answer_mask_expanded.unsqueeze(-1)
        ).sum(dim=1)

        answer_visual_global = (
            answer_visual_global /
            answer_mask_expanded.sum(
                dim=1,
                keepdim=True
            ).clamp(min=1.0)
        )
        
        question_expand   = question_global.unsqueeze(1).expand(B,C,question_global.size(-1)).contiguous().view(B * C,question_global.size(-1))
        if ecl_embedding is not None:  
            ecl_expand        = ecl_global.unsqueeze(1).expand(B,C,ecl_global.size(-1)).contiguous().view(B * C,ecl_global.size(-1))
        visual_expand     = visual_global.unsqueeze(1).expand(B,C,visual_global.size(-1) ).contiguous().view(B*C, visual_global.size(-1))
        if visual_causal_embedding is not None:
            visual_causal_expand  =  visual_causal_global.unsqueeze(1).expand(B,C,visual_causal_global.size(-1) ).contiguous().view(B*C, visual_causal_global.size(-1))
        if causal_global is not None:
            causal_expand        = causal_global.unsqueeze(1).expand(B,C,causal_global.size(-1) ).contiguous().view(B*C, causal_global.size(-1))
        

        
        
        
        if (
            ecl_expand is not None
            and visual_causal_expand is not None
            and causal_expand is not None
        ):

            out = torch.cat([
                visual_expand,
                question_expand,
                visual_causal_expand,
                ecl_expand,
                causal_expand,
                visual_answer_global,
                answer_visual_global
            ], dim=-1)

        elif (
            ecl_expand is not None
            and visual_causal_expand is not None and causal_expand is None
        ):

            out = torch.cat([
                visual_expand,
                question_expand,
                visual_causal_expand,
                ecl_expand,
                visual_answer_global,
                answer_visual_global
            ], dim=-1)

        else:

            out = torch.cat([
                visual_expand,
                question_expand,
                visual_answer_global,
                answer_visual_global
            ], dim=-1)


        if out.size(1) == 7 * question_global.size(-1):

            out = self.classifier7(out)

        elif out.size(1) == 6 * question_global.size(-1):

            out = self.classifier6(out)

        else:

            out = self.classifier4(out)
        logits = out.view(B, C)
        return logits

    


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        dropout=0.1,
        max_len=5000
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # [1, max_len, d]
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [B, L, d]
        """

        L = x.size(1)

        x = x + self.pe[:, :L]

        return self.dropout(x)
    

class PositionalEncodingLearned1D(nn.Module):

    def __init__(
        self,
        d_model,
        dropout=0.1,
        max_len=5000
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.pos_embed = nn.Embedding(
            max_len,
            d_model
        )

        nn.init.trunc_normal_(
            self.pos_embed.weight,
            std=0.02
        )

    def forward(self, x):
        """
        x: [B, L, d]
        """

        B, L, d = x.shape

        idx = torch.arange(
            L,
            device=x.device
        )

        pos = self.pos_embed(idx)  # [L,d]

        x = x + pos.unsqueeze(0)

        return self.dropout(x)
    
    
class TransformerEncoderLayer_QKV(nn.Module):
    """
    Clean batch-first Transformer layer for:
    - self attention
    - cross attention
    - graph-text fusion

    Input:
        x_q : [B, Lq, d]
        x_k : [B, Lk, d] or None

    """

    def __init__(
        self,
        embed_dim,
        num_heads=8,
        attn_dropout=0.1,
        res_dropout=0.1,
        activ_dropout=0.1,
        activation='gelu',
        ff_mult=4
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.res_dropout = res_dropout
        self.activ_dropout = activ_dropout

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True
        )

        hidden_dim = ff_mult * embed_dim

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

        act_map = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'elu': nn.ELU(),
            'prelu': nn.PReLU()
        }

        self.activ = act_map.get(activation, nn.GELU())
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout_attn = nn.Dropout(res_dropout)
        self.dropout_ffn = nn.Dropout(res_dropout)

        self._reset_parameters()

    def _reset_parameters(self):

        for m in self.modules():

            if isinstance(m, nn.Linear):

                nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

    def forward(
        self,
        x_q,
        x_k=None,
        q_mask=None,
        kv_mask=None,
        return_attn=False
    ):

        if x_k is None:
            x_k = x_q
            kv_mask = q_mask
        
        q = self.norm1(x_q)
        k = self.norm1(x_k)
        
        attn_out, attn_weights = self.attn(
            query=q,
            key=k,
            value=k,

            # MultiheadAttention expects:
            # True = IGNORE
            key_padding_mask=None if kv_mask is None else ~kv_mask,

            need_weights=return_attn
        )

        x_q = x_q + self.dropout_attn(attn_out)

        x = self.norm2(x_q)

        x = self.fc1(x)
        x = self.activ(x)

        x = F.dropout(
            x,
            p=self.activ_dropout,
            training=self.training
        )

        x = self.fc2(x)

        x_q = x_q + self.dropout_ffn(x)
        if q_mask is not None:
            x_q = x_q * q_mask.unsqueeze(-1)
        if return_attn:
            return x_q, attn_weights

        return x_q


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        embed_dim,
        pos_flag='learned',
        pos_dropout=0.1,
        num_heads=8,
        attn_dropout=0.1,
        res_dropout=0.1,
        activ_dropout=0.1,
        activation='gelu',
        num_layers=4,
        ff_mult=4
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.pos_flag = pos_flag
        if pos_flag == 'sincos':

            self.embed_scale = math.sqrt(embed_dim)

            self.pos_encoder = PositionalEncoding(
                embed_dim,
                pos_dropout
            )

        elif pos_flag == 'learned':

            self.embed_scale = 1.0

            self.pos_encoder = PositionalEncodingLearned1D(
                embed_dim,
                pos_dropout
            )

        else:

            self.embed_scale = 1.0
            self.pos_encoder = nn.Identity()

        self.layers = nn.ModuleList([

            TransformerEncoderLayer_QKV(
                embed_dim=embed_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                res_dropout=res_dropout,
                activ_dropout=activ_dropout,
                activation=activation,
                ff_mult=ff_mult
            )

            for _ in range(num_layers)

        ])

        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x_q,
        x_k=None,
        q_mask=None,
        kv_mask=None,
        return_all_layers=False,
        return_attn=False
    ):
      
        
        x_q = self.pos_encoder(
            self.embed_scale * x_q
        )

        if x_k is not None:

            x_k = self.pos_encoder(
                self.embed_scale * x_k
            )

        intermediates = []
        attentions = []

        for layer in self.layers:

            if return_attn:

                x_q, attn = layer(
                    x_q,
                    x_k,
                    q_mask=q_mask,
                    kv_mask=kv_mask,
                    return_attn=True
                )

                attentions.append(attn)

            else:

                x_q = layer(
                    x_q,
                    x_k,
                    q_mask=q_mask,
                    kv_mask=kv_mask
                )

            if return_all_layers:
                intermediates.append(x_q)

        x_q = self.final_norm(x_q)

        outputs = [x_q]

        if return_all_layers:
            outputs.append(intermediates)

        if return_attn:
            outputs.append(attentions)

        if len(outputs) == 1:
            return outputs[0]

        return tuple(outputs)
    
    
class OSTCRModule_Transformer(nn.Module):
    def __init__(self, motion_dim, node_dim, concept_dim, text_dim, model_dim, num_heads, num_layers, num_tokens, causal_num_tokens):
        super(OSTCRModule_Transformer, self).__init__()
        self.motion_dim = motion_dim
        self.node_dim = node_dim
        self.concept_dim = concept_dim
        self.text_dim = text_dim
        self.model_dim = model_dim
        self.num_tokens = num_tokens
        self.causal_num_tokens = causal_num_tokens
        

        self.motion_proj = nn.Sequential(
                                                nn.Linear(self.motion_dim, model_dim),
                                                nn.GELU(),
                                                nn.Dropout(p=0.1),                                             
                                                    )

        self.object_feat_proj = nn.Sequential(
                                                nn.Linear(self.node_dim, model_dim),
                                                nn.GELU(),
                                                nn.Dropout(p=0.1),                                             
                                                    )
        self.causal_feat_proj = nn.Sequential(
                                                nn.Linear(self.concept_dim, model_dim),
                                                nn.GELU(),
                                                nn.Dropout(p=0.1),                                             
                                                    )


        # =========================================================
        # Branch dropout
        # =========================================================

        self.branch_drop_prob = 0.10
        
        self.language_encoder = HybridLanguageEncoder(in_dim=self.text_dim, model_dim=model_dim,num_heads=num_heads,num_transformer_layers=2, gru_layers=1,dropout=0.1)
        self.QNodeTransformer = TransformerEncoder(embed_dim=model_dim, pos_flag='None',pos_dropout=0.1,num_heads=num_heads,attn_dropout=0.1,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=num_layers)
        self.ANodeTransformer = TransformerEncoder(embed_dim=model_dim, pos_flag='None',pos_dropout=0.1,num_heads=num_heads,attn_dropout=0.1,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=num_layers)
        self.QMotionTransformer =  TransformerEncoder(embed_dim=model_dim, pos_flag='sincos',pos_dropout=0.1,num_heads=num_heads,attn_dropout=0.1,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=num_layers)  
        self.AMotionTransformer =  TransformerEncoder(embed_dim=model_dim, pos_flag='sincos',pos_dropout=0.1,num_heads=num_heads,attn_dropout=0.1,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=num_layers)  
        self.QCausalTransformer = TransformerEncoder(embed_dim=model_dim, pos_flag='None',pos_dropout=0.1,num_heads=num_heads,attn_dropout=0.1,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=num_layers)     
        self.MotionReasoningTransformer = TransformerEncoder(embed_dim=model_dim, pos_flag='sincos',pos_dropout=0.1,num_heads=num_heads,attn_dropout=0.1,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=num_layers)    
        self.NodeReasoningTransformer = TransformerEncoder(embed_dim=model_dim, pos_flag='None',pos_dropout=0.1,num_heads=num_heads,attn_dropout=0.1,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=num_layers)   
        self.CausalReasoningTransformer = TransformerEncoder(embed_dim=model_dim, pos_flag='learned',pos_dropout=0.1,num_heads=num_heads,attn_dropout=0.1,res_dropout=0.1,activ_dropout=0.1,activation='gelu',num_layers=num_layers)   
        
       
        # self.branch_pool = AttentionPool(dim=model_dim)
        self.graph_pool  =  GraphTokenPooling(model_dim, self.num_tokens)
        self.causal_graph_pool = GraphTokenPooling(model_dim, self.causal_num_tokens)
        self.feature_aggregate = Question_guided_feature_aggregation(model_dim)
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_normal_(m.weight)
        #         nn.init.constant_(m.bias, 0)
                
    def forward(
        self,
        node_feat,
        motion_video_feat,
        casual_embedding,
        question_embedding,
        e_cl_embedding,
        answer_embedding,
        motion_mask,
        question_mask,
        ecl_mask,
        answer_mask,
        batch_idx,
        triplet_mask,
        triplet_batch_idx
    ):
        """
        Returns:
            fused:               [B, M, d]
            fused_causal:        [B, M2, d] or None
            fused_mask:          [B, M]
            fused_causal_mask:   [B, M2] or None
        """

        # =========================================================
        # Language encoding
        # =========================================================
        
        
        
        question_output = self.language_encoder(
            question_embedding,
            question_mask
        )
        

        question = question_output["token_features"]         # [B,L,d]

        question_global = question_output["global_question"] # [B,d]
        question_global = question_global.unsqueeze(1) # [B,1,d]
        
        if answer_embedding is not None:
            B, C, L, d = answer_embedding.shape

            answer_emb = answer_embedding.view(B * C, L, d)
            answer_mask_flat = answer_mask.view(B * C, L)
            answer_output = self.language_encoder(
                answer_emb,
                answer_mask_flat
            )
            answer = answer_output["token_features"]         # [B*C,L,d]
            answer_global = answer_output["global_question"] # [B*C,d]
            answer_global = answer_global.unsqueeze(1) # [B*C,1,d]
        
        B = question.size(0)
        if motion_video_feat is not None:
            motion_feature = self.motion_proj(
                motion_video_feat
            )


        feature_branches         = []
        ecl_question_branches    = []
        question_branches        = []
        causal_branches          = []
        feature_causal_branches  = []
        answer_feature_branches  = []
        answer_branches          = []
        ecl_question_visual = None
        causal_fused        = None
        feature_causal_fused = None
        ecl_masks       = None
        triplet_masks = None
        
        # =========================================================
        # Motion branch
        # =========================================================

        if motion_video_feat is not None:

            # -----------------------------
            # Question-guided motion
            # -----------------------------
            
            question_motion = self.QMotionTransformer(
                question,
                motion_feature,
                q_mask=None,
                kv_mask=motion_mask
            )  
            
            question_branches.append(
                question_motion
            )
            
            motion_reasoned = self.MotionReasoningTransformer(
            motion_feature,
            question_motion,
            q_mask=motion_mask,
            kv_mask=question_mask
            )

            feature_branches.append(
                motion_reasoned
            )
            
            if answer_embedding is not None:
                
                motion_feature_expand = motion_feature.unsqueeze(1).expand(
                    B,
                    C,
                    motion_feature.size(1),
                    motion_feature.size(2)
                ).contiguous().view(
                    B * C,
                    motion_feature.size(1),
                    motion_feature.size(2)
                )
                answer_motion = self.AMotionTransformer(
                    answer,
                    motion_feature_expand,
                    q_mask=answer_mask_flat,
                    kv_mask=None
                )
                answer_branches.append(
                    answer_motion
                )
                
                answer_motion_reason = self.MotionReasoningTransformer(
                    motion_feature_expand,
                    answer_motion,
                    q_mask=None,
                    kv_mask=answer_mask_flat
                    )
                
                answer_feature_branches.append(
                    answer_motion_reason)
                
        # =========================================================
        # Object branch
        # =========================================================

        if node_feat is not None:
    
            visual_node_feature = self.object_feat_proj(
                node_feat
            )
           

            visual_node_feature, node_mask = self.graph_pool(
                visual_node_feature,
                batch_idx,
                B
            )
           
            question_node = self.QNodeTransformer(
                question,
                visual_node_feature,
                q_mask=question_mask,
                kv_mask=node_mask
            )
            question_branches.append(
                question_node
            )
           
          
            node_reasoned = self.NodeReasoningTransformer(
                visual_node_feature,
                question_node,
                q_mask=node_mask,
                kv_mask=question_mask
            )
            
            feature_branches.append(
                node_reasoned
            )
            
            if answer_embedding is not None:
                
                visual_node_feature_expand = visual_node_feature.unsqueeze(1).expand(
                    B,
                    C,
                    visual_node_feature.size(1),
                    visual_node_feature.size(2)
                ).contiguous().view(
                    B * C,
                    visual_node_feature.size(1),
                    visual_node_feature.size(2)
                )
                answer_node = self.ANodeTransformer(
                    answer,
                    visual_node_feature_expand,
                    q_mask=answer_mask_flat,
                    kv_mask=None
                )
                answer_branches.append(
                    answer_node
                )
                
                answer_node_reason = self.NodeReasoningTransformer(
                    visual_node_feature_expand,
                    answer_node,
                    q_mask=None,
                    kv_mask=answer_mask_flat
                    )
                
                answer_feature_branches.append(
                    answer_node_reason)

        # =========================================================
        # Causal branch
        # =========================================================

        if casual_embedding is not None:
            

            causal_feature_batched = casual_embedding[triplet_mask]
            causal_feature_batched = self.causal_feat_proj(
                causal_feature_batched
            )
            
            casual_feature, triplet_masks = self.causal_graph_pool(
                causal_feature_batched,
                triplet_batch_idx,
                B
            )
            
            question_causal = self.QCausalTransformer(
                question,
                casual_feature,
                q_mask=question_mask,
                kv_mask=triplet_masks
            )
            
            causal_reason = self.CausalReasoningTransformer(
                casual_feature,
                question_causal,
                q_mask=triplet_masks,
                kv_mask=question_mask
            )
            
            causal_branches.append(
                causal_reason
            )

        if e_cl_embedding is not None:

            
            ecl_output = self.language_encoder(
                e_cl_embedding , ecl_mask
            )
            ecl = ecl_output["token_features"]         # [B,L,d]
            ecl_global = ecl_output["global_question"] # [B,d]
            ecl_global = ecl_global.unsqueeze(1) # [B,1,d]
           
            # -----------------------------------------------------
            # Motion
            # -----------------------------------------------------

            if motion_video_feat is not None:

                ecl_motion = self.QMotionTransformer(
                    ecl,
                    motion_feature,
                    q_mask=ecl_mask,
                    kv_mask=motion_mask
                )
                ecl_question_branches.append(
                    ecl_motion
                )
                e_cl_motion_reason = self.MotionReasoningTransformer(
                    motion_feature,
                    ecl_motion,
                    q_mask=motion_mask,
                    kv_mask=ecl_mask
                )
               
                feature_causal_branches.append(e_cl_motion_reason)

            # -----------------------------------------------------
            # Node
            # -----------------------------------------------------

            if node_feat is not None:
                

                ecl_node = self.QNodeTransformer(
                    ecl,
                    visual_node_feature,
                    q_mask=ecl_mask,
                    kv_mask=node_mask
                )
                ecl_question_branches.append(
                    ecl_node
                )
        
                e_cl_node_reason = self.NodeReasoningTransformer(
                    visual_node_feature,
                    ecl_node,
                    q_mask=node_mask,
                    kv_mask=ecl_mask
                )
                
                feature_causal_branches.append(e_cl_node_reason)
               
            # -----------------------------------------------------
            # Causal
            # -----------------------------------------------------

            if casual_embedding is not None:

                ecl_causal = self.QCausalTransformer(
                    ecl,
                    causal_feature_batched,
                    q_mask=ecl_mask,
                    kv_mask=triplet_masks
                )
                
                causal_reason = self.CausalReasoningTransformer(
                    causal_feature_batched,
                    ecl_causal,
                    q_mask=triplet_mask,
                    kv_mask=ecl_mask    
                )
                causal_branches.append(causal_reason)
                

        # =========================================================
        # Final fusion
        # =========================================================
        
        if len(question_branches) == 0:
            raise ValueError(
                "No active question branches."
            )
        
        if len(feature_branches) == 0:
            raise ValueError(
                "No active feature branches."
            )
            
        feature_tokens = torch.cat(
            feature_branches,
            dim=1
        )
        question_visual = torch.cat(
            question_branches,
            dim=1
        )
        if len(answer_branches) > 0:
            answer_visual = torch.cat(
                answer_branches,
                dim=1
            )
            answer_feature_tokens = torch.cat(
                answer_feature_branches,
                dim=1
            )
            answer_masks = torch.cat(
            [answer_mask_flat for _ in range(len(answer_branches))],
        dim=1) 
            
            answer_feature_fused = self.feature_aggregate(answer_global, answer_feature_tokens)
   
        
        question_masks = torch.cat(
        [question_mask for _ in range(len(question_branches))],
        dim=1)  # [B, total_L]
        
      
        
        feature_fused = self.feature_aggregate(question_global, feature_tokens) 
     
        if len(feature_causal_branches) > 0:
            feature_causal_fused = torch.cat(
            feature_causal_branches,
            dim=1)
            ecl_question_visual = torch.cat(
                ecl_question_branches,
                dim=1
            )
            feature_causal_fused = self.feature_aggregate(ecl_global, feature_causal_fused) 
            
            ecl_masks = torch.cat([
                ecl_mask for _ in range(len(ecl_question_branches))
            ], dim=1)
            
            
        if len(causal_branches) > 0:
            causal_fused = torch.cat(
                causal_branches,
                dim=1
            )
            causal_fused = self.feature_aggregate(question_global, causal_fused)
            
        if answer_branches:
            return (
                question_visual,
                answer_visual,
                ecl_question_visual,
                feature_fused,
                answer_feature_fused,
                feature_causal_fused,
                causal_fused,
                question_masks,
                ecl_masks,
                answer_masks
            )
        else:
            return ( 
                question_visual,
                ecl_question_visual,
                feature_fused,
                feature_causal_fused,
                causal_fused,
                question_masks,
                ecl_masks,
            )
    
    



class AttentionPool(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        num_queries=4,
        depth=2,
        dropout=0.1
    ):
        super().__init__()

        self.num_queries = num_queries
        self.depth = depth

        # -------------------------------------------------
        # Learnable reasoning queries
        # -------------------------------------------------

        self.queries = nn.Parameter(
            torch.randn(1, num_queries, dim)
        )

        # -------------------------------------------------
        # Token refinement
        # -------------------------------------------------

        self.self_attn_layers = nn.ModuleList([
            nn.ModuleDict({
                "norm1": nn.LayerNorm(dim),

                "attn": nn.MultiheadAttention(
                    embed_dim=dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True
                ),

                "norm2": nn.LayerNorm(dim),

                "ffn": nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * 4, dim)
                )
            })
            for _ in range(depth)
        ])

        # -------------------------------------------------
        # Query pooling attention
        # -------------------------------------------------

        self.pool_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # -------------------------------------------------
        # Gated fusion
        # -------------------------------------------------

        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

        # -------------------------------------------------
        # Final projection
        # -------------------------------------------------

        self.output = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,      # [B,S,d]
        mask    # [B,S]
    ):

        B, S, d = x.shape

        # =================================================
        # Token refinement
        # =================================================

        for layer in self.self_attn_layers:

            h = layer["norm1"](x)

            attn_out, _ = layer["attn"](
                query=h,
                key=h,
                value=h,
                key_padding_mask=~mask
            )

            x = x + attn_out

            h = layer["norm2"](x)

            x = x + layer["ffn"](h)

        # =================================================
        # Multi-query reasoning
        # =================================================

        q = self.queries.expand(B, self.num_queries, d)

        pooled, _ = self.pool_attn(
            query=q,
            key=x,
            value=x,
            key_padding_mask=~mask
        )

        # =================================================
        # Global token summary
        # =================================================

        masked_x = x * mask.unsqueeze(-1)

        global_feat = masked_x.sum(dim=1) / (
            mask.sum(dim=1, keepdim=True) + 1e-6
        )

        global_feat = global_feat.unsqueeze(1).expand_as(pooled)

        # =================================================
        # Gated fusion
        # =================================================

        gate = self.gate(
            torch.cat([pooled, global_feat], dim=-1)
        )

        pooled = gate * pooled + (1 - gate) * global_feat

        # =================================================
        # Aggregate queries
        # =================================================

        pooled = pooled.mean(dim=1, keepdim=True)

        pooled = self.output(pooled)

        return pooled



class GraphTokenPooling(nn.Module):

    def __init__(self, dim, num_tokens=64):
        super().__init__()

        self.score = nn.Linear(dim, 1)
        self.num_tokens = num_tokens

    def forward(self, x, batch_idx, B):

        pooled_tokens = []
        pooled_masks = []

        for b in range(B):

            x_b = x[batch_idx == b]

            if x_b.size(0) == 0:
                pooled_tokens.append(
                    torch.zeros(
                        self.num_tokens,
                        x.size(-1),
                        device=x.device
                    )
                )

                pooled_masks.append(
                    torch.zeros(
                        self.num_tokens,
                        dtype=torch.bool,
                        device=x.device
                    )
                )

                continue

            scores = self.score(x_b).squeeze(-1)

            K = min(self.num_tokens, x_b.size(0))

            topk = scores.topk(K).indices

            selected = x_b[topk]

            if K < self.num_tokens:

                pad = torch.zeros(
                    self.num_tokens - K,
                    x.size(-1),
                    device=x.device
                )

                selected = torch.cat([selected, pad], dim=0)

            mask = torch.zeros(
                self.num_tokens,
                dtype=torch.bool,
                device=x.device
            )

            mask[:K] = True

            pooled_tokens.append(selected)
            pooled_masks.append(mask)

        pooled_tokens = torch.stack(pooled_tokens)
        pooled_masks = torch.stack(pooled_masks)

        return pooled_tokens, pooled_masks
      
