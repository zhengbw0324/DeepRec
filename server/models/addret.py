import os
import random
import torch
import json
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from server.models.layers import *
from server.models.base import BaseRecallModel
from FlagEmbedding import FlagModel


class AddRet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.max_his_len = config["max_his_len"]
        self.n_items = config["n_items"]
    
        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.embedding_size = config["embedding_size"]
        self.hidden_size = config["hidden_size"]
        self.plm_size = config["plm_size"]
        self.hidden_dropout_prob = config["dropout_prob"]
        self.attn_dropout_prob = config["dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = eval(config["layer_norm_eps"])
        self.bidirectional = config["bidirectional"]
        self.n_exps = config["n_exps"]
        self.adaptor_layers = config["adaptor_layers"]
        self.adaptor_dropout_prob = config["adaptor_dropout_prob"]

        self.device = torch.device(config["device"])


        self.weight_path = config["weight_path"]

        self.item_embedding = nn.Embedding(
            self.n_items+1, self.embedding_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_his_len, self.embedding_size)

        self.trm_encoder = Transformer(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.embedding_size,
            inner_size=self.hidden_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )
        self.OutLayerNorm = nn.LayerNorm(self.embedding_size, eps=self.layer_norm_eps)
        
        self.LayerNorm = nn.LayerNorm(self.embedding_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.item_plm_embedding = nn.Embedding(self.n_items+1, self.plm_size, padding_idx=0)
        self.item_plm_embedding.weight.requires_grad_(False)    

        self.item_gating = nn.Linear(self.embedding_size, 1)
        self.fusion_gating = nn.Linear(self.embedding_size, 1)

        self.moe_adaptor = MoEAdaptorLayer(
            self.n_exps,
            self.adaptor_layers,
            self.adaptor_dropout_prob
        )

        self.complex_weight = nn.Parameter(torch.randn(1, self.max_his_len // 2 + 1, self.embedding_size, 2, dtype=torch.float32) * 0.02)
        # parameters initialization
        self.init_trained_weights()
        self.init_device()
        self.eval()

    
    def init_trained_weights(self):
        model_dict = torch.load(self.weight_path, weights_only=False, map_location="cpu")
        missing_keys, unexpected_keys = self.load_state_dict(model_dict, strict=False)
        print("missing_keys:", missing_keys)
        print("unexpected_keys:", unexpected_keys)
    
    def init_device(self):
        self.to(self.device)

    def contextual_convolution(self, item_emb, feature_emb):
        """Sequence-Level Representation Fusion
        """
        feature_fft = torch.fft.rfft(feature_emb, dim=1, norm='ortho')
        item_fft = torch.fft.rfft(item_emb, dim=1, norm='ortho')

        complext_weight = torch.view_as_complex(self.complex_weight)
        item_conv = torch.fft.irfft(item_fft * complext_weight, n = feature_emb.shape[1], dim = 1, norm = 'ortho')
        fusion_conv = torch.fft.irfft(feature_fft * item_fft, n = feature_emb.shape[1], dim = 1, norm = 'ortho')

        item_gate_w = self.item_gating(item_conv)
        fusion_gate_w = self.fusion_gating(fusion_conv)

        contextual_emb = 2 * (item_conv * torch.sigmoid(item_gate_w) + fusion_conv * torch.sigmoid(fusion_gate_w))
        return contextual_emb
    
    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
    
    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask

    def forward(self, item_seq, item_seq_len, query_text_emb):
        B, L = item_seq.size(0), item_seq.size(1)
        position_ids = torch.arange(L, dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand(B, -1)
        position_embedding = self.position_embedding(position_ids)

        item_text_emb = self.item_plm_embedding(item_seq)  # [B, L, H]
        item_id_emb = self.item_embedding(item_seq)  # [B, L, H]

        item_text_emb = self.moe_adaptor(item_text_emb)

        input_emb = self.contextual_convolution(item_id_emb, item_text_emb)
        input_emb = input_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq, bidirectional=self.bidirectional)
        trm_output = self.trm_encoder(input_emb, input_emb, extended_attention_mask, output_all_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)


        query_text_emb = self.moe_adaptor(query_text_emb)

        output = self.OutLayerNorm(output + query_text_emb)
        
        return output  # [B H]

    def full_sort_predict(self, item_seq, item_seq_len, query_text_emb):
        seq_output = self.forward(item_seq, item_seq_len, query_text_emb)
        seq_output = F.normalize(seq_output, dim=-1)

        test_item_emb = self.item_embedding.weight
        test_item_emb = F.normalize(test_item_emb, dim=-1)

        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # [B, n_items]
        scores[:, 0] = -1e9
        return scores
    
