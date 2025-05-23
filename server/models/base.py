import importlib
import json
import os
import random
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch
from FlagEmbedding import FlagModel

class BaseRecallModel(ABC):
    @abstractmethod
    def recall(self, idx_list, preference_list, current_step, recall_num_list, k) -> List[List[str]]:
        pass

class RecallModel(BaseRecallModel):
    def __init__(self, model_name, config):
        super().__init__()
        self.model_name = model_name
        self.config = config
        module = importlib.import_module(f'server.models.{model_name.lower()}')
        self.recall_model = getattr(module, model_name)(config)

        self.device = torch.device(config["device"])
        id2meta = json.load(open(config["id2meta_path"]))
        self.id2meta = self.meta_formatting(id2meta)
        self.idx2history = {}
        self.idx2label = dict()
        datasets = ["train_id.jsonl", "valid_id.jsonl", "test_id.jsonl"]
        for d in datasets:
            f_path = os.path.join(config["history_path"], d)
            with open(f_path, "r") as f:
                for line in f.readlines():
                    piece = json.loads(line)
                    self.idx2history[str(piece["idx"])] = piece["history"]
                    self.idx2label[str(piece["idx"])] = int(piece["target"])

        self.flag_model = FlagModel(
            model_name_or_path=config["flag_model_path"],
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
            use_fp16=False,
            device=[self.device]
        )

        self.max_his_len = config["max_his_len"]




    def meta_formatting(self, id2meta):
        id2meta_formatted = {}
        for item, meta in id2meta.items():
            if 'categories' in meta:
                categories = meta['categories'].replace('Musical Instruments ', '')
                formatted_meta = f"{meta['title']}\n  - categories: {categories}"
            else:
                formatted_meta = f"{meta['title']}\n  - category: {meta['category']}"

            id2meta_formatted[item] = formatted_meta
        return id2meta_formatted

    def recall(self, idx_list, preference_list, current_step,
               recall_num_list, k, **kwargs):
        history_list = [self.idx2history[str(idx)] for idx in idx_list]
        history_list = [
            his[-self.max_his_len:] if len(his) >= self.max_his_len else his + [0] * (self.max_his_len - len(his)) for
            his in history_list]
        history_list = torch.tensor(history_list, device=self.device).long()  # [B, L]
        seq_len = history_list.ne(0).sum(dim=1)  # [B]
        query_embeddings = self.flag_model.encode_queries(preference_list, convert_to_numpy=False,
                                                          max_length=256)  # [B, H]

        scores = self.recall_model.full_sort_predict(history_list, seq_len, query_embeddings)

        _, indices = torch.topk(scores, k, dim=1)
        indices = indices.cpu().numpy().tolist()

        titles_list = []
        for session_idx, recall_id_list, recall_num in zip(idx_list, indices, recall_num_list):
            titles = [self.id2meta[str(int(i))] for i in recall_id_list[:k]]

            titles_list.append(titles)

        return titles_list


