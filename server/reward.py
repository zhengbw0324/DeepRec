import argparse
import json
import os

import emoji
import numpy as np
import torch.nn
import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from openrlhf.datasets.prompts_dataset import *
from openrlhf.utils.logging_utils import init_logger
from transformers import AutoTokenizer, LlamaModel
from FlagEmbedding import FlagModel
import torch.nn.functional as F

from server.utils import parse_command_line_args

logger = init_logger(__name__)

class RecRuleProxy:
    def __init__(self, config):

        self.config = config


        self.idx2label = dict()
        with open(config["data_file_path"], encoding='utf-8') as f:
            for line in f:
                data_dict = json.loads(line)
                self.idx2label[str(data_dict['idx'])] = data_dict['target'].strip()



        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name_or_path"])
        self.eos_token = tokenizer.eos_token
        self.pad_token_escaped = re.escape(tokenizer.pad_token)
        self.eos_token_escaped = re.escape(tokenizer.eos_token)

        self.topk = config["topk"]
        self.item_split_identifier = config["item_split_identifier"]

        self.log_file_path = os.path.join(config["log_dir"], config["log_file"])
        if self.log_file_path is not None:
            os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)


        self.stage = config["stage"]

        self.format_error = config["format_error"]
        self.outcome_error = config["outcome_error"]
        self.outcome_correct = config["outcome_correct"]
        self.outcome_linear = config["outcome_linear"]
        self.recall_num_base = config["recall_num_base"]
        self.recall_num_max = config["recall_num_max"]
        self.diversity_base = config["diversity_base"]


        self.flag_model = FlagModel(
            model_name_or_path=config["flag_model_path"],
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
            use_fp16=False,
            device=[self.device]
        )


        id2meta = json.load(open(config["id2meta_path"]))
        self.title2id = self.get_title2id(id2meta)
        self.item_text_embs = self.get_text_embs(config["text_emb_path"])
        self.item_embs = self.get_item_embs(config["item_emb_path"])
        print(f"item_text_embs shape: {self.item_text_embs.shape}")
        print(f"item_embs shape: {self.item_embs.shape}")

    def get_title2id(self, id2meta):
        title2id = {}
        for item_id, meta in id2meta.items():
            title = meta['title'].strip().lower()
            title2id[title] = int(item_id)
        return title2id

    def get_item_embs(self, item_emb_path):
        item_embs = np.load(item_emb_path)
        item_embs = torch.from_numpy(item_embs).to(self.device)
        # norm
        item_embs = F.normalize(item_embs, p=2, dim=-1)

        return item_embs
    def get_text_embs(self, text_emb_path):

        text_embs = np.load(text_emb_path)
        pad_emb = np.zeros((1, text_embs.shape[1]), dtype=text_embs.dtype)
        text_embs = np.concatenate((pad_emb, text_embs), axis=0)
        text_embs = torch.from_numpy(text_embs).to(self.device)
        # text_embs = F.normalize(text_embs, p=2, dim=-1)

        return text_embs


    def _log_str(self, s):
        if self.log_file_path is not None:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(s + '\n')

    def _log_json(self, d):
        if self.log_file_path is not None:
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')

    def _process_query(self, text):
        pattern = f"^({self.eos_token_escaped}|{self.pad_token_escaped})+"
        text = re.sub(pattern, "", text)

        pattern = f"({self.eos_token_escaped}|{self.pad_token_escaped})+$"
        text = re.sub(pattern, "", text)

        return text

    def _get_qa(self, text):
        remove_prefix = " ".join(text.split("\nUser:")[1:])
        question = remove_prefix.split(f"\nAssistant:\n{THINK_BEG}")[0].strip()
        solution = text.split(f"\nAssistant:\n{THINK_BEG}")[-1].strip()
        return question, solution

    def _get_pred(self, text):
        return text.split(REC_BEG)[-1].split(REC_END)[0].strip()


    def _get_preferences(self, text):
        preferences = text.split(PREF_BEG)[1:]
        preferences = [pref.split(PREF_END)[0].strip() for pref in preferences]
        return preferences

    def _get_outcome_reward(self, pred, label) -> float:
        pred_list = pred.split(self.item_split_identifier)[:self.topk]
        pred_list = [pred.split('. ', 1)[-1].strip() for pred in pred_list]

        reward = self.outcome_error
        for i, pred in enumerate(pred_list):
            if label in pred:
                reward = (self.topk - i) * self.outcome_linear + self.outcome_correct
                break

        return reward

    def _get_sim_reward(self, pred, label) -> float:


        pred_list = pred.split(self.item_split_identifier)[:self.topk]
        pred_list = [pred.split('. ', 1)[-1].strip() for pred in pred_list]
        pred_id_list = [self.title2id.get(pred.strip().lower(), 0) for pred in pred_list]
        label_id = self.title2id.get(label.strip().lower(), 0)
        # print(pred_id_list)
        # print(label_id)

        pred_text_embs = self.item_text_embs[pred_id_list]
        label_text_emb = self.item_text_embs[label_id].unsqueeze(0)
        pred_text_sims = (pred_text_embs @ label_text_emb.T).squeeze(1).cpu()

        pred_item_embs = self.item_embs[pred_id_list]
        label_item_emb = self.item_embs[label_id].unsqueeze(0)
        pred_item_sims = (pred_item_embs @ label_item_emb.T).squeeze(1).cpu()

        pred_sims = (pred_text_sims + pred_item_sims)/2
        pred_sims = pred_sims.tolist()

        n = len(pred_list)
        weights = [(1 - i / n)** 2 for i in range(n)]
        weighted_sum = sum(weight * sim for weight, sim in zip(weights, pred_sims))
        reward = weighted_sum / sum(weights)


        return reward * self.sim_reward_base



    def _get_diversity_reward(self, preferences):
        n = len(preferences)

        if n <= 1:
            return 0

        preferences_embs = self.flag_model.encode_queries(preferences, convert_to_numpy=False,
                                                            max_length=256)  # [n, d]

        similarity = preferences_embs @ preferences_embs.T

        upper_triangle_indices = torch.triu_indices(n, n, 1)
        similarity_values = similarity[upper_triangle_indices[0], upper_triangle_indices[1]]
        avg_similarity = similarity_values.mean().item()

        diversity_reward = 1 - avg_similarity

        return diversity_reward * self.diversity_base

    def _get_recall_num_reward(self, recall_num, base=0.05, num_max=4):

        recall_num = min(recall_num, num_max)

        if recall_num <= 1:
            return 0

        reward = (recall_num - 1) * base

        return reward


    def _get_format_reward(self, text, recall_num, pred):

        format_punishment = False

        count_1 = text.count(PREF_BEG)
        count_2 = text.count(PREF_END)
        count_3 = text.count(ITEM_LIST_BEG)
        count_4 = text.count(f"{ITEM_LIST_BEG}\n")
        count_5 = text.count(f"{ITEM_LIST_BEG}\n1.")
        count_6 = text.count(ITEM_LIST_END)
        count_7 = text.count(f"{ITEM_LIST_END}\n")

        if count_1 == count_2 == count_3 == count_4 == count_5 == count_6 == count_7:
            pass
        else:
            self._log_str('tool call token error')
            format_punishment = True

        if recall_num == 0:
            self._log_str('no tool call')
            format_punishment = True

        count_assistant_1 = text.count("Assistant")
        count_assistant_2 = text.count("assistant")
        if count_assistant_1 == count_assistant_2 == 0:
            pass
        else:
            self._log_str('role token error')
            format_punishment = True

        count_example_preference_format = text.count("[user preference phrases, concatenated by ;]")
        count_example_think_format = text.count(
            "[Continue iterating through the process—reasoning before tool invocation,")
        count_example_item_list_format = text.count("[recalled item")
        count_example_history_format = text.count("[purchased item")
        count_example_recommendation_format = text.count(" to recommend]\n")

        if count_example_preference_format == count_example_think_format == count_example_item_list_format == count_example_history_format == count_example_recommendation_format == 0:
            pass
        else:
            self._log_str('example format error')
            format_punishment = True

        count_think_1 = text.count(THINK_BEG)
        count_think_2 = text.count(THINK_END)
        if count_think_1 == 0 and count_think_2 == 1:
            pass
        else:
            self._log_str('think token error')
            format_punishment = True

        count_answer_1 = text.count(REC_BEG)
        count_answer_2 = text.count(REC_END)
        text_after_think = text.split(THINK_END)[-1]
        count_answer_3 = text_after_think.count(REC_BEG)
        count_answer_4 = text_after_think.count(REC_END)
        # text_after_rec = text.split(REC_END)[-1].strip()
        if count_answer_1 == count_answer_2 == count_answer_3 == count_answer_4 == 1:
            pass
        else:
            self._log_str('answer token error')
            format_punishment = True

        text_after_last_tool_call = text.split(ITEM_LIST_END)[-1].split(THINK_END)[0].strip()
        if len(text_after_last_tool_call) == 0:
            self._log_str('text after last tool call error')
            format_punishment = True

        if inner_PREF_BEG not in pred and inner_PREF_END not in pred and inner_ITEM_LIST_BEG not in pred and inner_ITEM_LIST_END not in pred:
            pass
        else:
            self._log_str('illegal token in answer')
            format_punishment = True

        answer_len = len(pred.split(self.item_split_identifier))
        if answer_len != self.topk:
            self._log_str(f'item number in answer error, item_split_identifier: {self.item_split_identifier}, '
                          f'required: {self.topk}, actual: {answer_len}')
            format_punishment = True

        modified_text = re.sub(rf'{escaped_ITEM_LIST_BEG}.*?{escaped_ITEM_LIST_END}', '', text,
                                   flags=re.DOTALL)
        have_chinese = any('\u4e00' <= char <= '\u9fff' for char in modified_text)
        if have_chinese is True:
            self._log_str('has chinese')
            format_punishment = True

        if format_punishment is True:
            format_reward = self.format_error
        else:
            format_reward = 0

        return format_reward


    def get_reward(self, query_list, idx_list, current_step, recall_num_list, **kwargs):
        pred_list = []
        label_list = []

        question_list = []
        solution_list = []

        for idx, query in zip(idx_list, query_list):
            query = self._process_query(query)

            question, solution = self._get_qa(query)
            question_list.append(question)
            solution_list.append(solution)

            pred = self._get_pred(solution)
            label = self.idx2label[idx]

            pred_list.append(pred)
            label_list.append(label)

        score_list = []
        for idx, query, solution, pred, label, recall_num in zip(idx_list, query_list, solution_list, pred_list, label_list, recall_num_list):

            format_reward = self._get_format_reward(solution, recall_num, pred)
            outcome_reward = 0
            if format_reward > self.format_error and self.stage == "rec":
                outcome_reward = self._get_outcome_reward(pred, label)

            pred_sim_reward = 0
            if self.sim_reward_base != 0 and format_reward > self.format_error and self.stage == "rec":
                pred_sim_reward = self._get_sim_reward(pred, label)

            tool_call_reward = 0
            if self.recall_num_base != 0 and self.stage == "cold":
                tool_call_reward = self._get_recall_num_reward(recall_num, base=self.recall_num_base, num_max=self.recall_num_max)

            diversity_reward = 0
            if self.diversity_base !=0 and self.stage == "cold":
                preferences = self._get_preferences(solution)
                diversity_reward = self._get_diversity_reward(preferences)

            score_list.append(list(map(float, [outcome_reward, format_reward, pred_sim_reward, tool_call_reward, diversity_reward])))
            self._log_json(dict(
                step=current_step, idx=idx, recall_num=recall_num,
                outcome_reward=outcome_reward,
                format_reward=format_reward,
                pred_sim_reward=pred_sim_reward,
                tool_call_reward=tool_call_reward,
                diversity_reward=diversity_reward,
                query=query,
            ))

        return score_list


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", type=str, default="./server/config/Reward.yaml")

    parser.add_argument("--host", type=str, default="0.0.0.0", help="IP for the server")
    parser.add_argument("--port", type=int, default=5001, help="Port number for the server")

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)
    # print(args)

    with open(args.config_file) as f:
        config = yaml.safe_load(f)

    config.update(command_line_configs)

    print(config)
    # server
    reward_model = RecRuleProxy(config)

    app = FastAPI()

    @app.post("/reward")
    async def get_reward(request: Request):
        data = await request.json()
        rewards = reward_model.get_reward(**data)
        result = {"rewards": rewards}
        logger.info(f"Sent JSON: {result}")
        return JSONResponse(result)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
