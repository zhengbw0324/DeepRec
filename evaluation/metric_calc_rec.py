
import argparse
import json
import math
import os
import pdb
import sys

import numpy as np
import re
import string
from collections import Counter, defaultdict, OrderedDict
import pickle




def read_jsonl(test_results_dir):
    files  = os.listdir(test_results_dir)
    data = []
    for file in files:
        if file.endswith(".jsonl"):
            file_path = os.path.join(test_results_dir, file)
        else:
            continue
        with open(file_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
    return data


def rec_str2item_list(response:str)->list:
    item_list =  [item for item in response.strip().split('\n') if item]
    item_list = [item for item in item_list if item[0].isdigit()]
    return item_list

def get_rec_result(pred_item_list, target):
    pred_result = []
    for item in pred_item_list:
        if target in item:
            pred_result.append(1)
        else:
            pred_result.append(0)
    return pred_result

def get_recall(pred_result, topk):
    return sum(pred_result[:topk])

def get_ndcg(pred_result, topk):
    res = pred_result[:topk]
    ndcg = 0.0
    for i in range(len(res)):
        ndcg += res[i] / math.log(i + 2, 2)
    return ndcg

def get_mrr(pred_result, topk):
    if len(pred_result[:topk]) == 0:
        return 0
    idxs = np.argmax(pred_result[:topk])
    if pred_result[idxs] == 0:
        return 0
    return 1 / (idxs + 1)

metrics_fn = {
    "recall": get_recall,
    "ndcg": get_ndcg,
    "mrr": get_mrr,
}

def eval(args):
    result_dir = os.path.join(args.eval_dir, args.test_results_dir)
    data = read_jsonl(result_dir)
    total_data_num = len(data)
    print(f"Total data num: {total_data_num}")
    print(f"Eval {len(data)} from {args.test_results_dir}")
    metrics = args.metrics.split(",")
    metric_dict = defaultdict(int)

    error_num = defaultdict(int)
    for d in data:

        pred_answer = d["pred_ans"]
        # print(pred_answer)
        target = d["target"]
        if pred_answer == "I don't know.":
            error_num[d["stop_reason_final"]] += 1
            continue
        pred_item_list = rec_str2item_list(pred_answer)
        # print(len(pred_item_list))
        pred_result = get_rec_result(pred_item_list, target)
        for metric in metrics:
            metric_name, topk = metric.split("@")
            topk = int(topk)
            metric_dict[metric] += metrics_fn[metric_name](pred_result, topk)

    print(f"Error num: {error_num}")
    final_metrics = OrderedDict()
    for metric in metrics:
        final_metrics[metric] = round(metric_dict[metric] / total_data_num, 4)

    return final_metrics




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_results_dir", type=str, required=True)
    parser.add_argument("--eval_dir", type=str, default="./results/eval/")
    parser.add_argument("--metrics", type=str, default="recall@1,recall@5,recall@10,ndcg@5,ndcg@10")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    result = eval(args)
    print(result)


