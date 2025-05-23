import os
import argparse
import torch.distributed as dist
import json
from vllm import LLM, SamplingParams
from datasets import Dataset

import os

import copy




import re
from time import sleep
import requests
import json
import multiprocessing
from collections import defaultdict
from tqdm import tqdm
from openrlhf.datasets.prompts_dataset import ID_SPLIT, THINK_BEG, THINK_END, REC_BEG, REC_END, PREF_BEG, PREF_END, \
    ITEM_LIST_BEG, ITEM_LIST_END, base_prompt_rec, chat_prompt_rec, preprocess_history


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="None")
    parser.add_argument("--start_data_idx", type=int, default=-1)
    parser.add_argument("--end_data_idx", type=int, default=100000)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--model_path", type=str, default="None")
    parser.add_argument("--prompt_type", type=str, default="base")
    parser.add_argument("--gpu_memory_rate", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--chunk_size", type=int, default=2000)
    parser.add_argument("--rec_topk", type=int, default=10)
    parser.add_argument("--recall_server", type=str, default="http://XXXXX:6001/recall")
    parser.add_argument("--recall_topk", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="./results/eval/")
    return parser.parse_args()

def process_text(examples, topk, tokenizer, prompt_type='base'):

    history = examples["history"]
    history = preprocess_history(history)

    examples["history"] = history

    if prompt_type == 'base':
        prompt = base_prompt_rec.format(history=history, k=topk)
    elif prompt_type == 'chat':
        messages_chat_rec = [
            {"role": "user", "content": chat_prompt_rec.format(history=history, k=topk)}
        ]
        prompt = tokenizer.apply_chat_template(messages_chat_rec, tokenize=False, add_generation_prompt=True) + THINK_BEG + "\n"

    examples["prompt"] = prompt

    return examples

def load_raw_data(args):
    data_raw_all = []
    with open(args.data_file, "r") as f:
        for i, line in enumerate(f):
            if args.start_data_idx <= i < args.end_data_idx:
                raw_data = json.loads(line)
                data_raw_all.append(raw_data)
            if i >= args.end_data_idx - 1:
                break
    print("All Data Length: ", len(data_raw_all))
    return data_raw_all

def main():
    print("=Begin="*10)
    args = parse_args()
    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    temperature=args.temperature
    model_path=args.model_path
    prompt_type = args.prompt_type
    gpu_memory_rate=args.gpu_memory_rate
    chunk_size = args.chunk_size
    rec_topk = args.rec_topk
    recall_server = args.recall_server
    recall_topk = args.recall_topk

    output_file = f"start{args.start_data_idx}_end{args.end_data_idx}.jsonl"
    ckpt_name = model_path.split("/")[-3] + "_" + model_path.split("/")[-1]
    output_dir = os.path.join(args.output_dir, ckpt_name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, output_file)

    data_raw_all = load_raw_data(args)

    chunk_num = len(data_raw_all) // chunk_size
    if len(data_raw_all) % chunk_size != 0:
        chunk_num += 1

    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, tensor_parallel_size=1, gpu_memory_utilization=gpu_memory_rate, trust_remote_code=True)
    tokenizer = llm.get_tokenizer()

    for chunk_i in range(chunk_num):
        print("=="*80)
        print("Begin Chunk: ",chunk_i ,"All: ", chunk_num)
        data = data_raw_all[chunk_i*chunk_size:(chunk_i+1)*chunk_size]

        # data_keys = ["history", "target"]
        data_keys = data[0].keys()
        dataset = Dataset.from_dict({key: [d[key] for d in data] for key in data_keys})
        print("Data Number: ", len(dataset))
        dataset = dataset.map(
            process_text,
            num_proc=16,
            fn_kwargs={"topk": rec_topk, "tokenizer": tokenizer, "prompt_type": prompt_type},
        )
        print(dataset)
        # print(dataset[0]["history"])
        print(dataset[0]["prompt"])

        stop_tokens = ["<|im_end|>", "<|endoftext|>", PREF_END, REC_END]
        sampling_params = SamplingParams(temperature=temperature,
                                         top_p=0.95,
                                         max_tokens=2048,
                                         stop=stop_tokens)

        finished_all_list=[]
        continued_data = copy.deepcopy(data)

        for step in range(10):

            if len(dataset) ==0:
                print(len(dataset))
                break

            outputs = llm.generate(dataset['prompt'], sampling_params)

            finished_texts = []
            continued_texts = []

            recall_idx_list = []
            preference_list = []

            for i, output in enumerate(outputs):

                prompt = output.prompt
                idx = continued_data[i]["idx"]
                target = continued_data[i]["target"]
                history = dataset[i]["history"]
                gen_text_store = continued_data[i].get("gen_text_store", "")
                stop_reason = output.outputs[0].stop_reason
                generated_text = output.outputs[0].text

                if step == 8:
                    original_data = {
                            "idx": idx,
                            "history": history,
                            "target": target,
                            "gen_text_store": gen_text_store + generated_text,
                            "generated_text": generated_text,
                            "stop_reason_final": "many_recall",
                            "pred_ans": "I don't know."
                        }

                    finished_texts.append(original_data)
                    continue

                all_token_ids = list(output.prompt_token_ids) + list(output.outputs[0].token_ids)
                if len(all_token_ids) > 16384:
                    original_data = {
                        "idx": idx,
                        "history": history,
                        "target": target,
                        "gen_text_store": gen_text_store + generated_text,
                        "generated_text": generated_text,
                        "stop_reason_final": "context_too_long",
                        "pred_ans": "I don't know."
                    }
                    finished_texts.append(original_data)
                    continue

                if REC_BEG in generated_text and stop_reason==REC_END:
                    original_data = {
                        "idx": idx,
                        "history": history,
                        "target": target,
                        "pred_ans": generated_text.split(REC_BEG)[-1].split(REC_END)[0].strip(),
                        "stop_reason_final": "finished",
                        "gen_text_store": gen_text_store + generated_text + REC_END,
                }
                    finished_texts.append(original_data)

                elif PREF_BEG in generated_text and stop_reason==PREF_END: #retrieve

                    preference = generated_text.split(PREF_BEG)[-1].split(PREF_END)[0]
                    preference = preference.replace('"', "").strip()
                    preference = " ".join(preference.split())
                    if preference:
                        recall_idx_list.append(idx)
                        preference_list.append(preference)

                        original_data = {
                            "idx": idx,
                            "prompt": prompt + generated_text.strip(),
                            "history": history,
                            "target": target,
                            "stop_reason": stop_reason,
                            "gen_text_store": gen_text_store + generated_text.strip()
                            }
                        continued_texts.append(original_data)
                    else:
                        original_data = {
                            "idx": idx,
                            "history": history,
                            "target": target,
                            "gen_text_store": gen_text_store + generated_text.strip(),
                            "generated_text":generated_text,
                            "stop_reason_final": "preference_error",
                            "pred_ans": "I don't know."
                        }
                        finished_texts.append(original_data)

                else:
                    original_data = {
                    "idx": idx,
                    "history": history,
                    "target": target,
                    "gen_text_store": gen_text_store + generated_text.strip(),
                    "generated_text": generated_text,
                    "stop_reason_final": "shot_down",
                    "pred_ans": "I don't know."
                }
                    finished_texts.append(original_data)


            print("=="*80)

            assert len(recall_idx_list) == len(preference_list) == len(continued_texts), "Error in len of query_list and continued_texts"

            if len(preference_list)!=0 and len(recall_idx_list) != 0:
                response = requests.post(recall_server,
                             json={
                                 "idx_list": recall_idx_list,
                                 "preference_list": preference_list,
                                 "current_step": 0,
                                 "recall_num_list": [0] * len(recall_idx_list),
                                 "k": recall_topk
                             }
                )
                if response.status_code == 200:
                    result = response.json()
                    item_list = result["item_list"]
                    for i in range(len(item_list)):
                        recall_items = item_list[i]
                        continued_text_now = copy.deepcopy(continued_texts[i])
                        if len(recall_items) > 0:
                            item_content_list = []
                            for j in range(len(recall_items)):
                                # item = re.sub(r'^\d+\s+', '', recall_items[j])
                                item = recall_items[j]
                                item_content_list.append(f"{j + 1}. {item}\n")
                            item_content = ''.join(item_content_list)
                        else:
                            item_content = "None"
                        continued_text_now["prompt"] = continued_text_now["prompt"] + f"{PREF_END}\n\n"+ f"{ITEM_LIST_BEG}\n" +  item_content + f"{ITEM_LIST_END}\n\n"
                        continued_text_now["gen_text_store"] = continued_text_now["gen_text_store"] + f"{PREF_END}\n\n"+ f"{ITEM_LIST_BEG}\n" +  item_content + f"{ITEM_LIST_END}\n\n"
                        continued_texts[i] = continued_text_now
                else:
                    for i in range(len(continued_texts)):
                        current_data = continued_texts[i]
                        original_data = {
                            "idx": current_data["idx"],
                            # "prompt": current_data["prompt"],
                            "history": current_data["history"],
                            "target": current_data["target"],
                            "gen_text_store": current_data["gen_text_store"],
                            "generated_text": current_data["generated_text"],
                            "stop_reason_final": "recall_error",
                            "pred_ans": "I don't know."
                        }
                        finished_texts.append(original_data)

            finished_all_list.extend(finished_texts)


            print("==" * 80)
            print("Step: ", step, "New_Finished: ", len(finished_texts), "All_Finished ", len(finished_all_list),
                  "Continued: ", len(continued_texts))
            print("Begin Writing Epoch: ", step)

            if len(continued_texts)==0:
                if len(finished_texts)>0:
                    with open(output_file, "a") as f:
                        for text in finished_texts:
                            f.write(json.dumps(text) + "\n")
                break
            else:
                data_keys_again = continued_texts[0].keys()
                dataset = Dataset.from_dict({key: [d[key] for d in continued_texts] for key in data_keys_again})
                continued_data = copy.deepcopy(continued_texts)


            print("=="*80)
            if len(finished_texts)>0:
                with open(output_file, "a") as f:
                    for text in finished_texts:
                        f.write(json.dumps(text) + "\n")

    if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()


