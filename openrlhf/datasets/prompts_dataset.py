import re

from torch.utils.data import Dataset
from tqdm import tqdm

from openrlhf.datasets.utils import load_json

ID_SPLIT = "<|idx_prompt_split|>"
THINK_BEG = '<think>'
THINK_END = '</think>'
REC_BEG = '<recommendation_list>'
REC_END = '</recommendation_list>'
PREF_BEG = '<|begin_of_preference|>'
PREF_END = '<|end_of_preference|>'
ITEM_LIST_BEG = '<|begin_of_item_list|>'
ITEM_LIST_END = '<|end_of_item_list|>'

inner_PREF_BEG = 'begin_of_preference'
inner_PREF_END = 'end_of_preference'
inner_ITEM_LIST_BEG = 'begin_of_item_list'
inner_ITEM_LIST_END = 'end_of_item_list'

escaped_ITEM_LIST_BEG = re.escape(ITEM_LIST_BEG)
escaped_ITEM_LIST_END = re.escape(ITEM_LIST_END)

base_prompt_rec = f"""
You are the assistant to make item recommendations for the user. \
Remember that, you have no access to the candidate items. \
So, you need to call an item retrieval tool to find items you want to recommend. \
The item retrieval tool receives the description of user preference as input and returns a list of relevant items as output.
The format of input:
{PREF_BEG} [user preference phrases, concatenated by ;] {PREF_END}
The format of output:
{ITEM_LIST_BEG}
1. [retrieved item 1]
2. [retrieved item 2]
...
n. [retrieved item n]
{ITEM_LIST_END}

The user provides the items he/she interacted with in the past and lists them in order of interaction time. For example:
1. [interacted item 1]
2. [interacted item 2]
...
n. [interacted item n]

To make recommendations:

1. You need to reason about the interaction history and generate the description of user preference in the format of "{PREF_BEG} [user preference phrases, concatenated by ;] {PREF_END}" to call the item retrieval tool to find relevant items.

2. **Quality over quantity**: Even if you've found enough items, continue calling the item retrieval tool with varied preference descriptions to find potentially better items. Explore different aspects of user preferences to ensure comprehensive coverage.

3. **Iterative refinement**: After each tool call, analyze the returned items and adjust your next preference description based on what worked well and what didn't, or explore another aspect of user preferences.

5. After gathering a diverse pool of items, rank all the retrieved items by reasoning about the likelihood that the user would interact with them, according to the interaction history.

6. After the above reasoning, tool calling, and ranking process, output the final ranked item title list with the top {{k}} items.

The reasoning process and the final ranked item list should be enclosed within {THINK_BEG} {THINK_END} and {REC_BEG} {REC_END} tags, respectively. \
Your response should follow this format:
{THINK_BEG}
[Continue iterating through the process—reasoning before tool invocation, tool execution, and post-tool evaluation—until the desired items are found. Conclude by reasoning a ranked item list based on the iterative refinements.]
{THINK_END}
{REC_BEG}
1. [item title 1 to recommend]
2. [item title 2 to recommend]
...
n. [item title n to recommend]
{REC_END}

Now, let's begin.

User:
{{history}}

Assistant:
{THINK_BEG}
""".lstrip()






chat_prompt_rec = f"""
You are the assistant to make item recommendations for the user. \
Remember that, you have no access to the candidate items. \
So, you MUST call an item retrieval tool to find items you want to recommend. \
The item retrieval tool receives the description of user preference as input and returns a list of relevant items as output.
The format of input:
{PREF_BEG} [user preference phrases, concatenated by ;] {PREF_END}
The format of output:
{ITEM_LIST_BEG}
1. [retrieved item 1]
2. [retrieved item 2]
...
n. [retrieved item n]
{ITEM_LIST_END}

The user provides the items he/she interacted with in the past and lists them in order of interaction time. For example:
1. [interacted item 1]
2. [interacted item 2]
...
n. [interacted item n]

To make recommendations:

1. You need to reason about the interaction history and generate the description of user preference in the format of "{PREF_BEG} [user preference phrases, concatenated by ;] {PREF_END}" to call the item retrieval tool to find relevant items.

2. **Quality over quantity**: Even if you've found enough items, continue calling the item retrieval tool with varied preference descriptions to find potentially better items. Explore different aspects of user preferences to ensure comprehensive coverage.

3. **Iterative refinement**: After each tool call, analyze the returned items and adjust your next preference description based on what worked well and what didn't, or explore another aspect of user preferences.

5. After gathering a diverse pool of items, rank all the retrieved items by reasoning about the likelihood that the user would interact with them, according to the interaction history.

6. After the above reasoning, tool calling, and ranking process, output the final ranked item title list with the top {{k}} items.

The reasoning process and the final ranked item list should be enclosed within {THINK_BEG} {THINK_END} and {REC_BEG} {REC_END} tags, respectively. \
Your response should follow this format:
{THINK_BEG}
[Continue iterating through the process—reasoning before tool invocation, tool execution, and post-tool evaluation—until the desired items are found. Conclude by reasoning a ranked item list based on the iterative refinements.]
{THINK_END}
{REC_BEG}
1. [item title 1 to recommend]
2. [item title 2 to recommend]
...
n. [item title n to recommend]
{REC_END}


Now, let's begin.

User interaction history:
{{history}}

""".lstrip()


def preprocess_history(history):

    lines = history.split("\n")
    # 截断过长的行，注意排除1以1. 2.之类开头的行
    new_lines = []
    for line in lines:
        if re.match(r'^\d+\.', line):
            new_lines.append(line)
        elif line.strip().startswith("- description:"):
            continue
        elif line.strip().startswith("- categories:"):
            new_line = line
            new_lines.append(new_line)
        else:
            # continue
            max_len = 750
            new_line = line[:max_len] + "..." if len(line) > max_len else line
            new_lines.append(new_line)
    history = "\n".join(new_lines)
    return history



def preprocess_data(data, k, input_template=None, input_key="input", apply_chat_template=None) -> str:
    history = data["history"]
    history = preprocess_history(history)

    idx = data["idx"]

    if apply_chat_template:
        messages_chat_rec = [
            {"role": "user", "content": chat_prompt_rec.format(history=history, k=k)}
        ]
        prompt = apply_chat_template(messages_chat_rec, tokenize=False, add_generation_prompt=True) + THINK_BEG + "\n"
    else:
        prompt = base_prompt_rec.format(history=history, k=k)

    return str(idx) + ID_SPLIT + prompt


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        k = strategy.args.rec_topk
        filtered_idx = None
        if strategy.args.filtered_idx_file and strategy.args.filtered_idx_key:
            # Load filtered idx from file
            print(f"Loading filtered idx from {strategy.args.filtered_idx_file}")
            print(f"Filtered idx key: {strategy.args.filtered_idx_key}")
            filtered_idx = set(load_json(strategy.args.filtered_idx_file)[strategy.args.filtered_idx_key])

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            if filtered_idx and data["idx"] not in filtered_idx:
                continue
            prompt = preprocess_data(data, k, input_template, input_key, apply_chat_template)
            self.prompts.append(prompt)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]
