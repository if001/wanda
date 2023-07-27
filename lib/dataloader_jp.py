import json
import os.path as osp
from typing import Union

from datasets import load_dataset

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("./templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
    
def get_jp_loarder(template_name, data_path, tokenizer, val_set_size, seq_len):
    prompter = Prompter(template_name)

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        # print('full_prompt: ', full_prompt)
        data = tokenizer(full_prompt, return_tensors='pt', padding="max_length", truncation=True, max_length=seq_len)
        inp = data.input_ids
        tar = inp.clone()
        tar[:, :-1] = -100
        return (inp, tar)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    elif data_path.endswith(".csv"):
        data = load_dataset("csv", data_files=data_path)
    else:
        data = load_dataset(data_path)

    train_val = data["train"].train_test_split(
        test_size=val_set_size, shuffle=True, seed=42
    )
    trainloader = []
    for v in train_val["train"].shuffle():
        d = generate_and_tokenize_prompt(v)
        trainloader.append(d)
    val_enc = []
    for v in train_val["test"].shuffle():
        d = generate_and_tokenize_prompt(v)
        val_enc.append(d)
    return trainloader, val_enc