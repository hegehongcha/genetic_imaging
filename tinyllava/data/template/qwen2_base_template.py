from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union
import copy
from .formatter import EmptyFormatter, StringFormatter
from .base import Template
from .formatter import Formatter
from . import register_template
from ...utils.constants import *
from transformers import PreTrainedTokenizer
import torch

system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."


@register_template('qwen2_base')
@dataclass
class Qwen2BaseTemplate(Template):
    format_image_token: "Formatter" = StringFormatter(slot="{{content}}")
    format_user: "Formatter" = StringFormatter(slot="USER" + ": " + "{{content}}" + " ")
    format_assistant: "Formatter" = StringFormatter(slot="ASSISTANT" + ": " + "{{content}}" + "<|endoftext|>")
    system: "Formatter" = EmptyFormatter(slot=system + " ")
    separator: "Formatter" = EmptyFormatter(slot=[' ASSISTANT: ', '<|endoftext|>'])

    def make_labels(self, input_ids, prompt, tokenizer):
        labels = copy.deepcopy(input_ids)
        # print("labels: ", labels)
        sep, eos_token = self.separator.apply()
        # print("sep: ", sep)  # ASSISTANT:
        # print("eos_token: ", eos_token)  # </s>
        total_len = int(labels.ne(tokenizer.pad_token_id).sum()) + 1
        # print("total_len_before: ", total_len)  # 162
        # if tokenizer.pad_token_id == tokenizer.eos_token_id:
        #     total_len += prompt.count(eos_token)
        rounds = prompt.split(eos_token)
        # print("total_len_after: ", total_len)  # 163

        # ["A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers
        # to the user's questions. USER: <image>\nDescribe the given chest x-ray image in detail. ASSISTANT: The lungs are hyperexpanded with
        # severe emphysema. Right pigtail pleural drain has been removed. There is no evidence of pneumothorax. Mild subcutaneous gas persists
        # along the right chest wall. Heart size is normal. There is no pleural effusion. There are chronic interstitial abnormalities at the
        # lung bases. There is mild gaseous distention of loops of small bowel in the upper abdomen. Impression: 1. No appreciable pneumothorax.
        # 2. Severe emphysema is present.", '']
        # print("rounds_before: ", rounds)
        eos_token_length = len(tokenizer.encode(eos_token))  # 3
        labels, cur_len = self._make_masks(labels, tokenizer, sep, eos_token_length, rounds)
        if cur_len < tokenizer.model_max_length:
            import time
            if cur_len != total_len:
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
                print("number of rounds: ", len(rounds) - 1)
                print("rounds: ", rounds[:-1])
                print("prompt: ", prompt)
                print(labels)
                print(input_ids)
                time.sleep(5)
                labels[:] = IGNORE_INDEX
        # print("number of rounds: ", len(rounds) - 1)
        # print("rounds: ", rounds[:-1])
        # print("prompt: ", prompt)
        # print(labels)
        # print(input_ids)
        return labels

    def _make_masks(self, labels, tokenizer, sep, eos_token_length, rounds):
        cur_len = 0  # bos
        eos_token_length = 1
        bos_token_length = 1
        labels[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            rou = rou + "<|endoftext|>"
            target = rou
            round_len = len(self.tokenizer_image_token(rou, tokenizer))  # 180+4-1
            instruction_len = len(self.tokenizer_image_token(parts[0], tokenizer)) - 1  # 46-1-1
            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                print(111111)
                round_len -= 1
                instruction_len -= 1
            # print("len: ", round_len, instruction_len, cur_len, labels.shape)
            labels[cur_len: cur_len + instruction_len] = IGNORE_INDEX  # [1:1+44]
            cur_len = cur_len + round_len  # 1+183
            # cur_len = len(self.tokenizer_image_token(parts[0] + target, tokenizer))

        labels[cur_len:] = IGNORE_INDEX
        return labels, cur_len







