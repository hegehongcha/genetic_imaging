from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union
from packaging import version

from .formatter import EmptyFormatter, StringFormatter
from .base import Template
from .formatter import Formatter
from . import register_template
from ...utils.constants import *

from transformers import PreTrainedTokenizer
import torch
import tokenizers

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."


@register_template('llama')
@dataclass
class LlamaTemplate(Template):
    format_image_token: "Formatter" = StringFormatter(slot="{{content}}")
    format_user: "Formatter" = StringFormatter(slot="USER" + ": " + "{{content}}" + " ")
    format_assistant: "Formatter" = StringFormatter(slot="ASSISTANT" + ": " + "{{content}}" + "</s>")
    system: "Formatter" = EmptyFormatter(slot=system + " ")
    separator: "Formatter" = EmptyFormatter(slot=[' ASSISTANT: ', '</s>'])

    def _make_masks(self, labels, tokenizer, sep, eos_token_length, rounds):
        cur_len = 1  # bos
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
            rou = rou + "</s>"
            target = rou
            round_len = len(self.tokenizer_image_token(rou, tokenizer)) - bos_token_length  # 180+4-1
            instruction_len = len(self.tokenizer_image_token(parts[0], tokenizer)) - 1 - bos_token_length  # 46-1-1
            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                print(111111)
                round_len -= 1
                instruction_len -= 1
            # print("len: ", round_len, instruction_len, cur_len, labels.shape)
            labels[cur_len: cur_len + instruction_len] = IGNORE_INDEX  # [1:1+44]
            cur_len = cur_len + round_len + 1  # 1+183
            # cur_len = len(self.tokenizer_image_token(parts[0] + target, tokenizer))

        labels[cur_len:] = IGNORE_INDEX
        return labels, cur_len








