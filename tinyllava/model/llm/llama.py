from transformers import LlamaForCausalLM, AutoTokenizer

from . import register_llm

@register_llm('llama')
def return_llamaclass():
    def tokenizer_and_post_load(tokenizer):
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
        return tokenizer
    return LlamaForCausalLM, (AutoTokenizer, tokenizer_and_post_load)