from packaging import version
import pathlib

import tokenizers
import transformers

import sys

sys.path.append('/home/httang/project/llava/TinyLLaVA_Factory-main')
from tinyllava.train.tinyllava_trainer import LLaVAGRPOTrainer
from tinyllava.training_recipe import TrainingRecipeFactory
from tinyllava.utils import *
from tinyllava.model import *
from tinyllava.data.dataset import make_gene_grpo_data_module
from trl.trainer import DPOConfig
import re
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')
# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

def load_settings(model_arguments, data_arguments, training_arguments):
    model_arguments.tune_type_connector = training_arguments.tune_type_connector
    model_arguments.tune_type_llm = training_arguments.tune_type_llm
    model_arguments.tune_type_vision_tower = training_arguments.tune_type_vision_tower
    model_arguments.image_aspect_ratio = data_arguments.image_aspect_ratio

    model_args = {}
    model_args['llm'] = _load_llm_settings(model_arguments)
    model_args['vision_tower'] = _load_vision_settings(model_arguments)
    model_args['connector'] = _load_connector_settings(model_arguments)
    return model_args


def _load_llm_settings(model_arguments):
    llm_args = {}
    llm_args['model_name_or_path'] = model_arguments.model_name_or_path
    llm_args['cache_dir'] = model_arguments.cache_dir
    llm_args[
     'attn_implementation'] = model_arguments.attn_implementation  # flash_attention_2 only supports torch.float16 and torch.bfloat16 dtypes
    return llm_args


def _load_vision_settings(model_arguments):
    vision_args = {}
    vision_args['model_name_or_path'] = model_arguments.vision_tower.split(':')[-1]
    if model_arguments.vision_tower2 != '':
        vision_args['model_name_or_path2'] = model_arguments.vision_tower2.split(':')[-1]
    return vision_args


def _load_connector_settings(model_arguments):
    connector_args = {}
    connector_args['connector_type'] = model_arguments.connector_type
    return connector_args


def format_reward_func(completions, **kwargs):
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    rewards = []
    for content in completions:
        # 使用正则表达式检查格式
        format_match = re.search(pattern, content)
        if format_match:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    # format_match = re.fullmatch(pattern, content)
    return rewards
    # """Reward function that checks if the completion has a specific format."""
    # think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    # answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    # boxed_pattern = re.compile(r"\\boxed\{.*?\}", re.DOTALL)
    #
    # rewards = []
    # for content in completions:
    #     # 查找所有 think 和 answer 标签的内容
    #     think_matches = think_pattern.findall(content)
    #     answer_matches = answer_pattern.findall(content)
    #
    #     # 检查是否恰好有一个 think 和一个 answer
    #     if len(think_matches) != 1 or len(answer_matches) != 1:
    #         rewards.append(0.0)
    #         continue
    #
    #     think_content = think_matches[0]
    #
    #     # 检查 think 内部是否含有 answer 标签或 \boxed{}
    #     if "<answer>" in think_content or "</answer>" in think_content or boxed_pattern.search(think_content):
    #         rewards.append(0.0)
    #         continue
    #
    #     rewards.append(1.0)
    #
    # return rewards


iter_count = 0


def reward_func(completions, ground_truth, **kwargs):
    # 匹配 <answer>...</answer>
    answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

    global iter_count
    if iter_count % 40 == 0:
        print(completions)
    iter_count += 1

    contents = []
    for completion in completions:
        answer_match = answer_pattern.search(completion)
        if answer_match:
            # 提取并清理内容
            answer_content = answer_match.group(1).strip()
            contents.append(answer_content)
        else:
            contents.append("")

    # 计算 reward
    return [1.0 if c == gt.strip() else 0.0 for c, gt in zip(contents, ground_truth)]


def train():
    # load argument
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, GRPOTrainingArguments))
    model_arguments, data_arguments, training_arguments = parser.parse_args_into_dataclasses()
    logger_setting(getattr(training_arguments, 'output_dir', None))

    training_recipe = TrainingRecipeFactory(training_arguments.training_recipe)(training_arguments)
    # model_args contain arguements for huggingface model .from_pretrained function
    model_args = load_settings(model_arguments, data_arguments, training_arguments)
    model_args = training_recipe.add_args(model_args)
    use_grpo = True
    model_config = TinyLlavaConfig()
    model_config.load_from_config(model_arguments)
    model = TinyLlavaForConditionalGeneration(model_config)
    # load pretrained checkpoint
    # if use_grpo:
    #     model, _, _, _ = load_pretrained_model(model_name_or_path="xray-llava/llama-3-8b-instruct-xrayclip")
    # else:
    if training_arguments.pretrained_model_path is not None:
        model = training_recipe.load(model, model_args)
    else:
        model.load_llm(**model_args['llm'])
        model.load_vision_tower(**model_args['vision_tower'])
        model.load_connector(**model_args['connector'])


    model = training_recipe(model)
    model.config.use_cache = False
    model.config.image_aspect_ratio = data_arguments.image_aspect_ratio
    tokenizer = model.tokenizer
    data_arguments.image_processor = model.vision_tower._image_processor
    data_arguments.is_multimodal = True
    data_module = make_gene_grpo_data_module(tokenizer=tokenizer,
                                              data_args=data_arguments)
    log_trainable_params(model)  # not work well with zero3
    # trainer = LLaVAGRPOTrainer(model=model,  # does not require model.to(device), huggingface/deepspeed does it for you?
    #                        tokenizer=tokenizer,
    #                        args=training_arguments,
    #                        **data_module)
    # training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO", logging_steps=10)
    trainer = LLaVAGRPOTrainer(
        model=model,
        reward_funcs=[format_reward_func, reward_func],
        processing_class=tokenizer,
        args=training_arguments,
        train_dataset=data_module["train_dataset"])

    trainer.train(resume_from_checkpoint=False)

    training_recipe.save(model, trainer)


if __name__ == "__main__":
    train()
