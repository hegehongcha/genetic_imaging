from packaging import version
import pathlib

import tokenizers
import transformers

import sys
sys.path.append('/nfs_share/kunzhao/TinyLLaVA_Factory-main')
from tinyllava.train.age_prediction_trainer import AgePredictionTrainer, AgePredictioinNeural
from tinyllava.training_recipe import TrainingRecipeFactory
from tinyllava.utils import *
from tinyllava.model import *
from tinyllava.data.dataset import make_supervised_data_module, make_gene_data_module, make_gene_roi_data_module

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

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
    llm_args['attn_implementation'] = model_arguments.attn_implementation
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


def train():

    # load argument
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_arguments, data_arguments, training_arguments = parser.parse_args_into_dataclasses()

    logger_setting(getattr(training_arguments, 'output_dir', None))

    training_recipe = TrainingRecipeFactory(training_arguments.training_recipe)(training_arguments)
    # model_args contain arguements for huggingface model .from_pretrained function
    model_args = load_settings(model_arguments, data_arguments, training_arguments)
    model_args = training_recipe.add_args(model_args)
    model_config = TinyLlavaConfig()
    model_config.load_from_config(model_arguments)
    model = TinyLlavaForConditionalGeneration(model_config)

    # load pretrained checkpoint
    if training_arguments.pretrained_model_path is not None:
        model, _, _, _ = load_pretrained_model(model_name_or_path="/home/httang/project/llava/TinyLLaVA_Factory-main/gene_checkpoints/llava_factory/tiny-llava-Qwen2.5-7B-medroi3d-base-gene-finetune-v2-img-preprocess-5000-llm-frozen-vit-mixed-roi/")
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

    # Create age prediction neural network
    # Get the hidden size from the model config
    hidden_size = model.config.text_config.hidden_size if hasattr(model.config, 'text_config') else model.config.hidden_size
    age_predictor = AgePredictioinNeural(
        input_dim=hidden_size,
        hidden_dim=getattr(training_arguments, 'age_predictor_hidden_dim', 512),
        num_layers=getattr(training_arguments, 'age_predictor_num_layers', 3),
        dropout=getattr(training_arguments, 'age_predictor_dropout', 0.1)
    )

    data_module = make_gene_roi_data_module(tokenizer=tokenizer,
                                              data_args=data_arguments)

    # Create test dataset for evaluation using evaluation_data_path argument
    from tinyllava.data.dataset import GeneROISupervisedDataset


    eval_dataset = GeneROISupervisedDataset(
        data_path=data_arguments.evaluation_data_path,
        tokenizer=tokenizer,
        data_args=data_arguments
    )


    log_trainable_params(model)

    # Update data_module with evaluation dataset if available
    if eval_dataset:
        data_module['eval_dataset'] = eval_dataset

    trainer = AgePredictionTrainer(
        model=model,
        age_predictor=age_predictor,
        processing_class=tokenizer,
        args=training_arguments,
        **data_module
    )

    # Train age prediction model
    trainer.train(resume_from_checkpoint=training_arguments.resume_from_checkpoint)

    # Final evaluation on test dataset (only on main process)
    if eval_dataset:
        eval_results = trainer.evaluate(eval_dataset=eval_dataset)
        print(f"Age prediction test results: {eval_results}")
    else:
        print("No evaluation dataset provided")

#     # Only save the age predictor neural network (only on main process)
#     # if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
    import torch
    age_predictor_path = pathlib.Path(training_arguments.output_dir) / "age_predictor.pt"
    torch.save(age_predictor.state_dict(), age_predictor_path)
    print(f"Age predictor saved to {age_predictor_path}")

    # Synchronize all processes before LoRA saving to avoid DeepSpeed parameter conflicts
    # if hasattr(trainer, 'accelerator') and trainer.accelerator.distributed_type is not None:
    #     trainer.accelerator.wait_for_everyone()
    # elif torch.distributed.is_initialized():
    #     torch.distributed.barrier()

    # If using LoRA training recipe, also save the MLLM (only on main process)
    # if training_arguments.training_recipe == "lora" and (trainer.args.local_rank == 0 or trainer.args.local_rank == -1):
    #     # Remove age predictor from model before saving LoRA (training recipe only saves LLM)
    #     if hasattr(model, 'age_predictor'):
    #         delattr(model, 'age_predictor')
    #     training_recipe.save(model, trainer)

if __name__ == "__main__":
    train()