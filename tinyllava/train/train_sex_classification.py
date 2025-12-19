from packaging import version
import pathlib

import tokenizers
import transformers

import sys
sys.path.append('/nfs_share/kunzhao/TinyLLaVA_Factory-main')
from tinyllava.train.sex_classification_trainer import SexClassificationTrainer, SexClassificationNeural
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
        model, _, _, _ = load_pretrained_model(model_name_or_path="/nfs_share/kunzhao/TinyLLaVA_Factory-main/gene_checkpoints/llava_factory/llama_roi_mixed/")
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

    # Create sex classification neural network
    # Get the hidden size from the model config
    hidden_size = model.config.text_config.hidden_size if hasattr(model.config, 'text_config') else model.config.hidden_size
    sex_classifier = SexClassificationNeural(
        input_dim=hidden_size,
        hidden_dim=getattr(training_arguments, 'sex_classifier_hidden_dim', 512),
        num_layers=getattr(training_arguments, 'sex_classifier_num_layers', 3),
        dropout=getattr(training_arguments, 'sex_classifier_dropout', 0.1),
        num_classes=getattr(training_arguments, 'sex_classifier_num_classes', 2)  # Binary classification by default
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

    trainer = SexClassificationTrainer(
        model=model,
        sex_classifier=sex_classifier,
        processing_class=tokenizer,
        args=training_arguments,
        **data_module
    )

    # Train sex classification model
    trainer.train(resume_from_checkpoint=training_arguments.resume_from_checkpoint)

    # Final evaluation on test dataset (only on main process)
    if eval_dataset:
        eval_results = trainer.evaluate(eval_dataset=eval_dataset)
        print(f"Sex classification test results: {eval_results}")
    else:
        print("No evaluation dataset provided")

    # Only save the sex classifier neural network (only on main process)
    # if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
    import torch
    sex_classifier_path = pathlib.Path(training_arguments.output_dir) / "sex_classifier.pt"
    torch.save(sex_classifier.state_dict(), sex_classifier_path)
    print(f"Sex classifier saved to {sex_classifier_path}")
    # Synchronize all processes before LoRA saving to avoid DeepSpeed parameter conflicts
    # if hasattr(trainer, 'accelerator') and trainer.accelerator.distributed_type is not None:
    #     trainer.accelerator.wait_for_everyone()
    # elif torch.distributed.is_initialized():
    #     torch.distributed.barrier()

    # if training_arguments.training_recipe == "lora" and (trainer.args.local_rank == 0 or trainer.args.local_rank == -1):
    #     # Remove sex classifier from model before saving LoRA (training recipe only saves LLM)
    #     if hasattr(model, 'sex_classifier'):
    #         delattr(model, 'sex_classifier')
    #     training_recipe.save(model, trainer)
if __name__ == "__main__":
    train()