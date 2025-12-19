import os, sys, time
sys.path.append('/nfs_share/kunzhao/TinyLLaVA_Factory-main')
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Sampler
from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
import numpy as np
import torch.distributed as dist
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from ..utils.train_utils import *
from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *
from .tinyllava_trainer import LLaVATrainer


class SexClassificationNeural(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_layers=3, dropout=0.1, num_classes=2):
        super(SexClassificationNeural, self).__init__()

        layers = []
        current_dim = input_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)  # Use LayerNorm instead of BatchNorm for better distributed training compatibility
            ])
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SexClassificationTrainer(LLaVATrainer):
    def __init__(self, model, sex_classifier, **kwargs):
        super().__init__(model=model, **kwargs)
        self.sex_classifier = sex_classifier
        self.sex_criterion = nn.CrossEntropyLoss()

        # Track best accuracy for saving best model
        self.best_accuracy = 0.0

        # Add sex classifier as a submodule to the model for DeepSpeed compatibility
        if not hasattr(model, 'sex_classifier'):
            # Ensure the sex classifier is on the same device as the model
            sex_classifier = sex_classifier.to(model.device)
            model.add_module('sex_classifier', sex_classifier)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        data_dict = inputs
        sex_targets = data_dict.pop('sex_targets', None)
        # Move input data to vision model's device
        input_ids = data_dict["input_ids"]
        images = data_dict["images"].squeeze(1)
        attention_mask = data_dict["attention_mask"]
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,  # Don't compute LLM loss
            images=images,
            output_hidden_states=True
        )

        if sex_targets is not None:
            hidden_states = outputs.hidden_states[-1]
            last_token_indices = (input_ids != -100).sum(dim=1) - 1
            last_token_states = hidden_states[torch.arange(hidden_states.size(0)), last_token_indices]

            sex_logits = self.sex_classifier(last_token_states)
            sex_loss = self.sex_criterion(sex_logits, sex_targets.long())

            # Remove frequent logging during training steps

            if return_outputs:
                return sex_loss, outputs
            return sex_loss
        else:
            # If no sex targets, return zero loss
            zero_loss = torch.tensor(0.0, requires_grad=True, device=model.device)
            if return_outputs:
                return zero_loss, outputs
            return zero_loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        # Only print on main process to avoid duplicate output
        if self.is_world_process_zero():
            print(f"Starting {metric_key_prefix} evaluation...")

        # Temporarily disable training mode
        self.model.eval()
        self.sex_classifier.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for step, inputs in enumerate(eval_dataloader):
                inputs = self._prepare_inputs(inputs)

                if 'sex_targets' in inputs:
                    # Forward pass for hidden states
                    outputs = self.model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        images=inputs['images'].squeeze(1),
                        output_hidden_states=True
                    )

                    # Get last token hidden states
                    hidden_states = outputs.hidden_states[-1]
                    last_token_indices = (inputs['input_ids'] != -100).sum(dim=1) - 1
                    last_token_states = hidden_states[torch.arange(hidden_states.size(0)), last_token_indices]

                    # Predict sex
                    sex_logits = self.sex_classifier(last_token_states)
                    sex_targets = inputs['sex_targets'].long()

                    # Calculate loss and accuracy
                    loss = self.sex_criterion(sex_logits, sex_targets)
                    predictions = torch.argmax(sex_logits, dim=1)
                    correct = (predictions == sex_targets).sum().item()

                    total_loss += loss.item() * len(sex_targets)
                    total_correct += correct
                    total_samples += len(sex_targets)

        # Synchronize metrics across all processes for distributed training
        if dist.is_initialized():
            # Convert to tensors for all_reduce
            total_loss_tensor = torch.tensor(total_loss, device=self.model.device)
            total_correct_tensor = torch.tensor(total_correct, device=self.model.device)
            total_samples_tensor = torch.tensor(total_samples, device=self.model.device)

            # Sum across all processes
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

            # Convert back to Python scalars
            total_loss = total_loss_tensor.item()
            total_correct = total_correct_tensor.item()
            total_samples = total_samples_tensor.item()

        # Calculate average metrics
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        # Return to training mode
        self.model.train()
        self.sex_classifier.train()

        metrics = {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_accuracy": accuracy,
        }

        # Save best model based on accuracy - only on main process
        if metric_key_prefix == "eval" and self.is_world_process_zero():
            self.save_best_model(accuracy)
            # Save trainer state after evaluation is complete
            self.save_trainer_state()

        self.log(metrics)
        return metrics

    def create_optimizer(self):
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            # Get sex classifier parameter names (now part of the model)
            sex_classifier_parameters = [name for name, _ in opt_model.named_parameters() if "sex_classifier" in name]

            if self.args.mm_projector_lr is not None:
                connector_parameters = [name for name, _ in opt_model.named_parameters() if "connector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n not in connector_parameters and n not in sex_classifier_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "name": "decay_no_connector_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n not in connector_parameters and n not in sex_classifier_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "name": "no_decay_no_connector_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n in connector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                        "name": "decay_connector_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n in connector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                        "name": "no_decay_proj_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n in sex_classifier_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": getattr(self.args, 'sex_classifier_lr', self.args.learning_rate),
                        "name": "sex_classifier_parameters"
                    }
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n not in sex_classifier_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "name": "decay_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n not in sex_classifier_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "name": "no_decay_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n in sex_classifier_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": getattr(self.args, 'sex_classifier_lr', self.args.learning_rate),
                        "name": "sex_classifier_parameters"
                    }
                ]

            if getattr(self.args, "moe_enable", False):
                from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
                optimizer_grouped_parameters = split_params_into_different_moe_groups_for_optimizer(optimizer_grouped_parameters)

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes
                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        # Only save the sex classifier, not the LLM
        if output_dir is None:
            output_dir = self.args.output_dir

        import torch
        import pathlib
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        sex_classifier_path = output_dir / "prediction_sex_classifier.pt"
        torch.save(self.sex_classifier.state_dict(), sex_classifier_path)
        print(f"Sex classifier saved to {sex_classifier_path}")

    def save_best_model(self, accuracy):
        # Save the best model based on accuracy
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            import torch
            import pathlib
            output_dir = pathlib.Path(self.args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save the best sex classifier
            best_model_path = output_dir / "best_sex_classifier.pt"
            torch.save(self.sex_classifier.state_dict(), best_model_path)

            print(f"New best sex classifier saved with accuracy {accuracy:.4f} to {best_model_path}")
            return True
        return False

    def save_trainer_state(self):
        # Save trainer state (called after each epoch)
        import json
        import pathlib
        output_dir = pathlib.Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        trainer_state_path = output_dir / "trainer_state.json"
        trainer_state = {
            "best_accuracy": self.best_accuracy,
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "log_history": self.state.log_history,
            "train_batch_size": self.args.train_batch_size,
            "learning_rate": self.args.learning_rate,
            "num_train_epochs": self.args.num_train_epochs,
            "total_flos": self.state.total_flos,
            "trial_name": self.state.trial_name,
            "trial_params": self.state.trial_params,
        }
        with open(trainer_state_path, 'w') as f:
            json.dump(trainer_state, f, indent=2)

        print(f"Trainer state saved to {trainer_state_path} after epoch {self.state.epoch}")

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # Override to prevent saving the full model - only save if explicitly called
        pass

