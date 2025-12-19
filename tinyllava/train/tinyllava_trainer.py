import os, sys
sys.path.append('~/project/llava/TinyLLaVA_Factory-main')
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Sampler
from trl.trainer import DPOTrainer, GRPOTrainer
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    # ShardedDDPOption,
    logger,
)
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from typing import List, Optional
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from packaging import version
from ..utils.train_utils import *
from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    assert len(mm_indices) > 0, "Should have at least one multimodal sample."
    assert len(lang_indices) > 0, "Should have at least one language sample."

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) >= megabatch_size:
        megabatches = [additional_batch[:megabatch_size]] + megabatches
        additional_batch = additional_batch[megabatch_size:]

    if len(additional_batch) > 0:
        megabatches.append(additional_batch)

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
        #     return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                connector_parameters = [name for name, _ in opt_model.named_parameters() if "connector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in connector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "name": "decay_no_connector_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in connector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "name": "no_decay_no_connector_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in connector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                        "name": "decay_connector_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in connector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                        "name": "no_decay_proj_parameters"
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "name": "decay_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "name": "no_decay_parameters"
                    },
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


class LLaVADPOTrainer(DPOTrainer):

    def dpo_loss(
            self,
            policy_chosen_logps: torch.FloatTensor,
            policy_rejected_logps: torch.FloatTensor,
            reference_chosen_logps: torch.FloatTensor,
            reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if self.reference_free:
            ref_logratios = torch.tensor([0], dtype=pi_logratios.dtype, device=pi_logratios.device)
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        pi_logratios = pi_logratios.to(self.accelerator.device)
        ref_logratios = ref_logratios.to(self.accelerator.device)
        logits = pi_logratios - ref_logratios
        # print(f"pi log ratios: {pi_logratios}")
        # print(f"ref log ratios: {ref_logratios}")
        # print(f"logits: {logits}")
        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            losses = (
                    -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                    - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "kto_pair":
            # eqn (7) of the HALOs paper
            chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
            losses = torch.cat(
                (
                    1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']"
            )

        chosen_rewards = (
                self.beta
                * (
                        policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(
                    self.accelerator.device)
                ).detach()
        )
        rejected_rewards = (
                self.beta
                * (
                        policy_rejected_logps.to(self.accelerator.device)
                        - reference_rejected_logps.to(self.accelerator.device)
                ).detach()
        )

        return losses, chosen_rewards, rejected_rewards

    @staticmethod
    def get_batch_logps(
            logits: torch.FloatTensor,
            labels: torch.LongTensor,
            average_log_prob: bool = True,
            label_pad_token_id: int = -100,
            is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        # if not is_encoder_decoder:
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != -100

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == -100] = 0
        # print("logits: ", logits, logits.shape)
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        # print("per_token_logps: ", per_token_logps, per_token_logps.shape)
        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def get_sft_loss(self, logits, labels):
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        # Enable model/pipeline parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        return loss

    def concatenated_forward(
            self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        # win_input_ids = batch.pop('chosen_input_ids')
        # rej_input_ids = batch.pop('rejected_input_ids')
        #
        # images = batch.pop('concatenated_images')
        #
        # concatenated_input_ids = batch.pop('concatenated_input_ids')
        # concatenated_labels = batch.pop('concatenated_labels')
        # concatenated_attention_mask = batch.pop('concatenated_attention_mask')
        # concatenated_images = images
        len_chosen = batch["chosen_labels"].shape[0]
        # print(batch["concatenated_input_ids"].shape)
        # print(batch["chosen_labels"].shape, batch["rejected_labels"].shape)
        output, new_labels = model(
            batch["concatenated_input_ids"],
            attention_mask=batch["concatenated_attention_mask"],
            labels=batch["concatenated_labels"],
            images=batch["concatenated_images"],
            use_cache=False,
            dpo_forward=True,
        )
        all_logits = output.logits
        all_logits = all_logits.to(torch.float32)
        all_logps = self.get_batch_logps(
            all_logits,
            new_labels,
            average_log_prob=self.loss_type == "ipo",
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )
        # print("all_logps: ", all_logps.shape)

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        # don't count image embeds logits
        # loss_mask = new_labels != -100
        # logits = [all_logits[i][loss_mask[i]] for i in range(loss_mask.shape[0])]
        # chosen_logits = logits[:len_chosen]
        # rejected_logits = logits[len_chosen:]
        # chosen_logits = [l.detach().cpu().mean() for l in chosen_logits]
        # rejected_logits = [l.detach().cpu().mean() for l in rejected_logits]
        # chosen_logits = sum(chosen_logits)/len_chosen
        # rejected_logits = sum(rejected_logits)/len_chosen

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        chosen_labels = new_labels[:len_chosen]
        rejected_labels = new_labels[len_chosen:]
        # print("new_labels: ", new_labels.shape)
        # print("chosen_logits: ", chosen_logits.shape, chosen_logits)
        # print("rejected_logits: ", rejected_logits.shape, rejected_logits)
        # print("chosen_logps: ", chosen_logps.shape, chosen_logps)
        # print("rejected_logps: ", rejected_logps.shape, rejected_logps)
        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_labels, rejected_labels)

    def get_batch_loss_metrics(
            self,
            model,
            batch: Dict[str, Union[List, torch.LongTensor]],
            train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test.
        CHANGE: 1. add sft loss
        2. all gather metrics
        """

        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            chosen_labels,
            rejected_labels,
        ) = self.concatenated_forward(model, batch)

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                        ) = self.concatenated_forward(self.model, batch)[:2]
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                    ) = self.concatenated_forward(self.ref_model, batch)[:2]

        unscaled_dpo_losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        unscaled_dpo_losses = unscaled_dpo_losses.mean()
        dpo_losses = unscaled_dpo_losses * self.args.dpo_alpha
        # unscaled_sft_loss = self.get_sft_loss(policy_chosen_logits, chosen_labels)
        # sft_loss = unscaled_sft_loss * self.gamma

        # print(sft_loss.shape, dpo_losses.shape)
        # losses = dpo_losses + sft_loss
        # losses = sft_loss # sft only
        losses = dpo_losses # dpo only
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        def all_gather_tensor(tensor):
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                tensor = tensor.detach()
                gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(gathered_tensor, tensor)
                tensor = torch.cat(gathered_tensor, dim=0)
            # else:
            #     print('not distributed')
            return tensor

        # gather chosen_rewards across devices
        chosen_rewards = all_gather_tensor(chosen_rewards)
        rejected_rewards = all_gather_tensor(rejected_rewards)
        reward_accuracies = all_gather_tensor(reward_accuracies)
        policy_chosen_logps = all_gather_tensor(policy_chosen_logps)
        policy_rejected_logps = all_gather_tensor(policy_rejected_logps)
        reference_chosen_logps = all_gather_tensor(reference_chosen_logps)
        reference_rejected_logps = all_gather_tensor(reference_rejected_logps)

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}losses/dpo"] = unscaled_dpo_losses.cpu()
        # metrics[f"{prefix}losses/sft"] = unscaled_sft_loss.cpu()
        metrics[f"{prefix}losses/total"] = losses.cpu()
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        # policy logps
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        # policy logits (exclude image tokens)
        # metrics[f"{prefix}logits/rejected"] =policy_rejected_logits
        # metrics[f"{prefix}logits/chosen"] = policy_chosen_logits
        # reference logps
        metrics[f"{prefix}ref_logps/rejected"] = reference_rejected_logps.mean().cpu()
        metrics[f"{prefix}ref_logps/chosen"] = reference_chosen_logps.mean().cpu()

        # metrics all pick .4 digits
        # for k in metrics:
        #     metrics[k] = round(metrics[k].item(), 4)

        return losses, metrics

    def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None
    ):
        # print(1111)
        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss


def pad_sequence(sequences, batch_first=False, padding_value=0.0, padding_side='right'):
    """
    带 padding_side 的 pad_sequence 实现

    参数:
        sequences (list[Tensor]): list of Tensors with shape [seq_len, *]
        batch_first (bool): 输出是否为 [batch_size, max_len, *]，默认 False
        padding_value (float): 用于填充的值
        padding_side (str): 'right'（默认）或 'left'，决定填充在序列的哪一边

    返回:
        Tensor: padded tensor of shape [max_len, batch_size, *] or [batch_size, max_len, *]
    """
    assert padding_side in ('left', 'right'), "padding_side 只能是 'left' 或 'right'"

    max_len = max(seq.size(0) for seq in sequences)
    trailing_dims = sequences[0].size()[1:]  # 除了 seq_len 以外的维度
    size = (len(sequences), max_len) + trailing_dims
    out_tensor = sequences[0].new_full(size, padding_value)

    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        if padding_side == 'right':
            out_tensor[i, :length, ...] = tensor
        else:  # padding_side == 'left'
            out_tensor[i, -length:, ...] = tensor

    return out_tensor if batch_first else out_tensor.transpose(0, 1)


class LLaVAGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # 必须有这行

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"
        # print(111, self._buffered_inputs)
        # print(222, self._step)
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0 or self._step < self.args.gradient_accumulation_steps:
                inputs = self._generate_and_score_completions(inputs)
                # print(inputs)
                self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
            else:
                inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
        return inputs

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, images=None):
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(input_ids=input_ids, attention_mask=attention_mask, images=images).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        # Divide logits by sampling temperature.
        # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
        logits = logits / self.temperature
        if logits.dtype in [torch.float32, torch.float64]:
            selected_logits = torch.gather(logits, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
            # loop to reduce peak mem consumption
            logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
            per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
        else:
            # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
            per_token_logps = []
            for row_logits, row_labels in zip(logits, input_ids):  # loop to reduce peak mem consumption
                row_logps = F.log_softmax(row_logits, dim=-1)
                row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
                per_token_logps.append(row_per_token_logps)
            per_token_logps = torch.stack(per_token_logps)
        return per_token_logps
        # return selective_log_softmax(logits, input_ids)  # compute logprobs for the input tokens

    def _generate_and_score_completions(
            self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompt_ids, prompt, ground_truth = tuple([instance[key] for instance in inputs]
                                                 for key in ("input_ids", "prompt", "ground_truth"))
        prompt_ids = pad_sequence(
            prompt_ids,
            padding_value=self.processing_class.pad_token_id,
            batch_first=True,
            padding_side='left')
        prompt_ids = prompt_ids[:, :self.processing_class.model_max_length]
        prompt_mask = prompt_ids.ne(self.processing_class.pad_token_id)
        if 'image' in inputs[0]:
            images_list = [instance['image'] for instance in inputs]
            if all(x is not None and x.shape == images_list[0].shape for x in images_list):
                images = torch.stack(images_list)
            else:
                images = images_list

        if self.args.use_vllm:
            # print(1111)
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step
            # print(input_ids.shape)
            with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    _,
                    inputs_embeds,
                    _
                ) = unwrapped_model.prepare_inputs_labels_for_multimodal(
                    prompt_ids,
                    None,
                    None,
                    None,
                    None,
                    images=images,
                    image_sizes=None
                )

            # Step 1: gather embeddings from all processes to rank 0
            # inputs_embeds = [e.squeeze().detach().cpu().numpy() for e in inputs_embeds]  # safety first
            # print("before gathered: ", inputs_embeds.shape)
            all_prompt_embeds = gather(inputs_embeds)
            # print("after gathered: ", all_prompt_embeds_.shape)
            # all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                # all_prompt_embeds = [arr.detach().cpu().numpy() for arr in all_prompt_embeds_]
                # all_prompt_embeds_ = [t for t in all_prompt_embeds]  # 移动到 CPU
                # np.save("/home/httang/project/llava/TinyLLaVA_Factory-main/tinyllava/train/embedding.npy", all_prompt_embeds, allow_pickle=True)
                torch.save({'prompt_embeds': all_prompt_embeds}, "/home/httang/project/llava/TinyLLaVA_Factory-main/tinyllava/train/embeds.pt")
                completion_ids = self.vllm_client.generate(prompt_embeds_file="/home/httang/project/llava/TinyLLaVA_Factory-main/tinyllava/train/embeds.pt",
                                                        prompts=[],
                                                        n=self.num_generations,
                                                        repetition_penalty=self.repetition_penalty,
                                                        temperature=self.temperature,
                                                        top_p=self.top_p,
                                                        top_k=-1 if self.top_k is None else self.top_k,
                                                        min_p=0.0 if self.min_p is None else self.min_p,
                                                        max_tokens=self.max_completion_length,
                                                        guided_decoding_regex=self.guided_decoding_regex,
                                                        )
                # print("generated finished!!!")
            else:
                completion_ids = [None] * (len(all_prompt_embeds)*self.num_generations)
                # completion_ids = [None] * 36
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad_sequence(completion_ids, padding_value=self.processing_class.pad_token_id, batch_first=True, padding_side='left')
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    _,
                    inputs_embeds,
                    _
                ) = unwrapped_model.prepare_inputs_labels_for_multimodal(
                    prompt_ids,
                    None,
                    None,
                    None,
                    None,
                    images=images,
                    image_sizes=None
                )
                unwrapped_model.language_model.gradient_checkpointing_disable()
                generate_returned_result = unwrapped_model.language_model.generate(
                    tokenizer=self.processing_class,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    do_sample=True,
                    num_beams=1,
                    temperature=0.7,
                    max_new_tokens=4096,
                    stop_strings="</s>",
                    use_cache=False,
                    pad_token_id=self.processing_class.eos_token_id
                )
                unwrapped_model.language_model.gradient_checkpointing_enable()
                # generate_returned_result = unwrapped_model.generate(
                #     prompt_ids, images=images, attention_mask=prompt_mask, generation_config=self.generation_config
                # )

            # Compute prompt length and extract completion ids
            # prompt_length = prompt_ids.size(1)
            # prompt_ids = prompt_completion_ids[:, :prompt_length]
            # completion_ids = prompt_completion_ids[:, prompt_length:]
            completion_ids = generate_returned_result
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep, images
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep, images
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep, images
                    )

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        # print("prompts: ", prompts)
        # print("completions_text: ", completions_text)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = f"reward {reward_func.config._name_or_path.split('/')[-1]}"
            else:
                reward_func_name = reward_func.__name__
            # with profiling_context(self, reward_func_name):
            if isinstance(
                    reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)

        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[mode][f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "images": images,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model
        # print(111, inputs)
        # print(222, self._step)
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        images = inputs["images"]
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep, images=images)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                    torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "eval" if self.control.should_evaluate else "train"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics[mode].clear()

    def create_model_card(
            self,
            model_name: Optional[str] = None,
            dataset_name: Optional[str] = None,
            tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))







