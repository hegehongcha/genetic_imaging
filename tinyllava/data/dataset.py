import copy
from dataclasses import dataclass
import json
from typing import Dict,  Sequence, TYPE_CHECKING
from PIL import Image, ImageFile
import os

from .text_preprocess import TextPreprocess
from .image_preprocess import ImagePreprocess
from ..utils.arguments import DataArguments
from ..utils.constants import *
import nibabel as nib
import numpy as np
import transformers
import torch
from torch.utils.data import Dataset



ImageFile.LOAD_TRUNCATED_IMAGES = True

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.text_preprocess = TextPreprocess(tokenizer, data_args.conv_version)
        self.image_preprocess = ImagePreprocess(data_args.image_processor, data_args)

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        data_dict = self.text_preprocess(copy.deepcopy(sources["conversations"]))
        if 'image' in sources:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            image = self.image_preprocess(image)
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            # print(f'{i}:{sources}')
            crop_size = getattr(self.data_args.image_processor, 'crop_size', getattr(self.data_args.image_processor, 'size'))
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = labels[:, :self.tokenizer.model_max_length]
        # FIXME: This is a hack for handling phi and stablelm, as they have the same eos, pad and unk. We want the model
        # FIXME: to predict the eos in the input ids, but we also use the id of eos to pad sequence, so we use a temp
        # FIXME: eos id first, and convert them back.
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def concate_pad(tensorA, tensorB, padding_value):
    out = torch.nn.utils.rnn.pad_sequence(
        list(tensorA) + list(tensorB),
        batch_first=True,
        padding_value=padding_value)
    return out


class DPODataset(Dataset):
    """Dataset for DPO fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(DPODataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.text_preprocess = TextPreprocess(tokenizer, data_args.conv_version)
        self.image_preprocess = ImagePreprocess(data_args.image_processor, data_args)

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        chosen_data_dict = self.text_preprocess(copy.deepcopy(sources["chosen_conversations"]))
        rejected_data_dict = self.text_preprocess(copy.deepcopy(sources["rejected_conversations"]))
        data_dict = {}
        data_dict['chosen_input_ids'] = chosen_data_dict["input_ids"]
        data_dict["rejected_input_ids"] = rejected_data_dict["input_ids"]
        data_dict["chosen_labels"] = chosen_data_dict["labels"]
        data_dict["rejected_labels"] = rejected_data_dict["labels"]
        if 'image' in sources:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            image = self.image_preprocess(image)
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            # print(f'{i}:{sources}')
            crop_size = getattr(self.data_args.image_processor, 'crop_size', getattr(self.data_args.image_processor, 'size'))
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        data_dict['chosen_image'] = data_dict['image']
        data_dict['rejected_image'] = data_dict['image']
        return data_dict


def center_pad(image, target_shape=(130, 140, 140)):
    """
    将 3D 图像居中零填充到 target_shape。
    image: torch.Tensor of shape (1, D, H, W)
    """
    current_shape = image.shape[-3:]
    padding = []
    for i in range(3):
        total_pad = target_shape[i] - current_shape[i]
        if total_pad < 0:
            raise ValueError(f"当前图像在第{i}维比目标大，建议先裁剪。")
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        padding.extend([pad_before, pad_after])  # 注意顺序
    # padding: [W_before, W_after, H_before, H_after, D_before, D_after]
    return torch.nn.functional.pad(image, padding[::-1], mode='constant', value=0)


def crop_nonzero_region(image: np.ndarray):
    """裁剪三维图像中的非零区域（去除空白）"""
    assert image.ndim == 3, "图像必须是三维的"

    # 找到非零体素的范围
    nonzero = np.nonzero(image)
    x_min, x_max = np.min(nonzero[0]), np.max(nonzero[0])
    y_min, y_max = np.min(nonzero[1]), np.max(nonzero[1])
    z_min, z_max = np.min(nonzero[2]), np.max(nonzero[2])

    # 裁剪
    cropped = image[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
    return cropped


def clip_and_rescale(image: np.ndarray, lower_percentile=0.5, upper_percentile=99.5):
    """Clip to [0.5%, 99.5%] percentile and rescale to [0, 255]"""
    lower = np.percentile(image, lower_percentile)
    upper = np.percentile(image, upper_percentile)
    # print(lower, upper)
    # print(image.shape)
    # Clip values
    clipped = np.clip(image, lower, upper)
    # print(clipped.shape)
    # Rescale to [0, 255]
    rescaled = (clipped - lower) / (upper - lower + 1e-8) * 255.0
    return rescaled.astype(np.float32)


def process_single_image(image, target_shape=(97, 103, 89)):
    data = image
    # print(data.shape)
    # Step 1: Crop nonzero
    cropped = crop_nonzero_region(data)
    # print(cropped.shape)
    # Step 2: Clip & Rescale to [0, 255]
    normalized = clip_and_rescale(cropped)

    # Step 3: Padding
    # padded = center_pad(normalized, target_shape)

    return normalized


class GeneSupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(GeneSupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.text_preprocess = TextPreprocess(tokenizer, data_args.conv_version)
        # self.image_preprocess = ImagePreprocess(data_args.image_processor, data_args)

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        data_dict = self.text_preprocess(copy.deepcopy(sources["conversations"]))
        if 'image' in sources:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            if image_file != None:
                image = nib.load(os.path.join(image_folder, image_file)).get_fdata()
                image = process_single_image(image)
                image = torch.tensor(image).unsqueeze(0).float()
                image = center_pad(image, target_shape=(97, 103, 89))
            else:
                image = torch.zeros((1, 97, 103, 89))
            # image = self.image_preprocess(image)
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            # print(f'{i}:{sources}')
            crop_size = getattr(self.data_args.image_processor, 'crop_size', getattr(self.data_args.image_processor, 'size'))
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        if "sex" in sources:
            data_dict["sex"] = sources["sex"]
        if "age" in sources:
            data_dict["age"] = sources["age"]
        return data_dict


class GRPODataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(GRPODataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.text_preprocess = TextPreprocess(tokenizer, data_args.conv_version)
        self.image_preprocess = ImagePreprocess(data_args.image_processor, data_args)

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        sources_copy = copy.deepcopy(sources["conversations"])
        answer = sources["conversations"][1]['value']
        sources_copy[1]['value'] = None
        data_dict = self.text_preprocess(sources_copy, mode="eval")
        data_dict["ground_truth"] = answer
        if 'image' in sources:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            image = self.image_preprocess(image)
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            # print(f'{i}:{sources}')
            crop_size = getattr(self.data_args.image_processor, 'crop_size', getattr(self.data_args.image_processor, 'size'))
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


class GeneGRPODataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(GeneGRPODataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.text_preprocess = TextPreprocess(tokenizer, data_args.conv_version)
        self.image_preprocess = ImagePreprocess(data_args.image_processor, data_args)

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        sources_copy = copy.deepcopy(sources["conversations"])
        answer = sources["conversations"][1]['value']
        sources_copy[1]['value'] = None
        data_dict = self.text_preprocess(sources_copy, mode="eval")
        data_dict["ground_truth"] = answer
        if 'image' in sources:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            image = nib.load(os.path.join(image_folder, image_file)).get_fdata()
            image = torch.tensor(image).unsqueeze(0).float()
            image = center_pad(image, target_shape=(130, 140, 140))
            # image = self.image_preprocess(image)
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            # print(f'{i}:{sources}')
            crop_size = getattr(self.data_args.image_processor, 'crop_size', getattr(self.data_args.image_processor, 'size'))
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


class GeneROISupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(GeneROISupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        self.embed_dict = torch.load("/home/httang/project/llava/TinyLLaVA_Factory-main/genetic_dataset/roi_embeds.pt")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.text_preprocess = TextPreprocess(tokenizer, data_args.conv_version)
        # self.image_preprocess = ImagePreprocess(data_args.image_processor, data_args)

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        data_dict = self.text_preprocess(copy.deepcopy(sources["conversations"]))
        if 'image' in sources:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            if image_file != None:
                idx = image_file.split("/")[0]
                image = self.embed_dict[idx]
                data_dict["image"] = image.squeeze(0)
            else:
                example = self.embed_dict["099_S_0090_I134528"]
                example = example.squeeze(0)
                data_dict["image"] = torch.zeros_like(example)
            # if image_file != None:
            #     img_path = os.path.join(image_folder, image_file)
            #     roi_file = image_file.split("/")[0] + "/aseg-in-rawavg.nii"
            #     roi_path = os.path.join(image_folder, roi_file)
            #     data = nib.load(img_path).get_fdata()
            #     roi = nib.load(roi_path).get_fdata()
            #     nonzero_values = roi[roi > 0]
            #     roi_list = list(nonzero_values)
            #     nonzero_set = set(roi_list)
            #     image_np_list = []
            #     for r in nonzero_set:
            #         specific_mask = (roi == int(r)).astype(np.uint8) 
            #         image_np = data * specific_mask
            #         image_np = process_single_image(image_np)
            #         image_tensor = torch.tensor(image_np).unsqueeze(0).float()
            #         image_tensor = center_pad(image_tensor, target_shape=(97, 103, 89))
            #         volume3d = image_tensor
            #         image_np_list.append(volume3d)                
            #     while len(image_np_list) < 109:
            #         volum_3d_zero_like = torch.zeros_like(image_np_list[0])
            #         image_np_list.append(volum_3d_zero_like)
            # else:
            #     image_np_list = []
            #     while len(image_np_list) < 109:
            #         image = torch.zeros((1, 97, 103, 89))
            #         image_np_list.append(image)
                
            # data_dict['image'] = torch.stack(image_np_list)
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            # print(f'{i}:{sources}')
            crop_size = getattr(self.data_args.image_processor, 'crop_size', getattr(self.data_args.image_processor, 'size'))
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        
        if "sex" in sources:
            data_dict["sex"] = sources["sex"]
        if "age" in sources:
            data_dict["age"] = sources["age"]
        return data_dict
    
    
@dataclass
class DataCollatorForDPODataset(object):
    """Collate examples for DPO fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        chosen_input_ids, chosen_labels = tuple([instance[key] for instance in instances]
                                  for key in ("chosen_input_ids", "chosen_labels"))
        # print(instances)
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in chosen_input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300
        chosen_input_ids = torch.nn.utils.rnn.pad_sequence(
            chosen_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        chosen_labels = torch.nn.utils.rnn.pad_sequence(chosen_labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        chosen_input_ids = chosen_input_ids[:, :self.tokenizer.model_max_length]
        chosen_attention_mask = chosen_input_ids.ne(self.tokenizer.pad_token_id)
        chosen_labels = chosen_labels[:, :self.tokenizer.model_max_length]
        # FIXME: This is a hack for handling phi and stablelm, as they have the same eos, pad and unk. We want the model
        # FIXME: to predict the eos in the input ids, but we also use the id of eos to pad sequence, so we use a temp
        # FIXME: eos id first, and convert them back.
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in chosen_input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        rejected_input_ids, rejected_labels = tuple([instance[key] for instance in instances]
                                  for key in ("rejected_input_ids", "rejected_labels"))
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in rejected_input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300
        rejected_input_ids = torch.nn.utils.rnn.pad_sequence(
            rejected_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        rejected_labels = torch.nn.utils.rnn.pad_sequence(rejected_labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        rejected_input_ids = rejected_input_ids[:, :self.tokenizer.model_max_length]
        rejected_attention_mask = rejected_input_ids.ne(self.tokenizer.pad_token_id)
        rejected_labels = rejected_labels[:, :self.tokenizer.model_max_length]
        # FIXME: This is a hack for handling phi and stablelm, as they have the same eos, pad and unk. We want the model
        # FIXME: to predict the eos in the input ids, but we also use the id of eos to pad sequence, so we use a temp
        # FIXME: eos id first, and convert them back.
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in rejected_input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        concatenated_input_ids = concate_pad(chosen_input_ids, rejected_input_ids, self.tokenizer.pad_token_id)
        concatenated_labels = concate_pad(chosen_labels, rejected_labels, -100)
        concatenated_attention_mask = concatenated_input_ids.ne(self.tokenizer.pad_token_id)
        # concatenated_images = torch.cat([instance["image"], instance["image"]], dim=0) for instance in instances
        batch = dict(
            concatenated_input_ids=concatenated_input_ids,
            concatenated_labels=concatenated_labels,
            concatenated_attention_mask=concatenated_attention_mask,
            chosen_input_ids=chosen_input_ids,
            chosen_labels=chosen_labels,
            chosen_attention_mask=chosen_attention_mask,
            rejected_input_ids=rejected_input_ids,
            rejected_labels=rejected_labels,
            rejected_attention_mask=rejected_attention_mask,
        )
        if 'image' in instances[0]:
            # for instance in instances:
            #     print(instance['image'].shape)
            chosen_images = [instance['chosen_image'] for instance in instances]
            rejected_images = [instance['rejected_image'] for instance in instances]
            images = chosen_images + rejected_images
        if all(x is not None and x.shape == images[0].shape for x in images):
            batch['concatenated_images'] = torch.stack(images)
        else:
            batch['concatenated_images'] = images

        return batch


@dataclass
class DataCollatorForGeneDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = labels[:, :self.tokenizer.model_max_length]
        # FIXME: This is a hack for handling phi and stablelm, as they have the same eos, pad and unk. We want the model
        # FIXME: to predict the eos in the input ids, but we also use the id of eos to pad sequence, so we use a temp
        # FIXME: eos id first, and convert them back.
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        # print(batch)
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        
        # Handle age targets for age prediction
        if 'age' in instances[0]:
            age_targets = [instance['age'] for instance in instances]
            batch['age_targets'] = torch.tensor(age_targets, dtype=torch.float32)

        # Handle sex targets for sex classification
        if 'sex' in instances[0]:
            sex_targets = [instance['sex'] for instance in instances]
            batch['sex_targets'] = torch.tensor(sex_targets, dtype=torch.long)
        return batch


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


@dataclass
class DataCollatorForGRPODataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, prompt, ground_truth= tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "prompt", "ground_truth"))
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300
        input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_side='left')
        # labels = torch.nn.utils.rnn.pad_sequence(labels,
        #                                          batch_first=True,
        #                                          padding_value=IGNORE_INDEX,
        #                                          padding_side='left')
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        # labels = labels[:, :self.tokenizer.model_max_length]
        # FIXME: This is a hack for handling phi and stablelm, as they have the same eos, pad and unk. We want the model
        # FIXME: to predict the eos in the input ids, but we also use the id of eos to pad sequence, so we use a temp
        # FIXME: eos id first, and convert them back.
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch

@dataclass
class DataCollatorForGeneROIDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = labels[:, :self.tokenizer.model_max_length]
        # FIXME: This is a hack for handling phi and stablelm, as they have the same eos, pad and unk. We want the model
        # FIXME: to predict the eos in the input ids, but we also use the id of eos to pad sequence, so we use a temp
        # FIXME: eos id first, and convert them back.
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            
            if all(x is not None and x.shape == images[0].shape for x in images):
                # print(1111)
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        # print(type(batch['images']))
        
        # Handle age targets for age prediction
        if 'age' in instances[0]:
            age_targets = [instance['age'] for instance in instances]
            batch['age_targets'] = torch.tensor(age_targets, dtype=torch.float32)

        # Handle sex targets for sex classification
        if 'sex' in instances[0]:
            sex_targets = [instance['sex'] for instance in instances]
            batch['sex_targets'] = torch.tensor(sex_targets, dtype=torch.long)
        
        return batch    

def make_dpo_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for dpo fine-tuning."""
    train_dataset = DPODataset(tokenizer=tokenizer,
                               data_path=data_args.data_path,
                               data_args=data_args)
    data_collator = DataCollatorForDPODataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def make_gene_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = GeneSupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForGeneDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def make_grpo_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = GRPODataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForGRPODataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def make_gene_grpo_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = GeneGRPODataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForGRPODataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def make_gene_roi_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = GeneROISupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForGeneROIDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)