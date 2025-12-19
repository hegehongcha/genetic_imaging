import os
import argparse
import requests
from PIL import Image
from io import BytesIO
import nibabel as nib
import torch
from transformers import TextStreamer
from transformers import StoppingCriteria, StoppingCriteriaList
from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *
import random
import json
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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


model, tokenizer, image_processor, context_len = load_pretrained_model(model_name_or_path="/home/httang/project/llava/TinyLLaVA_Factory-main/gene_checkpoints/5fold/1/no_roi/tiny-llava-Qwen2.5-7B-med3d:Med3d-base-gene-finetune-v2-img-preprocess-5000-frozen-vit-no-roi")
model = model.to(device)
text_processor = TextPreprocess(tokenizer, "qwen2_base")
if getattr(text_processor.template, 'role', None) is None:
    roles = ['USER', 'ASSISTANT']
else:
    roles = text_processor.template.role.apply()
data_args = model.config
image_processor = ImagePreprocess(image_processor, data_args)
data_args = model.config
model.to("cuda")
# print(model)
with open("/home/httang/project/llava/TinyLLaVA_Factory-main/genetic_dataset/5fold/test/gene_test_classification_and_sft_1.json", "r") as f:
    samples = json.load(f)
test_examples = []
print("tiny-llava-Qwen2.5-7B-med3d:Med3d-base-gene-finetune-v2-img-preprocess-5000-frozen-noroi-nomix-5fold-1")
for l, sample in enumerate(samples):
    if l%500 == 0:
        print(l)
    img = "/home/httang/dataset/Image_genetic/Processed_T1/" + sample["image"]
    image = nib.load(img).get_fdata()
    image = process_single_image(image)
    image = torch.tensor(image).unsqueeze(0).float()
    image = center_pad(image, target_shape=(97, 103, 89))
    img = torch.zeros_like(image)
    image_tensor = image.unsqueeze(0).to("cuda")
    # image_tensor = img.unsqueeze(0).to("cuda")
    msg = Message()
    inp = f"{roles[0]}: " + sample["conversations"][0]["value"]
    msg.add_message(inp)
    data_dict = text_processor(msg.messages, mode='eval')
    input_ids = data_dict["input_ids"].unsqueeze(0).to(model.device)
    n = 0
    # model.train()
    # while n < 8:
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            tokenizer=tokenizer,
            images=image_tensor,
            do_sample=False,
            num_beams=1,
            top_p=0.9,
            top_k=100,
            max_new_tokens=10,
            temperature=0.2,
            stop_strings="</s>",
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    output_token = output_ids[0]
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    output_text = tokenizer.decode(output_token, add_special_tokens=False)
    print(l, output_text, "||", sample["conversations"][1]["value"])
    n = n + 1
    example = {}
    example["image"] = sample["image"]
    example["generated_caption"] = output_text
    example["reference"] = sample["conversations"][1]["value"]
    test_examples.append(example)

k = 0
for i, item in enumerate(test_examples):
    # print(i, item["predict"].strip().split(" ")[3])
    p = item["generated_caption"].split(" ")[-1].strip().split(".")[0].strip()
    if "\n" in p:
        predict = p.split("\n")[0]
    else:
        predict = p

    l = item["reference"].strip().split(" ")[-1].split(".")[0]
    if l in item["generated_caption"]:
        k = k + 1
    # else:
    #     print(i, item["generated_caption"], l)
    # # else:
    # #     print(item)
print(k / len(test_examples))

# if args.debug:
#     print("\n", {"prompt": prompt, "outputs": output_text}, "\n")
# json_samples = json.dumps(test_examples, ensure_ascii=False, indent=2)
# with open("/home/httang/project/llava/TinyLLaVA_Factory-main/generate/gene_prediction/generated_diseases_Meta-Llama-3-8B-Instruct-dinov2-base-base-gene-finetune-vit-llm-v2-img-preprocess-4407-1.json", 'w', encoding='utf-8') as f:
#     f.write(json_samples)