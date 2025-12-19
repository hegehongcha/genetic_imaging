'''
@Description:
@Author: jiajunlong
@Date: 2024-06-19 19:30:17
@LastEditTime: 2024-06-19 19:32:47
@LastEditors: jiajunlong
'''
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


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


def main(args):
    # Model
    disable_torch_init()
    if args.model_path is not None:
        model, tokenizer, image_processor, context_len = load_pretrained_model(model_name_or_path=args.model_path,
                                                                               load_8bit=args.load_8bit,
                                                                               load_4bit=args.load_4bit,
                                                                               device=args.device)
    else:
        assert args.model is not None, 'model_path or model must be provided'
        model = args.model
        if hasattr(model.config, "max_sequence_length"):
            context_len = model.config.max_sequence_length
        else:
            context_len = 2048
        tokenizer = model.tokenizer
        image_processor = model.vision_tower._image_processor
    print(111)
    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    data_args = model.config
    model.to(args.device)
    print(model)
    with open("/home/httang/project/llava/TinyLLaVA_Factory-main/genetic_dataset/gene_test.json", "r") as f:
        samples = json.load(f)
    test_examples = []
    print(222)
    for l, sample in enumerate(samples):
        print(l)
        img = "/home/httang/dataset/Image_genetic/Processed_T1/" + sample["image"]
        image = nib.load(img).get_fdata()
        image = torch.tensor(image).unsqueeze(0).float()
        image = center_pad(image, target_shape=(130, 140, 140))
        image_tensor = image.unsqueeze(0).to(args.device)
        data_dict = text_processor(sample["conversations"], mode="eval")
        input_ids = data_dict["input_ids"].unsqueeze(0).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                tokenizer=tokenizer,
                images=image_tensor,
                do_sample=True,
                num_beams=1,
                temperature=args.temperature,
                top_p=0.9,
                top_k=10,
                max_new_tokens=args.max_new_tokens,
                stop_strings="</s>",
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        output_token = output_ids[0]
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        output_text = tokenizer.decode(output_token, add_special_tokens=False)
        print(output_text)
        example = {}
        example["image"] = sample["image"]
        example["generated_caption"] = output_text
        example["reference"] = sample["conversations"][1]["value"]
        test_examples.append(example)

    # if args.debug:
    #     print("\n", {"prompt": prompt, "outputs": output_text}, "\n")
    json_samples = json.dumps(test_examples, ensure_ascii=False, indent=2)
    with open("/home/httang/project/llava/TinyLLaVA_Factory-main/generate/preference/generated_diseases_Meta-Llama-3-8B-Instruct-dinov2-base-base-gene-finetune-vit-llm.json", 'w', encoding='utf-8') as f:
        f.write(json_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default='llama')
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    # args = parser.parse_args()
    args = parser.parse_args(args=["--model-path", "/home/httang/project/llava/TinyLLaVA_Factory-main/checkpoints/llava_factory/tiny-llava-Meta-Llama-3-8B-Instruct-dinov2-base-base-gene-finetune-vit-llm"])
    main(args)
