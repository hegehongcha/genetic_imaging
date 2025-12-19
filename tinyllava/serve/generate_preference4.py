'''
@Description:
@Author: jiajunlong
@Date: 2024-06-19 19:30:17
@LastEditTime: 2024-06-19 19:32:47
@LastEditors: jiajunlong
'''
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import argparse
import requests
from PIL import Image
from io import BytesIO

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

    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    data_args = model.config
    image_processor = ImagePreprocess(image_processor, data_args)
    model.to(args.device)
    if getattr(text_processor.template, 'role', None) is None:
        roles = ['USER', 'ASSISTANT']
    else:
        roles = text_processor.template.role.apply()
    with open("/home/httang/project/llava/TinyLLaVA_Factory-main/tinyllava/serve/filter_cap_raw.json", "r") as f:
        samples = json.load(f)
    images_list = samples["annotations"]
    test_examples = []
    description_list = [
        "Describe the given chest x-ray image in detail.",
        "Take a look at this chest x-ray and describe the findings and impression.",
        "Could you provide a detailed description of the given x-ray image?",
        "Describe the given chest x-ray image as detailed as possible.",
        "What are the key findings in this chest x-ray image?"
    ]
    stop_words_ids = [torch.tensor([82, 29]).to(model.device)]  # '###' can be encoded in two different ways.
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    for l, sample in enumerate(images_list[75000:100000]):
        print(l+75000)
        img = "/home/httang/dataset/mimic/image/" + sample["image_id"] + ".jpg"
        msg = Message()
        image = load_image(img)
        # Similar operation in model_worker.py
        image_tensor = image_processor(image)
        image_tensor = image_tensor.unsqueeze(0).to(model.device, dtype=torch.float16)
        p = random.choice(description_list)
        inp = f"{roles[0]}: " + p
        # while True:
        #     try:
        #         inp = input(f"{roles[0]}: ")
        #     except EOFError:
        #         inp = ""
        #     if not inp:
        #         print("exit...")
        #         break
        #
        #     print(f"{roles[1]}: ", end="")

        if image is not None:
            # first message
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            msg.add_message(inp)
        result = text_processor(msg.messages, mode='eval')
        prompt = result['prompt']
        input_ids = result['input_ids'].unsqueeze(0).to(model.device)

        # stop_str = text_processor.template.separator.apply()[1]
        # keywords = [stop_str]
        # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                num_beams=1,
                temperature=args.temperature,
                top_p=0.9,
                top_k=10,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
            )
        output_token = output_ids[0]
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        msg.messages[-1]['value'] = outputs
        output_text = tokenizer.decode(output_token, add_special_tokens=False)
        example = {}
        example["image_id"] = img
        example["generated_caption"] = output_text
        example["reference"] = sample["caption"]
        test_examples.append(example)

    # if args.debug:
    #     print("\n", {"prompt": prompt, "outputs": output_text}, "\n")
    json_samples = json.dumps(test_examples, ensure_ascii=False, indent=2)
    with open("/home/httang/project/llava/TinyLLaVA_Factory-main/generate/preference/generated_report_Medical-Llama3-8B-XrayCLIP__vit-b-16__laion2b-s34b-b88k-base-mimic-finetune-vit-frozen-llm-4.json", 'w', encoding='utf-8') as f:
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
    args = parser.parse_args(args=["--model-path", "/home/httang/project/llava/TinyLLaVA_Factory-main/checkpoints/llava_factory/tiny-llava-Medical-Llama3-8B-XrayCLIP__vit-b-16__laion2b-s34b-b88k-base-mimic-finetune-vit-frozen-llm"])
    main(args)
