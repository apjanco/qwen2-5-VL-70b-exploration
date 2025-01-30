import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
from rich import print

if not os.path.exists('output'):
    os.makedirs('output')


dataset = load_dataset(
  "fmb-quibdo/sergio-notebooks"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2.5-VL-72B-Instruct',
    attn_implementation="flash_attention_2",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")

for item in tqdm(dataset['train']):
    item_path = Path('output') / f"{item['name'].split('.')[0]}.md"
    if not item_path.exists():
        # Preparation for inference
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": item['image'],
                    },
                    {"type": "text", "text": "Extract text."},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=1500)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        item_path.write_text(output_text[0])