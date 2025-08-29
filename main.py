import os
import json
import glob

from unsloth import FastVisionModel # FastLanguageModel for LLMs
from PIL import Image
import torch

image_001_path = "./data/image/001"
pattern = os.path.join(image_001_path, "frame_*.jpg")
image_paths = glob.glob(pattern)
image_paths.sort()

list_of_images = [Image.open(p) for p in image_paths]

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

FastVisionModel.for_inference(model)

messages = [
    {"role": "user", "content": [
        *map(lambda img: {"type": "image"}, list_of_images[::30]),
        {"type": "text", "text": "explain this images"}
    ]}
]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
inputs = tokenizer(
    list_of_images[::30],
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=10)
decoded_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

print(decoded_text)