from unsloth import FastVisionModel 

import os
import re
import random
import time
import glob
import torch
from tqdm import tqdm
from utils.util import set_seed, convert_to_messages, DiscourseVideoQADataset, custom_collate_fn

set_seed(42)

discourse_video_qa_dataset = DiscourseVideoQADataset(
    df_path="./data/dataset.jsonl",
    image_transform=None,
    answer=None
)

test_dataloader = torch.utils.data.DataLoader(
    dataset=discourse_video_qa_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=custom_collate_fn
)

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit",
    load_in_4bit=True,  # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
)

FastVisionModel.for_inference(model)

for batch in test_dataloader:
    messages = convert_to_messages(
        images=batch["batch_images"][0],
        question=batch["batch_questions"][0],
        answer=batch["batch_answers"][0] if "batch_answers" in batch else None
    )

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        batch["batch_images"][0],
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=64)
    decoded_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
    print(decoded_text)





# for batch in test_dataloader:
#     messages = convert_to_messages(
#         images=batch["images"],
#         question=batch["questions"],
#         answer=batch["answers"] if "answers" in batch else None
#     )
#     print("----")
#     print(messages)

#     print("\n----\n")

#     input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
#     inputs = tokenizer(
#         batch["images"],
#         input_text,
#         add_special_tokens=False,
#         return_tensors="pt",
#     ).to("cuda")
#     generated_ids = model.generate(**inputs, max_new_tokens=10)
#     decoded_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
#     print(decoded_text)