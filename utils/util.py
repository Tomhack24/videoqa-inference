import torch
import random
import numpy as np
import pandas as pd
import glob
from PIL import Image
import os
# ...existing code...

PROMPT_PATH = "./prompts/prompt_inference.txt"
with open(PROMPT_PATH, "r") as f:
    PROMPT = f.read()

def set_seed(seed):
    """
    シードを固定する．

    Parameters
    ----------
    seed : int
        乱数生成に用いるシード値．
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_text(text):
    pass

# class DiscourseVideoQADataset(torch.utils.data.Dataset):
#     def __init__(self, df_path, image_transform=None, answer=True):
#         # self.transform = image_transform
#         self.df = pd.read_json(df_path, lines=True)
#         self.answer = answer #boolean


#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):

#         image_paths = [f"{self.df['images_dir'][idx]}/{frame}" for frame in self.df['frames'][idx]]
#         question = self.df['question'][idx]
#         answer = self.df['answer'][idx] if self.answer else None

#         if self.answer:
#             return {
#                 "images_path": image_paths  ,
#                 "question": question,
#                 "answer": answer
#             }
#         else:
#             return {
#                 "images_path": image_paths,
#                 "question": question
#             }

# ...existing code...


# ...existing code...

class DiscourseVideoQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_transform=None, answer=True):
        # self.transform = image_transform
        self.df = pd.read_json(df_path, lines=True)
        self.answer = answer #boolean
        self.image_transform = image_transform

        self.samples = []
        for _, row in self.df.iterrows():
            images_dir = row['images_dir']
            image_paths = [os.path.join(images_dir, f) for f in row['frames']]
            for qa in row['qa_pairs']:
                self.samples.append({
                    "images_path": image_paths,
                    "question": qa.get("question"),
                    "answer": qa.get("answer") if self.answer else None,
                    "evidence" : qa.get("evidence") if self.answer else None,
                    "question_id": qa.get("question_id")
                })


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        out = {
            "images": sample["images_path"],  
            "question": sample["question"]
        }

        if self.answer and sample.get("answer") is not None:
            out["answer"] = sample["answer"]
            out["evidence"] = sample["evidence"]

        if sample.get("question_id") is not None:
            out["question_id"] = sample["question_id"]

        return out
        

def custom_collate_fn(batch):

    # 各要素をまとめる
    list_of_images = [item['images'] for item in batch]
    list_of_question = [item['question'] for item in batch]


    batched_data = {
        "batch_images": list_of_images,
        "batch_questions": list_of_question,
    }

    if 'answer' in batch[0]:
        list_of_answer = [item['answer'] for item in batch if 'answer' in item]
        list_of_evidence = [item['evidence'] for item in batch if 'evidence' in item]
        batched_data["batch_answers"] = list_of_answer
        batched_data["batch_evidences"] = list_of_evidence

    if 'question_id' in batch[0]:
        list_of_question_id = [item['question_id'] for item in batch if 'question_id' in item]
        batched_data["batch_question_ids"] = list_of_question_id

    return batched_data

def convert_to_messages(images, question, answer=None):
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": f"{PROMPT}"},
            *map(lambda img: {"type": "image"}, images),
            {"type": "text", "text": question}
        ]}
    ]

    if answer:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})

    return messages


if __name__ == "__main__":
    dataset = DiscourseVideoQADataset(
        df_path="./data/dataset.jsonl",
        image_transform=None,
        answer=True
    )
    print(len(dataset))

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    batch = next(iter(dataloader))
    print(batch)