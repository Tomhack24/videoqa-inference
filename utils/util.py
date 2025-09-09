import torch
import random
import numpy as np
import pandas as pd
import glob
from PIL import Image
from torchvision import transforms
import os
import cv2
# ...existing code...

VIDEO_DIR = "./data/video"
FPS = 0.5
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

def extract_frames_from_video(video_path, fps=FPS):
    """
    mp4からフレームを抽出してPIL.Imageのリストを返す
    """
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(video_fps / fps)

    frames = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            # OpenCV: BGR → RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)
            frames.append(pil_img)
        idx += 1
    cap.release()
    return frames



class DiscourseVideoQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, answer=True, image_transform=None,):
        
        self.df = pd.read_json(df_path, lines=True)
        self.answer = answer 
        self.image_transform = image_transform

        self.samples = []
        for _, row in self.df.iterrows():

            for qa in row['qa_pairs']:
                self.samples.append({
                    "video_id": row.get("video_id"),
                    "question": qa.get("question"),
                    "answer": qa.get("answer") if self.answer else None,
                    "evidence" : qa.get("evidence") if self.answer else None,
                    "question_id": qa.get("question_id")
                })


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = extract_frames_from_video(f"{VIDEO_DIR}/{sample['video_id']}.mp4", fps=FPS)

        if self.image_transform:
            frames = [self.image_transform(img) for img in frames]
            
        out = {
            "images": frames,  
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
        answer=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    for batch in test_dataloader:
        print(batch)