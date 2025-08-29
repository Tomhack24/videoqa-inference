import torch
import random
import numpy as np
import pandas as pd
import glob
from PIL import Image

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

class DiscourseVideoQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, image_transform=None, answer=True):
        # self.transform = image_transform
        self.df = pd.read_json(df_path, lines=True)
        self.image_dir = image_dir
        self.answer = answer #boolean


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 修正案
        vid = str(int(self.df["video_id"][idx])).zfill(4)
        image_paths = sorted(glob.glob(f"{self.image_dir}/{vid}/frame_*.jpg"))
        images = [Image.open(p) for p in image_paths]
        question = self.df['question'][idx]
        answer = self.df['answer'][idx] if self.answer else None

        if self.answer:
            return {
                "images": images,
                "question": question,
                "answer": answer
            }
        else:
            return {
                "images": images,
                "question": question
            }
        

def custom_collate_fn(batch):
    # batchは __getitem__ が返す辞書のリスト
    # 例: [{'images': tensor1, 'question': 'q1'}, {'images': tensor2, 'question': 'q2'}]

    # 各要素をまとめる
    images_list = [item['images'] for item in batch]
    questions = [item['question'] for item in batch]
    
    # 画像はフレーム数が異なる可能性があるため、ここではTensorのリストのままにしておく
    # (もしフレーム数を揃える場合は、ここでパディング処理などを行う)
    
    batched_data = {
        "images": images_list,
        "questions": questions
    }

    # answerが存在する場合のみバッチに追加
    if 'answer' in batch[0]:
        answers = [item['answer'] for item in batch]
        batched_data["answers"] = answers
        
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
        df_path="./data/data.jsonl",
        image_dir="./data/images",
        image_transform=None,
        answer=True
    )
    # print(dataset[0])

    test_dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    for batch in test_dataloader:
        messages = convert_to_messages(
            images=batch["images"],
            question=batch["questions"],
            answer=batch["answers"] if "answers" in batch else None
        )

        print(messages)