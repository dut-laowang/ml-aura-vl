import os
from pathlib import Path
from PIL import Image
import torch
import pandas as pd
from transformers import ProcessorMixin  # 可为 AutoProcessor 或 LlavaProcessor

from .collators import DictCollatorWithPadding


class LLaVAOSSDataset(torch.utils.data.Dataset):
    LABEL_NAMES = ["non-toxic", "toxic"]

    def __init__(self, path: Path, split: str, tokenizer: ProcessorMixin):
        self.data = pd.read_excel(path / f"{split}.xlsx")
        self.tokenizer = tokenizer
        self.index = torch.arange(len(self.data))
        self.image_root = "/workspace/ml-aura/pics"

        all_labels = torch.tensor(self.data["label"].values)
        bincount = torch.bincount(all_labels)
        print(f"[DEBUG-dataset] 标签总分布 (index=label): {bincount}")

    def set_label(self, label: str) -> None:
        if label is None:
            return
        mapping = {"non-toxic": 0, "toxic": 1}
        assert label in mapping, f"Label {label} not in {list(mapping.keys())}"
        labels = torch.tensor(self.data["label"].values)

        self.index = torch.where(labels == mapping[label])[0]

        print(f"[DEBUG-dataset] 设置 label = '{label}' 后剩余样本数量: {len(self.index)}")

    def __getitem__(self, idx: int):
        row = self.data.iloc[int(self.index[idx])]

        # 加载图像
        try:
            filename = os.path.basename(row["name"])
            local_path = os.path.join(self.image_root, filename)
            image = Image.open(local_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load local image at {local_path}: {e}")

        # 获取 prompt 和 output
        prompt_text = row.get("prompt", "")
        output_text = row.get("output", "")

        if not isinstance(prompt_text, str):
            prompt_text = ""
        if not isinstance(output_text, str):
            output_text = ""

        merged_text = prompt_text.strip() + " " + output_text.strip()

        # 构造标准对话输入
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": merged_text},
                ]
            }
        ]

        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise RuntimeError("Tokenizer does not support apply_chat_template")

        prompt = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=False
        )

        inputs = self.tokenizer(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=1024  # 避免过长 input_ids 报错
        )

        print(f"[DEBUG-input] 图像 shape: {image.size}")
        print(f"[DEBUG-input] 拼接文本: {merged_text}")
        print(f"[DEBUG-input] input_ids shape: {inputs['input_ids'].shape}")
        print(f"[DEBUG-input] pixel_values shape: {inputs['pixel_values'].shape}")

        sample_id = os.path.splitext(os.path.basename(row["name"]))[0]

        return {
            "id": sample_id,
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "responses": torch.zeros(1),
        }

    def __len__(self):
        return len(self.index)


def get_llava_dataset(path: Path, split: str, tokenizer: ProcessorMixin):
    dataset = LLaVAOSSDataset(path, split, tokenizer)
    print("[DEBUG-new] tokenizer class:", type(tokenizer))
    print("[DEBUG-new] tokenizer.tokenizer class:", type(tokenizer.tokenizer))
    return dataset, DictCollatorWithPadding(tokenizer=tokenizer.tokenizer)
