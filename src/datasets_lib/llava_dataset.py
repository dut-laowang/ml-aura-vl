import os
from pathlib import Path
from PIL import Image
import torch
import pandas as pd
from transformers import ProcessorMixin  # 可为 AutoProcessor 或 LlavaProcessor

from .collators import DictCollatorWithPadding


class LLaVAOSSDataset(torch.utils.data.Dataset):
    LABEL_NAMES = ["non-toxic", "toxic"]

    def __init__(self, path: Path, split: str, tokenizer: ProcessorMixin, model_path_str: str = ""):
        self.data = pd.read_excel(path / f"{split}.xlsx")
        self.tokenizer = tokenizer
        self.index = torch.arange(len(self.data))
        self.image_root = "/workspace/ml-aura-vl/MM-SafetyBench/data/pics"

        self.is_llama_vision = "llama-3.2" in model_path_str.lower() and "vision" in model_path_str.lower()

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

        filename = os.path.basename(row["name"])
        local_path = os.path.join(self.image_root, filename)
        try:
            image = Image.open(local_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image at {local_path}: {e}")

        prompt_text = str(row.get("prompt", "") or "").strip()
        output_text = str(row.get("output", "") or "").strip()
        #merged_text = prompt_text + " " + output_text
        #merged_text = prompt_text
        merged_text =  output_text
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
            raise RuntimeError("Tokenizer missing apply_chat_template()")

        prompt = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=False
        )

        image_input = [image] if self.is_llama_vision else image
        text_input = [prompt] if self.is_llama_vision else prompt

        inputs = self.tokenizer(
            text=text_input,
            images=image_input,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=1024
        )

        # 统一升维
        def try_unsqueeze(key: str):
            if key in inputs and isinstance(inputs[key], torch.Tensor):
                if inputs[key].ndim == 1:
                    inputs[key] = inputs[key].unsqueeze(0)

        for key in ["input_ids", "attention_mask", "aspect_ratio_ids", "aspect_ratio_mask", "cross_attention_mask"]:
            try_unsqueeze(key)

        if "pixel_values" in inputs:
            pv = inputs["pixel_values"]
            if isinstance(pv, list):
                pv = torch.stack(pv)

            if self.is_llama_vision:
                # LLaMA-Vision 模型要求 6D: [B, N, M, C, H, W]
                if pv.ndim == 4:
                    pv = pv.unsqueeze(0).unsqueeze(0)  # -> [1, 1, 1, C, H, W]
                elif pv.ndim == 5:
                    pv = pv.unsqueeze(1)  # -> [1, 1, M, C, H, W]
                elif pv.ndim != 6:
                    raise RuntimeError(f"Unexpected shape for LLaMA-V pixel_values: {pv.shape}")
            else:
                if pv.ndim == 3:
                    pv = pv.unsqueeze(0)
                elif pv.ndim == 4:
                    pass
                else:
                    raise RuntimeError(f"LLaVA unexpected pixel_values shape: {pv.shape}")

            inputs["pixel_values"] = pv

        print(f"[DEBUG] 图像 shape: {image.size}")
        print(f"[DEBUG] 文本: {merged_text}")
        print(f"[DEBUG] input_ids shape: {inputs['input_ids'].shape}")
        print(f"[DEBUG] pixel_values shape: {inputs['pixel_values'].shape}")
        print(f"[DEBUG] inputs keys: {list(inputs.keys())}")

        sample_id = os.path.splitext(os.path.basename(row["name"]))[0]

        return {
            "id": sample_id,
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"] if self.is_llama_vision else inputs["pixel_values"].squeeze(0),
            "aspect_ratio_ids": inputs.get("aspect_ratio_ids", torch.tensor([])).squeeze(0),
            "aspect_ratio_mask": inputs.get("aspect_ratio_mask", torch.tensor([])).squeeze(0),
            "cross_attention_mask": inputs.get("cross_attention_mask", torch.tensor([]))
            if self.is_llama_vision else inputs.get("cross_attention_mask", torch.tensor([])).squeeze(0),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "responses": torch.zeros(1),
        }

    def __len__(self):
        return len(self.index)


def get_llava_dataset(path: Path, split: str, tokenizer: ProcessorMixin, model_path_str: str = ""):
    dataset = LLaVAOSSDataset(path, split, tokenizer, model_path_str)
    print("[DEBUG-new] tokenizer class:", type(tokenizer))
    print("[DEBUG-new] tokenizer.tokenizer class:", type(tokenizer.tokenizer))
    return dataset, DictCollatorWithPadding(tokenizer=tokenizer.tokenizer)
