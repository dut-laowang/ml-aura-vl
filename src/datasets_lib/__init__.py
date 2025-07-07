# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import typing as t
from pathlib import Path

import torch
import transformers

from .jigsaw_dataset import get_jigsaw_dataset
from .llava_dataset import get_llava_dataset  # ✅ 覆盖为基于 oss 的新版本


# 注册所有数据集加载函数
DATASET_LOADERS_REGISTRY = {
    "jigsaw": get_jigsaw_dataset,
    "llava1": get_llava_dataset,  # ✅ 注册 oss 图文数据集加载器
}


def get_dataset(
    name: str,
    datasets_folder: Path,
    split: str,
    tokenizer: t.Optional[transformers.PreTrainedTokenizer] = None,
    model_path_str: str = "",  # ✅ 添加：兼容 llama-3.2-vision
) -> t.Tuple[torch.utils.data.Dataset, t.Callable]:
    """
    Loads and returns a dataset split given its name.
    Returns both a PyTorch Dataset and a collate function.

    Args:
        name (str): dataset name, e.g., "jigsaw" or "llava"
        datasets_folder (Path): path to datasets root folder
        split (str): "train", "val", or "test"
        tokenizer (transformers.PreTrainedTokenizer or ProcessorMixin): tokenizer or processor
        model_path_str (str): 模型路径名，用于判断是否使用 llama-3.2-vision 的分支逻辑

    Returns:
        Tuple[Dataset, collate_fn]
    """
    assert name in DATASET_LOADERS_REGISTRY, f"Unknown dataset: {name}"
    data_loader_fn = DATASET_LOADERS_REGISTRY[name]

    if name == "llava" or name == "llava1":
        return data_loader_fn(
            datasets_folder / name,
            split=split,
            tokenizer=tokenizer,
            model_path_str=model_path_str  # ✅ 向数据加载器传递模型路径
        )
    else:
        return data_loader_fn(datasets_folder / name, split=split, tokenizer=tokenizer)


def get_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    collate_fn: t.Callable,
    drop_last: bool,
    shuffle: bool,
    **kwargs: dict,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
        shuffle=shuffle,
        **kwargs,
    )


if __name__ == "__main__":
    pass
