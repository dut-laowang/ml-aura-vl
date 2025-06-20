import transformers
from collections import defaultdict
import typing as t
import torch


class DictCollatorWithPadding(transformers.DataCollatorWithPadding):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        return_tensors: str = "pt",
        pad_to_multiple_of: int = 2,
        **kwargs,
    ):
        # ✅ 官方建议：padding 应设置为 "left"
        tokenizer.padding_side = "left"

        super().__init__(
            tokenizer=tokenizer,
            return_tensors=return_tensors,
            pad_to_multiple_of=pad_to_multiple_of,
            **kwargs,
        )

        # ✅ 添加 pixel_values，使其参与主张量处理逻辑
        self.tensor_set = {"input_ids", "attention_mask", "pixel_values"}

    def __call__(self, batch: t.List[dict]) -> dict:
        ret = defaultdict(list)
        tensors = []
        meta_set = set(batch[0].keys()) - self.tensor_set
        for sample in batch:
            tensors.append({k: sample[k] for k in self.tensor_set if k in sample})
            for k in meta_set:
                ret[k].append(sample[k])

        # 使用官方 DataCollator 处理文本部分
        tensors = super().__call__(tensors)
        ret.update(tensors)

        # ✅ pixel_values 必须手动 stack
        if "pixel_values" in ret and isinstance(ret["pixel_values"], list):
            ret["pixel_values"] = torch.stack(ret["pixel_values"])

        return ret
