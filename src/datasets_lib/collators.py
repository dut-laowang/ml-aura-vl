import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any


class DictCollatorWithPadding:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def pad_tensor_list(self, tensor_list, padding_value=0):
        """
        自动补 pad 的 stack 实现（支持 2D, 3D, 4D...）
        """
        max_shape = [max([t.size(dim) for t in tensor_list]) for dim in range(tensor_list[0].dim())]
        padded = []
        for t in tensor_list:
            pad = []
            for i in reversed(range(t.dim())):
                pad.extend([0, max_shape[i] - t.size(i)])
            padded.append(torch.nn.functional.pad(t, pad, value=padding_value))
        return torch.stack(padded)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_output = {}
        keys = batch[0].keys()

        for key in keys:
            values = [sample[key] for sample in batch]

            if key in ["input_ids", "attention_mask"]:
                batch_output[key] = pad_sequence(values, batch_first=True, padding_value=self.tokenizer.pad_token_id)

            elif key == "pixel_values":
                x = torch.stack(values)

                # ----------- 维度说明 ------------------------
                # 常见合法输入维度（取决于模型）：
                # 4D: [B, C, H, W]            → LLaVA 单图输入
                # 6D: [B, N, M, C, H, W]       → LLaMA-V 多图输入
                #     B: batch size
                #     N: concurrent multimodal rounds (常为 1)
                #     M: 每轮的图像数量（1 或更多）
                #     C: 图像通道数（3）
                #     H, W: 图像高宽
                #
                # 7D: [B, A, N, M, C, H, W] → 可能是数据增强或嵌套过深
                #     A: augmentation 数量（或乱嵌套维度）
                # ---------------------------------------------

                if x.ndim == 4:
                    # LLaVA 格式
                    batch_output[key] = x  # [B, C, H, W]
                elif x.ndim == 6:
                    # 标准 LLaMA-V 格式
                    batch_output[key] = x  # [B, N, M, C, H, W]
                elif x.ndim == 7:
                    # 错误嵌套，如 [B, A, N, M, C, H, W]
                    print(f"[WARNING] pixel_values 维度为 7，将尝试降维: {x.shape}")
                    B, A, N, M, C, H, W = x.shape[:7]
                    # 合并 A 和 N → 新 N；保留 M,C,H,W
                    x = x.view(B, A * N, M, C, H, W)  # → [B, N’, M, C, H, W]
                    batch_output[key] = x
                else:
                    raise RuntimeError(f"pixel_values 必须是 4/6/7 维，当前为 {x.ndim}: {x.shape}")

            elif key in ["aspect_ratio_ids", "aspect_ratio_mask"]:
                try:
                    batch_output[key] = pad_sequence(values, batch_first=True, padding_value=0)
                except Exception as e:
                    print(f"[WARNING] collate {key} failed: {e}")
                    batch_output[key] = torch.zeros(len(batch_output["input_ids"]), 1, dtype=torch.long)

            elif key == "cross_attention_mask":
                try:
                    if values[0].ndim == 4:
                        # 维度: [B, L, N, M] or [B, L, N, 1]
                        # 注意：L 是 token 长度，不等长，所以不能 stack，需手动 pad
                        batch_output[key] = self.pad_tensor_list(values, padding_value=0)
                    else:
                        batch_output[key] = pad_sequence(values, batch_first=True, padding_value=0)
                except Exception as e:
                    print(f"[WARNING] collate cross_attention_mask failed: {e}")
                    batch_output[key] = torch.zeros(len(batch_output["input_ids"]), 1, 1, 1, dtype=torch.long)

            elif key in ["label", "responses"]:
                batch_output[key] = torch.stack(values)

            elif key == "id":
                batch_output[key] = values

            else:
                try:
                    batch_output[key] = torch.stack(values)
                except Exception as e:
                    print(f"[WARNING] collate 未能 stack key={key}: {e}")

        return batch_output
