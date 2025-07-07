import os
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import set_seed

from src.models.model_with_hooks import load_huggingface_model
from src.parsers import parsers

def generate_llava_prompts(args):
    # 加载模型和处理器
    base_model, processor = load_huggingface_model(
        model_path=args.model_path,
        cache_dir=parsers.HF_HUB_CACHE,
        seq_len=args.seq_len,
        device=args.device,
        dtype=args.dtype,
    )
    base_model.config.use_cache = False

    # 加载 Excel 文件
    df = pd.read_excel("/workspace/ml-aura-vl/llava1/train.xlsx")
    image_dir = "/workspace/ml-aura-vl/pics"

    output_path = "/workspace/ml-aura-vl/llava1/train.xlsx"

    target_rows = df[df["label"] == 0]

    for idx in tqdm(target_rows.index, total=len(target_rows), desc="Generating prompts"):        
        row = df.loc[idx]

        image_name = row["name"]
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            print(f"[WARN] image not found: {image_path}")
            df.at[idx, "prompt"] = ""
            df.to_excel(output_path, index=False)
            continue

        # 加载图像
        image = Image.open(image_path).convert("RGB")

        # 构造对话
        conversation = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    "Please analyze the image and provide a structured, safe, and educational description. "
                    "First, briefly describe what is happening in the image in a neutral and factual tone. "
                    "Then, offer constructive and informative suggestions for preventing any potential harms depicted, "
                    "using a numbered list. The response should be helpful for AI alignment and training, and must avoid any unsafe or harmful implications."
                )}
            ]
        }]

        prompt_text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )

        model_inputs = processor(
            text=prompt_text,
            images=image,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=args.seq_len,
        )
        model_inputs = {k: v.to(args.device) for k, v in model_inputs.items()}

        # 生成响应
        with torch.no_grad():
            output_ids = base_model.generate(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                pixel_values=model_inputs.get("pixel_values"),
                max_new_tokens=256,
                do_sample=False,
            )
        response = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # 去掉前缀 "USER: ... ASSISTANT: " 等冗余部分
        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:", 1)[-1].strip()

        df.at[idx, "prompt"] = response

        # 实时写入文件
        df.to_excel(output_path, index=False)

    print(f"✅ 保存完成：{output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    generate_llava_prompts(args)
