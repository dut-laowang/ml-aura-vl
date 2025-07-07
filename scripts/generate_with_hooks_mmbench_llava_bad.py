import argparse
import logging
import os
import typing as t
import json
from collections import OrderedDict
from pathlib import Path
from PIL import Image
import random
import torch
import pandas as pd
from transformers import set_seed

from src.models.model_with_hooks import load_huggingface_model
from src.parsers import parsers

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger().setLevel(logging.INFO)

SCENARIOS = [
    "01-Illegal_Activitiy",
    "02-HateSpeech",
    "03-Malware_Generation",
    "04-Physical_Harm",
    "05-EconomicHarm",
    "06-Fraud",
    "07-Sex",
    "08-Political_Lobbying",
    "09-Privacy_Violence",
    "10-Legal_Opinion",
    "11-Financial_Advice",
    "12-Health_Consultation",
    "13-Gov_Decision",
]

def try_tensor(batch: dict, name: str, device: str):
    val = batch.get(name, None)
    if isinstance(val, list):
        try:
            val = torch.stack(val, dim=0)
        except Exception as e:
            print(f"[ERROR] Cannot convert {name} to tensor: {e}")
            return None
    return val.to(device) if isinstance(val, torch.Tensor) else val

def run_scenario(scenario: str, args: argparse.Namespace, base_model, processor):
    import openpyxl  # 确保依赖存在
    question_file = f"/workspace/ml-aura-vl/MM-SafetyBench/data/processed_questions/{scenario}.json"
    image_dir = f"/workspace/ml-aura-vl/MM-SafetyBench/data/imgs/{scenario}/SD_TYPO"
    output_path = f"/workspace/ml-aura-vl/MM-SafetyBench/questions_with_answers/{scenario}.xlsx"

    with open(question_file, "r", encoding="utf-8") as f:
        question_data = json.load(f)

    is_llama_vision = "llama-3.2" in args.model_path.lower() and "vision" in args.model_path.lower()
    gen_kwargs = {
        "max_new_tokens": 512,
        "do_sample": True,
    }

    # 初始化 Excel 文件并写入表头（只执行一次）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not os.path.exists(output_path):
        df_header = pd.DataFrame(columns=["name", "prompt", "output", "label"])
        df_header.to_excel(output_path, index=False)

    for qid, qobj in question_data.items():
        image_name = f"{qid}.jpg"
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            logging.warning(f"[WARN] Missing image: {image_path}, skipping.")
            continue

        image = Image.open(image_path).convert("RGB")
        prompt = qobj.get("Rephrased Question(SD)", None)
        if prompt is None:
            logging.warning(f"[WARN] No rephrased question found for {qid}, skipping.")
            continue

        conversation = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }]

        prompt_text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )

        model_inputs = processor(
            text=[prompt_text] if is_llama_vision else prompt_text,
            images=[image] if is_llama_vision else image,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=args.seq_len,
        )
        model_inputs = {k: v.to(args.device) for k, v in model_inputs.items()}

        if is_llama_vision:
            for name in ["aspect_ratio_ids", "aspect_ratio_mask", "cross_attention_mask"]:
                if name not in model_inputs:
                    model_inputs[name] = torch.zeros_like(model_inputs["input_ids"])

        aspect_ratio_ids = try_tensor(model_inputs, "aspect_ratio_ids", args.device) if is_llama_vision else None
        aspect_ratio_mask = try_tensor(model_inputs, "aspect_ratio_mask", args.device) if is_llama_vision else None
        cross_attention_mask = try_tensor(model_inputs, "cross_attention_mask", args.device) if is_llama_vision else None

        with torch.no_grad():
            output_ids = base_model.generate(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                pixel_values=model_inputs.get("pixel_values", None),
                aspect_ratio_ids=aspect_ratio_ids,
                aspect_ratio_mask=aspect_ratio_mask,
                cross_attention_mask=cross_attention_mask,
                **gen_kwargs
            )
        out_no_hook = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print("original", out_no_hook)

        # 每条推理完就写入一行（实时写入）
        df_row = pd.DataFrame([{
            "name": image_name,
            "prompt": prompt,
            "output": out_no_hook,
            "label": 1
        }])
        with pd.ExcelWriter(output_path, mode="a", engine="openpyxl", if_sheet_exists="overlay") as writer:
            df_row.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)

        logging.info(f"[{scenario}][{qid}] ✅ done")

    logging.info(f"\n✅ [{scenario}] 实时写入完成，结果保存在: {output_path}")


def generate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    random.seed(args.seed)

    base_model, processor = load_huggingface_model(
        model_path=args.model_path,
        cache_dir=parsers.HF_HUB_CACHE,
        seq_len=args.seq_len,
        device=args.device,
        dtype=args.dtype,
    )
    base_model.config.use_cache = False

    for idx, scenario in enumerate(SCENARIOS):
        logging.info(f"========== 开始处理 [{idx+1}/{len(SCENARIOS)}] {scenario} ==========")
        run_scenario(scenario, args, base_model, processor)

def main(args: argparse.Namespace) -> None:
    args = parsers.merge_config_into_args(args)
    generate(args)

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Batch Generate with base model only for MM-SafetyBench",
        description="Run base LLaVA model on all MM-SafetyBench scenarios and export xlsx."
    )
    parser = parsers.add_config_args(parser)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--module-names", nargs="*", type=str, default=[".*"])
    parser.add_argument("--intervention-name", type=str, default="dummy")  # unused
    parser.add_argument("--intervention-state-path", type=str, default=None)  # unused
    parser.add_argument("--interventions-cache-dir", type=str, default=parsers.INTERVENTIONS_CACHE_DIR)  # unused
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=0)
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    if args.verbose == 1:
        logging.basicConfig(level=logging.INFO)
    elif args.verbose >= 2:
        logging.basicConfig(level=logging.DEBUG)
    main(args)
