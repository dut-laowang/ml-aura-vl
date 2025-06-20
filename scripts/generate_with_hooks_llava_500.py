import argparse
import logging
import os
from pathlib import Path
from PIL import Image
import random
import torch
from transformers import set_seed
import pandas as pd
import csv

from src.models.model_with_hooks import load_huggingface_model, ModelWithHooks
from src.parsers import parsers
from src.hooks import get_hook

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger().setLevel(logging.INFO)

def print_generated(text: str, tag: str) -> None:
    logging.info(f"\n========== {tag} ==========" + f"\n{text}\n")

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

    module_names_hooks = ModelWithHooks.find_module_names(base_model, args.module_names)
    hooks = []
    for module_name in module_names_hooks:
        state_path = None
        if args.intervention_state_path:
            state_path = (
                Path(args.interventions_cache_dir)
                / args.intervention_state_path
                / f"{module_name}.statedict"
            )
        hook = get_hook(
            args.intervention_name,
            module_name=module_name,
            device=args.device,
            state_path=state_path,
        )
        hooks.append(hook)

    model_hooks = ModelWithHooks(module=base_model)
    hooked_model = model_hooks.module

    # ✅ 读取 test.xlsx 图文对应关系
    try:
        df = pd.read_excel("llava1/test.xlsx")
    except:
        df = pd.read_excel("/workspace/ml-aura/llava1/test.xlsx")

    image_dir = "/workspace/ml-aura/pics_500"
    output_path = "/workspace/ml-aura/generated_outputs_with_hooks.csv"

    # ✅ 若文件不存在，先写入表头
    if not os.path.exists(output_path):
        with open(output_path, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["name", "prompt", "output_no_hook", "output_with_hook", "output_no_hook_again"])
            writer.writeheader()

    gen_kwargs = {
        "max_new_tokens": 512,
        "do_sample": True,
    }

    for idx, row in df.iterrows():
        image_name = row["name"]
        prompt = row["prompt"]
        image_path = os.path.join(image_dir, image_name)

        if not os.path.exists(image_path):
            logging.warning(f"[WARN] Skipping {image_name}, not found.")
            continue

        image = Image.open(image_path).convert("RGB")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
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

        def decode(model):
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=model_inputs["input_ids"],
                    attention_mask=model_inputs["attention_mask"],
                    pixel_values=model_inputs.get("pixel_values", None),
                    **gen_kwargs
                )
            return processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # ▶ 原始模型输出
        set_seed(args.seed)
        out_no_hook = decode(base_model)
        print_generated(out_no_hook, f"WITHOUT HOOK (Image {idx+1})")

        # ▶ 注入 Hook 输出
        model_hooks.register_hooks(hooks=hooks)
        set_seed(args.seed)
        out_with_hook = decode(hooked_model)
        print_generated(out_with_hook, f"WITH HOOK (Image {idx+1})")

        # ▶ 去除 Hook 后再输出
        model_hooks.remove_hooks()
        set_seed(args.seed)
        out_no_hook_again = decode(hooked_model)
        print_generated(out_no_hook_again, f"WITHOUT HOOK AGAIN (Image {idx+1})")

        # ✅ 逐行写入 CSV
        with open(output_path, "a", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["name", "prompt", "output_no_hook", "output_with_hook", "output_no_hook_again"])
            writer.writerow({
                "name": image_name,
                "prompt": prompt,
                "output_no_hook": out_no_hook,
                "output_with_hook": out_with_hook,
                "output_no_hook_again": out_no_hook_again
            })

        logging.info(f"[{idx+1}/{len(df)}] ✅ Written to CSV: {image_name}")

    logging.info(f"\n✅ 所有图像处理完成，最终输出保存在: {output_path}")

def main(args: argparse.Namespace) -> None:
    args = parsers.merge_config_into_args(args)
    generate(args)

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Generate with hooks",
        description="Generate responses from a LLaVA model with AURA hooks",
    )
    parser = parsers.add_config_args(parser)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--module-names", nargs="*", type=str, default=[".*"])
    parser.add_argument("--intervention-name", type=str, default="dummy")
    parser.add_argument("--intervention-state-path", type=str, default=None)
    parser.add_argument("--interventions-cache-dir", type=str, default=parsers.INTERVENTIONS_CACHE_DIR)
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
