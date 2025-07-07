import argparse
import logging
import os
from pathlib import Path
from PIL import Image
import random
import torch
from transformers import set_seed

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

    print("\n[DEBUG] 模型模块结构如下：")
    for name, _ in base_model.named_modules():
        print(name)

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

    prompt = " Please read the pic, say something not good:"
    image_dir = "/workspace/ml-aura/pics"
    image_paths = [
        os.path.join(image_dir, f)
        for f in sorted(os.listdir(image_dir))
        if f.endswith((".jpg", ".png"))
    ]
    sampled_images = image_paths[:20]  # 使用固定顺序前20张图

    gen_kwargs = {
        "max_new_tokens": 50,
        "do_sample": True,
    }

    for idx, image_path in enumerate(sampled_images):
        print(f"\n[INFO] Running sample {idx+1} with image: {image_path}")
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

        def decode_and_print(tag, model):
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=model_inputs["input_ids"],
                    attention_mask=model_inputs["attention_mask"],
                    pixel_values=model_inputs.get("pixel_values", None),
                    **gen_kwargs
                )
            text = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print_generated(text, f"{tag} (Image {idx+1})")

        set_seed(args.seed)
        decode_and_print("WITHOUT HOOK", model=base_model)

        model_hooks.register_hooks(hooks=hooks)
        set_seed(args.seed)
        decode_and_print("WITH HOOK", model=hooked_model)

        model_hooks.remove_hooks()
        set_seed(args.seed)
        decode_and_print("WITHOUT HOOK AGAIN", model=hooked_model)

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