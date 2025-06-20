import argparse
import logging
import os
import typing as t
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src import hooks
from src.datasets_lib import get_dataloader, get_dataset
from src.models.model_with_hooks import ModelWithHooks, load_huggingface_model
from src.parsers import parsers
from src.utils import utils

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_model_name_from_path(model_path: t.Union[Path, str]) -> str:
    return str(Path(model_path).name)

def compute_responses(args: argparse.Namespace) -> None:
    [dataset, ] = parsers.get_single_args(args, ["dataset"])
    model_name = get_model_name_from_path(args.model_path)
    print("[DEBUG] 正在运行正确的11111 compute_responses")

    output_path = Path(args.responses_cache_dir)
    base_path = output_path / args.tag / model_name / dataset

    if args.verbose == 1:
        logging.getLogger().setLevel(logging.INFO)

    for arg, value in sorted(vars(args).items()):
        logging.info("Argument %s: %r", arg, value)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device in ["cuda", None] and torch.cuda.is_available():
        args.device = "cuda"
    elif args.device == "cuda":
        raise RuntimeError("Cuda not available")
    elif args.device is None:
        args.device = "cpu"

    model, processor = load_huggingface_model(
        model_path=args.model_path,
        cache_dir=parsers.HF_HUB_CACHE,
        device=args.device,
        dtype=args.dtype,
        rand_weights=(args.rand_weights == 1),
        seq_len=args.seq_len,
    )

    model = ModelWithHooks(module=model)

    print("\n[DEBUG重要] 模型模块结构如下：\n")
    for name, module in model.module.named_modules():
        print(name)
    print("\n[DEBUG] ===== 模型与Processor类型验证 =====")
    print(f"[DEBUG] model class: {type(model.module)}")
    print(f"[DEBUG] processor/tokenizer class: {type(processor)}")
    print("[DEBUG] =================================\n")

    train_dataset, collate_fn = get_dataset(
        name=dataset,
        datasets_folder=Path(args.data_dir),
        split="train",
        tokenizer=processor,
    )

    print("[DEBUG] 所有模型模块:")
    for name, _ in model.module.named_modules():
        print(name)

    module_names = model.find_module_names(model.module, args.module_names)
    print("[DEBUG] 匹配到的模块:", module_names)

    if not module_names:
        raise RuntimeError("❌ 未匹配到任何模块，请检查 --module-names 参数是否正确")

    assert isinstance(args.subset, list)
    subsets = train_dataset.LABEL_NAMES if len(args.subset) == 0 or args.subset == ["*"] else args.subset

    for subset in subsets:
        logging.info(f"Current subset: {subset}")
        train_dataset.set_label(subset)
        label_output_path = base_path / subset

        for module_name in module_names:
            os.makedirs(label_output_path / module_name, exist_ok=True)
            utils.dump_yaml(vars(args), label_output_path / "config.yaml")

        hook_fns = [
            hooks.get_hook(
                "postprocess_and_save",
                module_name=module_name,
                pooling_op_names=["mean"],
                output_path=label_output_path,
                save_fields=["id", "label", "responses"],
                threaded=False,
            )
            for module_name in module_names
        ]

        model.remove_hooks()
        model.hooks = []
        for hook in hook_fns:
            for name, module in model.module.named_modules():
                if name == hook.module_name:
                    module.register_forward_hook(hook)
                    model.hooks.append(hook)

        if not model.hooks:
            raise RuntimeError("❌ Hook 注册失败，hooks 为空")

        checkpoint = {"current_batch": 0}
        checkpoint_path = label_output_path / "checkpoint.pt"
        logging.info(f"Checkpointing to {str(checkpoint_path)}")

        if args.resume == 1 and checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            logging.info("Loaded existing checkpoint.")

        current_batch = checkpoint["current_batch"]

        loader = get_dataloader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            drop_last=False,
            shuffle=True,
        )

        max_batches = len(loader) if args.max_batches is None else min(len(loader), args.max_batches)

        if current_batch == max_batches:
            logging.warning(f"All batches found in [{output_path / args.tag}], nothing to compute.")
        else:
            logging.info("Computing batch responses" if current_batch == 0 else f"Resuming from batch {current_batch}")

            iloader = iter(loader)
            for idx in tqdm(range(max_batches)):
                batch = next(iloader)
                if idx >= current_batch:
                    for hook in model.hooks:
                        try:
                            hook.update(batch=batch, batch_idx=idx)
                        except Exception as e:
                            print(f"[ERROR] Hook update failed on batch {idx}: {e}")
                            continue

                    with torch.inference_mode():
                        input_ids = batch["input_ids"].to(args.device)
                        attention_mask = batch["attention_mask"].to(args.device)

                        if "pixel_values" in batch:
                            batch["pixel_values"] = batch["pixel_values"].to(args.device)

                        try:
                            model.module(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                pixel_values=batch.get("pixel_values", None),
                            )
                        except Exception as e:
                            print("[ERROR] model forward failed:", e)
                            continue

                checkpoint["current_batch"] = idx + 1
                torch.save(checkpoint, checkpoint_path)
            logging.info("Done")

    logging.warning(f"Responses saved in {output_path / args.tag}.")

def main(args: argparse.Namespace) -> None:
    args = parsers.merge_config_into_args(args)
    compute_responses(args)

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Compute Responses",
        description="Extracts and saves responses from a LLaVA model",
    )
    parser = parsers.add_config_args(parser)
    parser = parsers.add_responses_args(parser)
    parser = parsers.add_job_args(parser)

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--rand-weights", type=int, default=0)
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
