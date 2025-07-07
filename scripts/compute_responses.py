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

os.environ["TOKENIZERS_PARALLELISM"] = "False"

def get_model_name_from_path(model_path: t.Union[Path, str]) -> str:
    return str(Path(model_path).name)

def compute_responses(args: argparse.Namespace) -> None:
    [dataset, ] = parsers.get_single_args(args, ["dataset"])
    model_name = get_model_name_from_path(args.model_path)

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

    module, processor  = load_huggingface_model(
        model_path=args.model_path,
        cache_dir=parsers.HF_HUB_CACHE,
        device=args.device,
        dtype=args.dtype,
        rand_weights=(args.rand_weights == 1),
        seq_len=args.seq_len,
    )
    
    model = ModelWithHooks(module=module)

    print("[DEBUG] 模型可用模块如下：")
    for name, _ in model.module.named_modules():
        print(name)

    train_dataset, collate_fn = get_dataset(
        name=dataset,
        datasets_folder=Path(args.data_dir),
        split="train",
        tokenizer=tokenizer,
    )

    module_names = model.find_module_names(module, args.module_names)
    print("[DEBUG] 匹配到的模块:", module_names)
    if not module_names:
        raise RuntimeError("未匹配到任何模块，请检查 --module-names 参数是否正确")

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
                pooling_op_names=args.pooling_op,
                output_path=label_output_path,
                save_fields=["id"],
                threaded=False,
            )
            for module_name in module_names
        ]
        model.remove_hooks()
        model.register_hooks(hook_fns)

        if not model.hooks:
            raise RuntimeError("Hooks 未成功注册，请检查模块名是否正确")

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
            continue

        logging.info("Computing batch responses" if current_batch == 0 else f"Resuming from batch {current_batch}")
        iloader = iter(loader)
        for idx in tqdm(range(max_batches)):
            batch = next(iloader)
            if idx >= current_batch:
                with torch.inference_mode():
                    model.update_hooks(batch_idx=idx, batch=batch)
                    input_ids = batch["input_ids"].to(args.device)
                    attention_mask = batch["attention_mask"].to(args.device)
                    try:
                        module(input_ids=input_ids, attention_mask=attention_mask)
                    except hooks.custom_exceptions.TargetModuleReached:
                        pass
            checkpoint["current_batch"] = idx + 1
            torch.save(checkpoint, checkpoint_path)
        logging.info("Done")

    logging.warning(f"Responses saved in {output_path / args.tag}.")
)
