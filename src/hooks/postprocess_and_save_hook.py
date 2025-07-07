import typing as t
from .pooling_ops import get_pooling_op
from pathlib import Path
import torch
import logging
from collections import defaultdict
import threading
import os


class PostprocessAndSaveHook(torch.nn.Module):
    def __init__(
        self,
        module_name: str,
        pooling_op_names: t.List[str],
        output_path: t.Optional[Path],
        save_fields: t.List[str],
        return_outputs: bool = False,
        threaded: bool = True,
    ):
        super().__init__()
        self.module_name = module_name
        self.pooling_ops = torch.nn.ModuleList(
            [get_pooling_op(name, dim=1) for name in pooling_op_names]
        )
        self.output_path = output_path
        self.save_fields = save_fields
        self.return_outputs = return_outputs
        self.batch_idx = None
        self.attention_mask = None
        self.threaded = threaded
        self.thread_handle = None
        self.outputs = defaultdict(list)

    def update(self, batch_idx: int, batch: dict) -> None:
        assert "id" in batch
        self.batch_idx = batch_idx
        self.batch = batch
        self.outputs.clear()

    def save(self, module_name: str, output: torch.Tensor) -> None:
        for pooling_op in self.pooling_ops:
            attention_mask = self.batch.get("attention_mask")
            pooled_output = pooling_op(output.detach().clone(), attention_mask=attention_mask)

            for sample_index in range(len(pooled_output)):
                datum = {}
                sample_id = self.batch["id"][sample_index]
                sample_outputs = pooled_output[sample_index].cpu()

                for field in self.save_fields:
                    if field in self.batch:
                        datum[field] = self.batch[field][sample_index]

                if pooling_op.name == "all" and attention_mask is not None:
                    sample_outputs = sample_outputs[attention_mask[sample_index].bool()]

                datum.update({"responses": sample_outputs})

                if self.output_path is not None:
                    output_path = (
                        self.output_path
                        / module_name
                        / pooling_op.name
                        / f"{sample_id}.pt"
                    )
                    os.makedirs(output_path.parent, exist_ok=True)
                    torch.save(datum, output_path)

                if self.return_outputs:
                    self.outputs[module_name].append(datum)

    def __call__(self, module, input, output) -> None:
        print(f"[HOOK CALL] {self.module_name} called")
        assert self.batch_idx is not None, "update() must be called before executing the hook"

        def _hook(module_name: str, output: t.Any):
            if isinstance(output, torch.Tensor):
                self.save(module_name, output)
            elif isinstance(output, (list, tuple)):
                for idx in range(len(output)):
                    _hook(f"{module_name}:{idx}", output[idx])
            else:
                logging.warning(f"Unsupported output type {type(output)} in {self.module_name}")

        if self.threaded:
            self.thread_handle = threading.Thread(
                target=_hook, args=(self.module_name, output)
            )
            self.thread_handle.start()
        else:
            _hook(self.module_name, output)
