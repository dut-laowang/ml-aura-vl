import pathlib
import re
import typing as t
from pathlib import Path

import torch
from torch.utils.hooks import RemovableHandle
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers import LlavaForConditionalGeneration
from transformers import MllamaForConditionalGeneration

TASK_MAPPING = {
    "text-generation": AutoModelForCausalLM,
    "sequence-classification": AutoModelForSequenceClassification,
}

def is_module_name_in_regex(module_name: str, regex_list: t.List[str]) -> bool:
    return any(re.fullmatch(regex, module_name) for regex in regex_list)

def load_huggingface_model(
    model_path: t.Union[str, Path],
    cache_dir: t.Optional[t.Union[str, Path]],
    dtype: t.Any,
    device: str,
    seq_len: t.Optional[int] = None,
    rand_weights: bool = False,
    task: str = "text-generation",
) -> t.Tuple[PreTrainedModel, t.Any]:
    """
    Loads a Hugging Face model and processor/tokenizer.
    Supports LLaVA, LLaMA 3.2 Vision Instruct, and fallback HuggingFace models.
    """
    model_path_str = str(model_path).lower()

    # LLaVA support
    if "llava" in model_path_str:
        from transformers import AutoProcessor
        print(f"[DEBUG] Detected LLaVA model path: {model_path}")
        tokenizer = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            torch_dtype=getattr(torch, dtype),
        )
        model = model.to(device)
        print(f"[DEBUG] Loaded LLaVA model: {type(model)}")
        print(f"[DEBUG] Loaded LLaVA processor: {type(tokenizer)}")
        return model, tokenizer

    # LLaMA 3.2 Vision Instruct support
    if "Llama-3.2" in model_path_str or "vision" in model_path_str:
        from transformers import AutoProcessor
        print(f"[DEBUG] Detected LLaMA 3.2 Vision Instruct model path: {model_path}")
        processor = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir, token="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")
        model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            torch_dtype=getattr(torch, dtype),
            device_map="auto" if device == "auto" else None,
            token="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"
        )
        if device != "auto":
            model = model.to(device)
        print(f"[DEBUG] Loaded LLaMA 3.2 model: {type(model)}")
        print(f"[DEBUG] Loaded LLaMA 3.2 processor: {type(processor)}")
        return model, processor

    # Fallback to generic HuggingFace models
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.get_default_dtype()

    model_class = TASK_MAPPING.get(task, AutoModel)
    cache_dir = Path(cache_dir).expanduser().absolute()

    full_model_path = cache_dir / model_path
    if full_model_path.exists():
        model_path = full_model_path

    print(f"[DEBUG] Fallback loading model path: {model_path}")

    if rand_weights:
        config = AutoConfig.from_pretrained(model_path, cache_dir=cache_dir)
        model = model_class.from_config(config)
        from transformers import AutoProcessor
        tokenizer = AutoProcessor.from_pretrained(model_path)
        print("[DEBUG] Loaded model with random weights")
    else:
        tokenizer = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir)
        model = model_class.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            torch_dtype=getattr(torch, dtype),
        )

    model = model.to(device)

    if seq_len:
        tokenizer.model_max_length = seq_len
    tokenizer.padding_side = "left"
    if getattr(tokenizer, "pad_token", None) is None:
        if getattr(tokenizer, "bos_token", None):
            tokenizer.pad_token = tokenizer.bos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            model.resize_token_embeddings(len(tokenizer))
            tokenizer.pad_token = "<pad>"

    print(f"[DEBUG] Loaded fallback tokenizer: {type(tokenizer)}")
    return model, tokenizer



class ModelWithHooks:
    def __init__(
        self,
        module: torch.nn.Module,
        hooks: t.Optional[t.List[torch.nn.Module]] = None,
    ) -> None:
        self.module = module
        self.hooks: t.Dict[str, torch.nn.Module] = (
            {h.module_name: h for h in hooks} if hooks is not None else {}
        )
        self._forward_hook_handles: t.List[RemovableHandle] = []

    @staticmethod
    def find_module_names(
        model: torch.nn.Module,
        regex_list: t.List[str],
        skip_first: int = 0,
    ) -> t.List[str]:
        module_names = [name for name, _ in model.named_modules()][skip_first:]
        return [name for name in module_names if is_module_name_in_regex(name, regex_list)]

    def get_module(self) -> torch.nn.Module:
        return self.module

    def register_hooks(self, hooks=None):
        if hooks is not None:
            assert len(self.hooks) == 0, "Hooks already registered"
            self.hooks = {h.module_name: h for h in hooks}

        self._forward_hook_handles = []
        for hook in self.hooks.values():
            if hasattr(hook, "register") and callable(hook.register):
                handle = hook.register(self.module)
                if handle is not None:
                    self._forward_hook_handles.append(handle)
            else:
                raise TypeError(f"❌ Hook {hook} has no callable 'register' method")

    def remove_hooks(self):
        for h in self._forward_hook_handles:
            h.remove()
        self._forward_hook_handles = []
        self.hooks = {}

    def get_hook_outputs(self):
        outputs = {}
        for name, hook in self.hooks.items():
            outputs.update(hook.outputs)
        return outputs

    def update_hooks(self, *args, **kwargs):
        assert self.hooks
        for hook in self.hooks.values():
            if hasattr(hook, "update") and callable(hook.update):
                hook.update(*args, **kwargs)

    def get_target_module_names(self) -> t.List[str]:
        return list(self.hooks.keys())
