# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import typing as t
from pathlib import Path
import torch

HOOK_LOG_COUNT = 0
HOOK_LOG_LIMIT = 10

def make_hook(alpha: torch.Tensor, module_name: str):
    def hook_fn(module, input, output):
        global HOOK_LOG_COUNT
        out = output * alpha
        if HOOK_LOG_COUNT < HOOK_LOG_LIMIT:
            print(f"[HOOK CALL] ✅ {module_name}")
            print(f"[HOOK ACTIVE] {module_name} | output norm: {output.norm():.4f} → {out.norm():.4f}")
            HOOK_LOG_COUNT += 1
        return out
    return hook_fn

class _DampeningHook:
    def __init__(
        self,
        module_name: str,
        state_path: t.Optional[t.Union[Path, str]] = None,
        alpha: t.Optional[torch.Tensor] = None,
        device: str = "cuda",
    ):
        self.module_name = module_name
        self.device = device
        self.handle = None

        if state_path:
            state_dict = torch.load(state_path, map_location=device)
            alpha = state_dict["alpha"].to(device)
        elif alpha is None:
            raise ValueError("❌ Must provide either state_path or alpha tensor.")

        if len(alpha.shape) == 1:
            alpha = alpha.unsqueeze(0).unsqueeze(0)

        self.alpha = alpha
        print(f"[DEBUG] ✅ Loaded alpha for {module_name}, mean={self.alpha.mean():.4f}, shape={self.alpha.shape}")

    def register(self, model: torch.nn.Module):
        #print(f"[DEBUG] Trying to register hook on: {self.module_name}")
        available_modules = dict(model.named_modules())
        if self.module_name not in available_modules:
            print("[DEBUG] Available modules:")
            for name in available_modules:
                print(f" - {name}")
            raise ValueError(f"❌ 找不到模块: {self.module_name}")
        target = available_modules[self.module_name]
        self.handle = target.register_forward_hook(make_hook(self.alpha, self.module_name))
        #print(f"[DEBUG] ✅ Hook registered on {self.module_name}")
        return self.handle

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            print(f"[DEBUG] 🔧 Hook removed from {self.module_name}")

    def state_dict(self):
        return {"alpha": self.alpha}

class Det0Hook(_DampeningHook):
    pass

class DampHook(_DampeningHook):
    pass

class AURAHook(_DampeningHook):
    pass
