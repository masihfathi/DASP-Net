from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class PromptRefiner(nn.Module):
    def __init__(self, in_channels: int = 7, prompt_channels: int = 4,
                 hidden_channels: int = 32, initial_residual_scale: float = 0.10) -> None:
        super().__init__()
        self.refiner = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, prompt_channels, 3, padding=1),
            nn.Tanh(),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(initial_residual_scale)))

    def forward(self, dasp_input: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if dasp_input.ndim != 4 or dasp_input.shape[1] != 7:
            raise ValueError(f"Expected [B, 7, H, W], got {tuple(dasp_input.shape)}")
        rgb = dasp_input[:, :3]
        prompts = dasp_input[:, 3:7]
        delta = self.refiner(dasp_input)
        refined = torch.clamp(prompts + self.residual_scale * delta, 0.0, 1.0)
        return torch.cat([rgb, refined], dim=1), {
            "original_prompts": prompts,
            "refined_prompts": refined,
            "prompt_delta": delta,
            "residual_scale": self.residual_scale,
        }


class PromptReliabilityGate(nn.Module):
    def __init__(self, in_channels: int = 7, prompt_channels: int = 4,
                 hidden_channels: int = 32) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, prompt_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, dasp_input: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if dasp_input.ndim != 4 or dasp_input.shape[1] != 7:
            raise ValueError(f"Expected [B, 7, H, W], got {tuple(dasp_input.shape)}")
        rgb = dasp_input[:, :3]
        prompts = dasp_input[:, 3:7]
        reliability = self.gate(dasp_input)
        gated = prompts * reliability
        return torch.cat([rgb, gated], dim=1), {
            "original_prompts": prompts,
            "reliability": reliability,
            "gated_prompts": gated,
        }


class HybridLearnablePromptModule(nn.Module):
    def __init__(self, hidden_channels: int = 32,
                 initial_residual_scale: float = 0.10) -> None:
        super().__init__()
        self.refiner = PromptRefiner(
            hidden_channels=hidden_channels,
            initial_residual_scale=initial_residual_scale,
        )
        self.gate = nn.Sequential(
            nn.Conv2d(7, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 4, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, dasp_input: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        refined_input, aux = self.refiner(dasp_input)
        rgb = refined_input[:, :3]
        refined = refined_input[:, 3:7]
        reliability = self.gate(dasp_input)
        final_prompts = refined * reliability
        return torch.cat([rgb, final_prompts], dim=1), {
            **aux,
            "reliability": reliability,
            "final_prompts": final_prompts,
        }


class LearnablePromptGatedDASPNet(nn.Module):
    def __init__(self, backbone: nn.Module, adapter_mode: str = "hybrid",
                 hidden_channels: int = 32,
                 initial_residual_scale: float = 0.10) -> None:
        super().__init__()
        mode = adapter_mode.lower().strip()
        if mode == "refine":
            adapter = PromptRefiner(hidden_channels=hidden_channels,
                                    initial_residual_scale=initial_residual_scale)
        elif mode == "gate":
            adapter = PromptReliabilityGate(hidden_channels=hidden_channels)
        elif mode == "hybrid":
            adapter = HybridLearnablePromptModule(
                hidden_channels=hidden_channels,
                initial_residual_scale=initial_residual_scale,
            )
        else:
            raise ValueError("adapter_mode must be refine, gate, or hybrid")
        self.backbone = backbone
        self.prompt_adapter = adapter
        self.adapter_mode = mode

    def forward(self, dasp_input: torch.Tensor, return_aux: bool = False):
        adapted_input, aux = self.prompt_adapter(dasp_input)
        output = self.backbone(adapted_input)
        if return_aux:
            return output, {**aux, "adapted_input": adapted_input}
        return output


def build_existing_pgdasp_backbone() -> nn.Module:
    from model_unet import build_prompt_gated_dasp_net
    errors = []
    for kwargs in ({}, {"in_channels": 7}, {"input_channels": 7}):
        try:
            return build_prompt_gated_dasp_net(**kwargs)
        except TypeError as exc:
            errors.append(f"{kwargs}: {exc}")
    raise RuntimeError("Could not build PG-DASP backbone:\n" + "\n".join(errors))


def build_lpgdasp_net(adapter_mode: str = "hybrid", hidden_channels: int = 32,
                       initial_residual_scale: float = 0.10) -> LearnablePromptGatedDASPNet:
    return LearnablePromptGatedDASPNet(
        backbone=build_existing_pgdasp_backbone(),
        adapter_mode=adapter_mode,
        hidden_channels=hidden_channels,
        initial_residual_scale=initial_residual_scale,
    )
