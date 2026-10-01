"""Muon for hidden matrices, AdamW for embeddings and scalar/vector/output parameters."""
from __future__ import annotations

import torch
from torch import nn


class HybridOptimizer:
    def __init__(self, muon, adamw, names):
        self.muon, self.adamw, self.names = muon, adamw, names

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.muon.step()
        self.adamw.step()

    def state_dict(self):
        return {"kind": "muon_adamw", "names": self.names,
                "muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state):
        if state.get("kind") != "muon_adamw" or state.get("names") != self.names:
            raise ValueError("optimizer checkpoint parameter assignment differs")
        self.muon.load_state_dict(state["muon"])
        self.adamw.load_state_dict(state["adamw"])


def build_optimizer(model, config):
    kind = config.get("optimizer", "adamw")
    kwargs = dict(lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    if kind == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if kind != "muon":
        raise ValueError("optimizer must be adamw or muon")
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("Muon requires a PyTorch build exposing torch.optim.Muon; refresh setup")
    embedding_ids = {id(p) for module in model.modules() if isinstance(module, nn.Embedding)
                     for p in module.parameters()}
    groups, names = {"muon": [], "adamw": []}, {"muon": [], "adamw": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        use_muon = (parameter.ndim == 2 and id(parameter) not in embedding_ids
                    and not name.startswith("scoring_head."))
        key = "muon" if use_muon else "adamw"
        groups[key].append(parameter)
        names[key].append(name)
    if not all(groups.values()):
        raise ValueError("Muon requires both hidden matrix and auxiliary parameter groups")
    muon = torch.optim.Muon(groups["muon"], lr=float(config.get("muon_learning_rate", 0.0002)),
                            weight_decay=kwargs["weight_decay"],
                            momentum=float(config.get("muon_momentum", 0.95)),
                            ns_steps=int(config.get("muon_ns_steps", 5)),
                            adjust_lr_fn="match_rms_adamw")
    print(f"Optimizer: Muon ({sum(p.numel() for p in groups['muon'])} parameters) + "
          f"AdamW ({sum(p.numel() for p in groups['adamw'])} parameters)", flush=True)
    return HybridOptimizer(muon, torch.optim.AdamW(groups["adamw"], **kwargs), names)
