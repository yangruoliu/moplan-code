"""Diffusion-DPO on flow matching: prefer the clip with the better physics, relative to a frozen reference policy.

For a pair (winner w, loser l) generated under the same condition, noised with the SAME sigma and noise:

    e_x(theta) = || v_theta(z_sigma(x)) - (eps - x) ||^2      flow-matching error of the trained model on clip x
    r_x        = the same error of the frozen reference
    d          = (e_w - r_w) - (e_l - r_l)
    L          = -log sigmoid(-beta * d)

L falls when the trained model explains the winner better than the reference does, and the loser worse.  With
theta = reference, d = 0 and L = log 2.

Two-pass form (memory: one clip with gradients at a time): dL/dd = beta * sigmoid(beta * d) =: w is a scalar that can be
computed without gradients; then  grad L = w * grad e_w - w * grad e_l, i.e. two ordinary backward passes with the
losses  +w * e_w  and  -w * e_l.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_terms(e_win: torch.Tensor, r_win: torch.Tensor, e_lose: torch.Tensor, r_lose: torch.Tensor, beta: float) -> dict:
    """Per-pair {"loss", "weight" (= dL/dd, detached), "d"} from per-sample errors [B]."""
    d = (e_win - r_win) - (e_lose - r_lose)
    return {"loss": -F.logsigmoid(-beta * d), "weight": (beta * torch.sigmoid(beta * d)).detach(), "d": d}
