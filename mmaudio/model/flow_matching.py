import logging
from typing import Callable, Iterable, Optional

import torch
from torchdiffeq import odeint

# from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher

log = logging.getLogger()
from comfy.utils import ProgressBar
from tqdm import tqdm

# Partially from https://github.com/gle-bellier/flow-matching
class FlowMatching:

    def __init__(self, min_sigma: float = 0.0, inference_mode='euler', num_steps: int = 25, shift: float = 1.0):
        # inference_mode: 'euler', 'midpoint', 'heun', 'rk4', or 'adaptive'
        # num_steps: number of steps in the ODE inference mode
        # shift: time shift schedule factor (1.0 = linear, > 1.0 shifts towards noise, < 1.0 shifts towards data)
        super().__init__()
        self.min_sigma = min_sigma
        self.inference_mode = inference_mode
        self.num_steps = num_steps
        self.shift = shift

        assert self.inference_mode in ['euler', 'midpoint', 'heun', 'rk4', 'adaptive'], f"Unsupported inference mode: {self.inference_mode}"
        if self.inference_mode == 'adaptive' and num_steps > 0:
            log.info('The number of steps is ignored in adaptive inference mode')

    def get_conditional_flow(self, x0: torch.Tensor, x1: torch.Tensor,
                             t: torch.Tensor) -> torch.Tensor:
        # which is psi_t(x), eq 22 in flow matching for generative models
        t = t[:, None, None].expand_as(x0)
        return (1 - (1 - self.min_sigma) * t) * x0 + t * x1

    def loss(self, predicted_v: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        # return the mean error without reducing the batch dimension
        reduce_dim = list(range(1, len(predicted_v.shape)))
        target_v = x1 - (1 - self.min_sigma) * x0
        return (predicted_v - target_v).pow(2).mean(dim=reduce_dim)

    def get_x0_xt_c(
        self,
        x1: torch.Tensor,
        t: torch.Tensor,
        Cs: list[torch.Tensor],
        generator: Optional[torch.Generator] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x0 = torch.randn_like(x1, generator=generator)
        x0 = torch.empty_like(x1).normal_(generator=generator)

        # find mini-batch optimal transport
        # x0, x1, _, Cs = self.fm.ot_sampler.sample_plan_with_labels(x0, x1, None, Cs, replace=True)

        xt = self.get_conditional_flow(x0, x1, t)
        return x0, x1, xt, Cs

    def to_prior(self, fn: Callable, x1: torch.Tensor) -> torch.Tensor:
        return self.run_t0_to_t1(fn, x1, 1, 0)

    def to_data(self, fn: Callable, x0: torch.Tensor) -> torch.Tensor:
        return self.run_t0_to_t1(fn, x0, 0, 1)

    def run_t0_to_t1(self, fn: Callable, x0: torch.Tensor, t0: float, t1: float) -> torch.Tensor:
        # fn: a function that takes (t, x) and returns the direction x0->x1

        if self.inference_mode == 'adaptive':
            out = odeint(fn, x0, torch.tensor([t0, t1], device=x0.device, dtype=x0.dtype), method='dopri5')
            return out[-1]

        # Time step construction
        raw_steps = torch.linspace(t0, t1 - self.min_sigma, self.num_steps + 1)
        if self.shift != 1.0 and abs(t1 - t0) > 0:
            # Apply time shift warping: t_shifted = (shift * t) / (1 + (shift - 1) * t)
            normalized_t = (raw_steps - t0) / (t1 - t0)
            warped_t = (self.shift * normalized_t) / (1.0 + (self.shift - 1.0) * normalized_t)
            steps = t0 + warped_t * (t1 - t0)
        else:
            steps = raw_steps

        x = x0
        comfy_pbar = ProgressBar(len(steps) - 1)
        tqdm_pbar = tqdm(total=len(steps) - 1, desc=f'Flow Matching ({self.inference_mode})')

        for ti in range(len(steps) - 1):
            t = steps[ti]
            next_t = steps[ti + 1]
            dt = next_t - t

            if self.inference_mode == 'euler':
                flow = fn(t, x)
                x = x + dt * flow

            elif self.inference_mode == 'midpoint':
                # 2nd-order Midpoint Method (RK2)
                k1 = fn(t, x)
                t_mid = t + 0.5 * dt
                x_mid = x + 0.5 * dt * k1
                k2 = fn(t_mid, x_mid)
                x = x + dt * k2

            elif self.inference_mode == 'heun':
                # 2nd-order Heun's Method (Improved Euler)
                k1 = fn(t, x)
                x_pred = x + dt * k1
                k2 = fn(next_t, x_pred)
                x = x + 0.5 * dt * (k1 + k2)

            elif self.inference_mode == 'rk4':
                # 4th-order Classical Runge-Kutta
                k1 = fn(t, x)
                t_mid = t + 0.5 * dt
                k2 = fn(t_mid, x + 0.5 * dt * k1)
                k3 = fn(t_mid, x + 0.5 * dt * k2)
                k4 = fn(next_t, x + dt * k3)
                x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            comfy_pbar.update(1)
            tqdm_pbar.update(1)

        return x
