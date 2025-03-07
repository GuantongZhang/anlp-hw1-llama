from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            max_grad_norm: float = None,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias, max_grad_norm=max_grad_norm)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            # TODO: Clip gradients if max_grad_norm is set
            if group['max_grad_norm'] is not None:
                torch.nn.utils.clip_grad_norm_(group["params"], group["max_grad_norm"])

            
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                if p not in self.state:
                    self.state[p] = {}
                    self.state[p]["step"] = 0
                    self.state[p]["exp_avg"] = torch.zeros_like(p.data)
                    self.state[p]["exp_avg_sq"] = torch.zeros_like(p.data)

                state = self.state[p]

                # TODO: Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                weight_decay = group["weight_decay"]
                correct_bias = group["correct_bias"]

                # TODO: Update first and second moments of the gradients
                state["exp_avg"] = beta1 * state["exp_avg"] + (1 - beta1) * grad
                state["exp_avg_sq"] = beta2 * state["exp_avg_sq"] + (1 - beta2) * (grad ** 2)
                state["step"] += 1

                # TODO: Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                if correct_bias:
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    step_size = alpha * (bias_correction2 ** 0.5) / bias_correction1
                else:
                    step_size = alpha

                # TODO: Update parameters
                p.data -= step_size * state["exp_avg"] / (state["exp_avg_sq"].sqrt() + eps)

                # TODO: Add weight decay after the main gradient-based updates.
                if weight_decay > 0:
                    p.data -= alpha * weight_decay * p.data

                # Please note that the learning rate should be incorporated into this update.

        return loss