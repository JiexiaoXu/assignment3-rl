import torch
import torch.optim as optim
from typing import Optional


class AdamW(optim.Optimizer):
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        """ Initialize your optimizer
        
        Arguments:
            params: iterable
                A collection of parameters to optimize, or parameter groups
                (for applying different hyperparameters to different parts of the model).
        
        Additional arguments depending on the optimizer
        (e.g., learning rate is common).
        ...
        """
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)


    @torch.cuda.nvtx.range("AdamW step", color="green")
    def step(self, closure: Optional[callable]=None):
        loss = None if closure is None else closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                # gradient
                d_p = p.grad
                grad = d_p.data
                
                state = self.state[p]
                t = state.get('t', 0) + 1

                m = state.get('m', torch.zeros_like(p.data))
                v = state.get('v', torch.zeros_like(p.data))
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad * grad)

                state['m'] = m
                state['v'] = v
                state['t'] = t

                adjust_lr_numerator = (1 - beta2 ** t) ** 0.5
                adjust_lr_denominator = (1 - beta1 ** t)
                adjust_lr = lr * (adjust_lr_numerator / adjust_lr_denominator)
                p.data = p.data - adjust_lr * (m / (v ** 0.5 + eps))
                p.data = p.data - lr * weight_decay * p.data

        return loss
