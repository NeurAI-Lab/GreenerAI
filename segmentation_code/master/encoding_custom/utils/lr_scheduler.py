##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import math
from random import uniform


class LR_Scheduler(object):
    """Learning Rate Scheduler
    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``
    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``
    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``
    CLR mode: ``lr = baselr * triangle``

    if rand_grad is True:   lr = lr * r  where r ~ U[0,1]

    Args:
        args:  :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`

        iters_per_epoch: number of iterations per epoch
    """

    def __init__(
        self,
        mode,
        base_lr,
        num_epochs,
        iters_per_epoch=0,
        lr_step=0,
        warmup_epochs=0,
        stepsize=None,
        max_lr=None,
        rand_grad=False,
    ):
        self.mode = mode
        self.random = rand_grad
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = base_lr
        if mode == "step":
            assert lr_step
        if mode == "clr":
            assert stepsize is not None
            assert max_lr is not None
            self.stepsize = stepsize
            self.max_lr = max_lr
            self.min_lr = base_lr
            self.triangular2 = True
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch

    def _relative(self, i, stepsize):
        cycle = math.floor(1 + i / (2 * stepsize))
        x = abs(i / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)), cycle

    def __call__(self, optimizer, i, epoch, best_pred):
        T = epoch * self.iters_per_epoch + i
        if self.mode == "cos":
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == "poly":
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == "step":
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
        elif self.mode == "clr":
            factor, cycle = self._relative(i + epoch * self.iters_per_epoch, self.stepsize)
            lr = self.min_lr + (self.max_lr - self.min_lr) * factor
            if self.triangular2:
                lr = lr / 2 ** (cycle - 1)
        else:
            raise NotImplemented

        if self.random:
            lr *= uniform(0, 1)

        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        if epoch > self.epoch:
            print(
                (
                    f"=>Epoch {epoch+1}, learning rate = {lr:.4f}, "
                    f"previous best = {best_pred:.4f}"
                )
            )
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)
        return lr

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]["lr"] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]["lr"] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]["lr"] = lr * 10
