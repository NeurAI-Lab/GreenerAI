import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha).cuda()
        self.size_average = size_average

    def forward(self, input, target, ignore_index=-1):
        """
        :param input: logits: N, C, H, W
        :param target: labels: N, H, W  (grayscale intensity encoding)
        :param ignore_index: int (class label to ignore)
        :return:
        """
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        log_p_t = F.log_softmax(input, dim=-1)

        # cv2.imwrite('name.png', target[0].cpu().numpy())
        target = target.view(-1, 1)  # N,H,W => N*H*W

        # ignore
        mask = target == ignore_index
        mask = mask.view(-1)
        target[mask] = 0  # technically not correct, but ignore_class will reset (+++)
        log_p_t = log_p_t.gather(1, target)  # regular Cross-Entropy
        log_p_t = log_p_t.view(-1)
        log_p_t[mask] = 0  # (+++) this line makes ignore_class work

        p_t = F.softmax(input, dim=-1)
        p_t = p_t.gather(1, target)
        p_t = p_t.view(-1)

        loss = -1 * (1 - p_t).pow(self.gamma) * log_p_t

        if self.alpha is not None:
            target = target.view(-1).long()
            alpha_t = self.alpha.gather(0, target)  # ignored are already 0 (+++)
            loss = loss * alpha_t

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
