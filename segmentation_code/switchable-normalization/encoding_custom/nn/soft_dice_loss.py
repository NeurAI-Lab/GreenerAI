import torch


class SoftDiceLoss(torch.nn.Module):
    def __init__(self, num_class, epsilon=1e-7):
        super(SoftDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.num_class = num_class

    def forward(self, input, target, ignore_index=-1):
        mask = target == ignore_index
        target[mask] = 0
        tt = self._make_one_hot(target)
        mask = mask.unsqueeze(1).expand_as(tt)
        tt[mask] = 0
        mask = mask.float()
        input = input * (1 - mask)
        numerator = 2.0 * torch.sum(input * tt, dim=(-1, -2))
        denominator = torch.sum(input * input + tt * tt, dim=(-1, -2))

        return 1.0 - torch.mean(numerator / (denominator + self.epsilon))

    def _make_one_hot(self, labels):
        """
        Converts an integer label torch.autograd.Variable to a one-hot Variable.

        Parameters
        ----------
        labels : torch.autograd.Variable of torch.cuda.LongTensor
            N x 1 x H x W, where N is batch size.
            Each value is an integer representing correct classification.
        C : integer.
            number of classes in labels.

        Returns
        -------
        target : torch.autograd.Variable of torch.cuda.FloatTensor
            N x C x H x W, where C is class number. One-hot encoded.
        """
        one_hot = torch.zeros(
            (labels.shape[0], self.num_class, labels.shape[1], labels.shape[2]),
            dtype=torch.float32,
        ).cuda()

        target = one_hot.scatter_(1, labels.unsqueeze(1).data, 1)
        return target
