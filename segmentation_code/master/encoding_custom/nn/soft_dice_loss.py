import torch
import torch.nn.functional as F


class SoftDiceLoss(torch.nn.Module):
    def __init__(self, num_class, epsilon=1e-7):
        super(SoftDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.num_class = num_class

    def forward(self, input, target):
        """
        :param input: logits: N, C, H, W
        :param target: labels: N, H, W
        """
        IGNORE_INDEX = -1
        mask = target != IGNORE_INDEX
        mask = mask.unsqueeze(1).expand(-1, self.num_class, -1, -1).float()

        input = F.softmax(input, dim=1)
        input = input * mask
        dim = tuple(range(1, len(input.shape) - 1))

        # not supported in that version of pytorch
        # tt = F.one_hot(target + 1, self.num_class + 1)[:, :, :, 1:].permute(0, 3, 1, 2).float()
        # #                       ^-------------------^---- ignoring the ignore class -1
        tt = self._make_one_hot(target).float()

        numerator = 2.0 * torch.sum(input * tt, dim=(2, 3))
        denominator = torch.sum(input + tt, dim=(2, 3))
        dice = numerator / (denominator + self.epsilon)
        return 1.0 - torch.mean(dice)

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
            (labels.shape[0], self.num_class + 1, labels.shape[1], labels.shape[2]),
            dtype=torch.float32,
        ).cuda()

        target = one_hot.scatter_(1, (labels + 1).unsqueeze(1).data, 1)
        target = target[:, 1:, :, :]
        return target


if __name__ == "__main__":
    a = SoftDiceLoss(3)
    input1 = (
        torch.tensor(
            [
                [[[1.0, 1.0]], [[1.0, 1.0]], [[0.0, 0.0]]],
                [[[1.0, 1.0]], [[1.0, 1.0]], [[0.0, 0.0]]],
            ]
        )
        .float()
        .cuda()
    )
    input2 = (
        torch.tensor(
            [
                [[[0.0, 0.0]], [[0.0, 0.0]], [[1.0, 1.0]]],
                [[[0.0, 0.0]], [[0.0, 0.0]], [[1.0, 1.0]]],
            ]
        )
        .float()
        .cuda()
    )
    target = 2 * torch.ones((2, 1, 2)).long().cuda()
    r1 = a.forward(input1, target)
    r2 = a.forward(input2, target)
    # Expected results 1 and 0 respectively after commenting out "input = F.softmax(input, dim=1)"
    print(r1, r2)
