import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

__all__ = ["SegmentationLosses", "OhemCrossEntropy2d", "OHEMSegmentationLosses"]

# ignore_index=-1 is hardcoded in this method.
class ClassBalancedSegmentationLossesWithLabelRelaxation(nn.Module):
    def __init__(
        self,
        se_loss=False,
        se_weight=0.2,
        nclass=-1,
        aux=False,
        aux_weight=0.4,
        weight=None,
        ignore_index=-1,
        num_outputs=1,
        aux_indexes=[],
        beta=1 - 1e-3,
    ):
        super(ClassBalancedSegmentationLossesWithLabelRelaxation, self).__init__()

        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.bceloss = nn.BCELoss(weight)
        self.normal_loss_indexes = [
            i for i in range(num_outputs) if i not in aux_indexes
        ]
        self.aux_indexes = aux_indexes
        self.num_outputs = num_outputs
        self.beta = beta

        if len(aux_indexes) > 0:
            self.aux_weight_per_loss = aux_weight / len(aux_indexes)
        else:
            self.aux_weight_per_loss = 0

        self.ignore_index = ignore_index

    def forward(self, *inputs):
        ignore_mask = inputs[-1] == -1

        label_one_hot = self._make_one_hot(inputs[-1], self.nclass)
        label_one_hot = label_one_hot.float()
        label_one_hot_edges = F.max_pool2d(label_one_hot, (5, 5), 1, 2)
        # edge_mask = torch.sum(label_one_hot_edges, dim=1) <= 1
        # inputs[-1][edge_mask] = -1
        border_weights = label_one_hot_edges.sum(1)
        # ignore_mask = (border_weights == 0)
        border_weights[ignore_mask] = 1
        ignore_mask = ignore_mask.unsqueeze(1).expand(label_one_hot.shape)
        label_one_hot_edges[ignore_mask] = 0

        if len(inputs) == 2:
            normal_loss_indexes = [0]
            aux_indexes = []
        else:
            normal_loss_indexes = self.normal_loss_indexes
            aux_indexes = self.aux_indexes

        loss = 0
        weights = self._class_balanced_weights(
            inputs[-1], self.nclass, self.beta
        ).type_as(inputs[0])
        for ind in normal_loss_indexes:
            pred = inputs[ind]
            loss += (
                -1
                / border_weights.float()
                * (
                    label_one_hot_edges.float()
                    * weights.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                    * self.customsoftmax(pred, label_one_hot_edges.float())
                ).sum(1)
            ).mean()

        for ind in aux_indexes:
            pred = inputs[ind]
            loss += self.aux_weight_per_loss * F.cross_entropy(
                pred, inputs[-1], ignore_index=-1
            )

        return loss

    @staticmethod
    def customsoftmax(inp, multihotmask):
        """
        Custom Softmax
        """
        soft = F.softmax(inp, 1)
        # This takes the mask * softmax ( sums it up hence summing up the classes in border
        # then takes of summed up version vs no summed version
        return torch.log(
            torch.max(soft, (multihotmask * (soft * multihotmask).sum(1, keepdim=True)))
        )

    @staticmethod
    def _class_balanced_weights(target, nclass, beta=1 - 1e-3):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = torch.zeros(batch, nclass)
        for i in range(batch):
            hist = torch.histc(
                target[i].cpu().data.float(), bins=nclass, min=0, max=nclass - 1
            )
            # vect = hist>0
            tvect[i] = hist
        tvect_sum = torch.sum(tvect, 0).float()
        tvect_sum = ((tvect_sum != 0).float() * 1.0 * (1 / tvect_sum)) + 1
        tvect_sum[torch.isnan(tvect_sum)] = 0
        return tvect_sum

    @staticmethod
    def _make_one_hot(labels, num_class):
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
            (labels.shape[0], num_class + 1, labels.shape[1], labels.shape[2])
        ).type_as(labels)

        target = one_hot.scatter_(1, (labels + 1).unsqueeze(1).data, 1)
        target = target[:, 1:, :, :]
        return target


class ClassBalancedSegmentationLosses(nn.Module):
    def __init__(
        self,
        se_loss=False,
        se_weight=0.2,
        nclass=-1,
        aux=False,
        aux_weight=0.4,
        weight=None,
        ignore_index=-1,
        num_outputs=1,
        aux_indexes=[],
        beta=1 - 1e-3,
    ):
        super(ClassBalancedSegmentationLosses, self).__init__()

        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.bceloss = nn.BCELoss(weight)
        self.normal_loss_indexes = [
            i for i in range(num_outputs) if i not in aux_indexes
        ]
        self.aux_indexes = aux_indexes
        self.num_outputs = num_outputs
        self.beta = beta

        if len(aux_indexes) > 0:
            self.aux_weight_per_loss = aux_weight / len(aux_indexes)
        else:
            self.aux_weight_per_loss = 0

        self.ignore_index = ignore_index

    def forward(self, *inputs):
        if len(inputs) == 2:
            normal_loss_indexes = [0]
            aux_indexes = []
        else:
            normal_loss_indexes = self.normal_loss_indexes
            aux_indexes = self.aux_indexes

        loss = 0
        weights = self._class_balanced_weights(
            inputs[-1], self.nclass, self.beta
        ).type_as(inputs[0])
        for ind in normal_loss_indexes:
            pred = inputs[ind]
            loss += F.cross_entropy(pred, inputs[-1], weight=weights, ignore_index=-1)

        for ind in aux_indexes:
            pred = inputs[ind]
            loss += self.aux_weight_per_loss * F.cross_entropy(
                pred, inputs[-1], weight=weights, ignore_index=-1
            )

        return loss

    @staticmethod
    def _class_balanced_weights(target, nclass, beta=1 - 1e-3):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = torch.zeros(batch, nclass)
        for i in range(batch):
            hist = torch.histc(
                target[i].cpu().data.float(), bins=nclass, min=0, max=nclass - 1
            )
            # vect = hist>0
            tvect[i] = hist
        tvect_sum = torch.sum(tvect, 0)
        tvect_sum = (1 - beta) / (1 - beta ** (tvect_sum))
        tvect_sum[tvect_sum == np.inf] = 0
        return tvect_sum


class SegmentationLosses(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""

    def __init__(
        self,
        se_loss=False,
        se_weight=0.2,
        nclass=-1,
        aux=False,
        aux_weight=0.4,
        weight=None,
        ignore_index=-1,
        num_outputs=1,
        aux_indexes=[],
    ):
        super(SegmentationLosses, self).__init__(weight, None, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.bceloss = nn.BCELoss(weight)
        self.normal_loss_indexes = [
            i for i in range(num_outputs) if i not in aux_indexes
        ]
        self.aux_indexes = aux_indexes
        self.num_outputs = num_outputs
        if len(aux_indexes) > 0:
            self.aux_weight_per_loss = aux_weight / len(aux_indexes)
        else:
            self.aux_weight_per_loss = 0

    def forward(self, *inputs):
        if len(inputs) == 2:
            loss = super(SegmentationLosses, self).forward(inputs[0], inputs[-1])
            return loss
        else:
            assert (
                len(inputs) - 1
            ) == self.num_outputs, "number of outputs should match num_outputs"
            if not self.se_loss:
                loss = 0
                for ind in self.normal_loss_indexes:
                    pred = inputs[ind]
                    loss += super(SegmentationLosses, self).forward(pred, inputs[-1])

                for ind in self.aux_indexes:
                    pred = inputs[ind]
                    loss += self.aux_weight_per_loss * super(
                        SegmentationLosses, self
                    ).forward(pred, inputs[-1])

                return loss

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(
                target[i].cpu().data.float(), bins=nclass, min=0, max=nclass - 1
            )
            vect = hist > 0
            tvect[i] = vect
        return tvect


# adapted from https://github.com/PkuRainBow/OCNet/blob/master/utils/loss.py
class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label=-1, thresh=0.7, min_kept=100000, use_weight=True):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            print("w/ class balance")
            weight = torch.FloatTensor(
                [
                    0.8373,
                    0.918,
                    0.866,
                    1.0345,
                    1.0166,
                    0.9969,
                    0.9754,
                    1.0489,
                    0.8786,
                    1.0023,
                    0.9539,
                    0.9843,
                    1.1116,
                    0.9037,
                    1.0865,
                    1.0955,
                    1.0865,
                    1.1529,
                    1.0507,
                ]
            )
            self.criterion = torch.nn.CrossEntropyLoss(
                weight=weight, ignore_index=ignore_label
            )
        else:
            print("w/o class balance")
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    def forward(self, predict, target, weight=None):
        """
        Args:
            predict:(n, c, h, w)
            target:(n, h, w)
            weight (Tensor, optional): a manual rescaling weight given to each class.
                                       If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(
            predict.size(0), target.size(0)
        )
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(
            predict.size(2), target.size(1)
        )
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(
            predict.size(3), target.size(3)
        )

        n, c, h, w = predict.size()
        input_label = target.data.cpu().numpy().ravel().astype(np.int32)
        x = np.rollaxis(predict.data.cpu().numpy(), 1).reshape((c, -1))
        input_prob = np.exp(x - x.max(axis=0).reshape((1, -1)))
        input_prob /= input_prob.sum(axis=0).reshape((1, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if self.min_kept >= num_valid:
            print("Labels: {}".format(num_valid))
        elif num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = pred.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if pred[threshold_index] > self.thresh:
                    threshold = pred[threshold_index]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        valid_flag_new = input_label != self.ignore_label
        # print(np.sum(valid_flag_new))
        target = Variable(
            torch.from_numpy(input_label.reshape(target.size())).long().cuda()
        )

        return self.criterion(predict, target)


class OHEMSegmentationLosses(OhemCrossEntropy2d):
    """2D Cross Entropy Loss with Auxilary Loss"""

    def __init__(
        self,
        se_loss=False,
        se_weight=0.2,
        nclass=-1,
        aux=False,
        aux_weight=0.4,
        weight=None,
        ignore_index=-1,
    ):
        super(OHEMSegmentationLosses, self).__init__(ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def forward(self, *inputs):
        if not self.se_loss and not self.aux:
            return super(OHEMSegmentationLosses, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(OHEMSegmentationLosses, self).forward(pred1, target)
            loss2 = super(OHEMSegmentationLosses, self).forward(pred2, target)
            return loss1 + self.aux_weight * loss2
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(
                target, nclass=self.nclass
            ).type_as(pred)
            loss1 = super(OHEMSegmentationLosses, self).forward(pred, target)
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.se_weight * loss2
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(
                target, nclass=self.nclass
            ).type_as(pred1)
            loss1 = super(OHEMSegmentationLosses, self).forward(pred1, target)
            loss2 = super(OHEMSegmentationLosses, self).forward(pred2, target)
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.aux_weight * loss2 + self.se_weight * loss3

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(
                target[i].cpu().data.float(), bins=nclass, min=0, max=nclass - 1
            )
            vect = hist > 0
            tvect[i] = vect
        return tvect
