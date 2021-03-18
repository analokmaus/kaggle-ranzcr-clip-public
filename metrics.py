import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from kuma_utils.metrics import MetricTemplate
import torch.nn as nn
import segmentation_models_pytorch as smp


class MacroAverageAUC(MetricTemplate):

    def __init__(self, maximize=True, reduce=True, verbose=False):
        super().__init__(maximize=maximize)
        self.reduce = reduce
        self.verbose = verbose

    def _test(self, target, approx):
        assert approx.shape[1] == target.shape[1]
        scores = []
        target = np.round(target)
        for i in range(target.shape[1]):
            scores.append(roc_auc_score(target[:, i], approx[:, i]))
        if self.reduce:
            return np.mean(scores)
        else:
            return scores


class MacroAveragePRAUC(MetricTemplate):

    def __init__(self, maximize=True, reduce=True, verbose=False):
        super().__init__(maximize=maximize)
        self.reduce = reduce
        self.verbose = verbose

    def _test(self, target, approx):
        assert approx.shape[1] == target.shape[1]
        scores = []
        target = np.round(target)
        for i in range(target.shape[1]):
            precision, recall, thresholds = precision_recall_curve(target[:, i], approx[:, i])
            scores.append(auc(recall, precision))
        if self.reduce:
            return np.mean(scores)
        else:
            return scores


class IoU(nn.Module):

    def __init__(self, threshold=0.5):
        super().__init__()
        self.loss = smp.utils.metrics.IoU(threshold)

    def forward(self, approx, target):
        return self.loss(approx, target).item()
