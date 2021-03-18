import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositeLoss(nn.Module):

    def __init__(self, losses=[nn.BCEWithLogitsLoss(), nn.MSELoss()], weights=[1.0, 1.0]):
        super().__init__()
        assert len(losses) == len(weights)
        self.losses = losses
        self.weights = weights
    
    def forward(self, approxs, targets):
        total_loss = 0.0
        for i in range(len(self.losses)):
            total_loss += self.weights[i] * self.losses[i](
                approxs[i], targets[i])
        return total_loss

    def __repr__(self):
        desc = ['CompositeLoss(']
        for i in range(len(self.losses)):
            desc.append(f'\t{self.weights[i]} * {self.losses[i].__class__.__name__}, ')
        desc.append(')')
        return '\n'.join(desc)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1. - pt)**self.gamma * bce_loss
        return focal_loss.mean()


class FocalLoss2(nn.Module):
    def __init__(self, alpha=1, gamma=2, smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        if not isinstance(smoothing, torch.Tensor):
            self.smoothing = nn.Parameter(
                torch.tensor(smoothing), requires_grad=False)
        else:
            self.smoothing = nn.Parameter(
                smoothing, requires_grad=False)
        assert 0 <= self.smoothing.min() and self.smoothing.max() < 1
    
    @staticmethod
    def _smooth(targets:torch.Tensor, smoothing:torch.Tensor):
        with torch.no_grad():
            if smoothing.shape != targets.shape:
                _smoothing = smoothing.expand_as(targets)
            else:
                _smoothing = smoothing
            return targets * (1.0 - _smoothing) + 0.5 * _smoothing

    def forward(self, inputs, targets):
        targets = FocalLoss2._smooth(targets, self.smoothing)
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1. - pt)**self.gamma * bce_loss
        return focal_loss.mean()

    def __repr__(self):
        return f'FocalLoss2(smoothing={self.smoothing})'


class DummyLoss(nn.Module):

    def __init__(self):
        super(DummyLoss, self).__init__()

    def forward(self):
        return torch.tensor(.0, requires_grad=True)
