import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from PREPROCESS.preprocess import spilt_feature_Label


class EWC(object):
    def __init__(self, model: nn.Module, dataset):

        self.model = model
        self.dataset = dataset

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            # self._means[n] = torch.tensor(p.data)
            self._means[n] = p.clone().detach()

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            # precision_matrices[n] = torch.tensor(p.data)
            precision_matrices[n] = p.clone().detach()

        self.model.eval()

        feature, label = spilt_feature_Label(self.dataset)
        self.model.zero_grad()
        # feature = torch.tensor(feature)
        feature = feature.clone().detach()
        pre = self.model(feature)
        loss = F.mse_loss(label, pre.squeeze(-1))
        loss.backward()

        for n, p in self.model.named_parameters():
            precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss


# class EWC(nn.Module):
#     def __init__(self, model, dataset):
#         super(EWC, self).__init__()
#         self.model = model
#         self.dataset = dataset
#         self.params = {}
#         self.optimizer = None
#
#     def forward(self, x):
#         return self.model(x)
#
#     def calculate_ewc_loss(self):
#         mu = 0.01
#         sigma = 0.01
#         gamma = 0.1
#         self.optimizer.zero_grad()
#         logits = self(self.dataset.data)
#         loss = nn.BCELoss()(logits, self.dataset.labels)
#         # Calculate importance weights
#         importance_weights = torch.zeros(len(self.dataset))
#         for i in range(len(self.dataset)):
#             diff = self.model.parameters() - self.dataset.models[i].parameters()
#             dist = (diff ** 2).sum()
#             dist_max, _ = dist.max()
#             importance_weights[i] = torch.exp(-dist_max) / torch.exp(-dist_max).sum()
#             # Calculate EWC loss
#         mu_bar = mu * len(self.dataset) / (len(self.dataset) - 1) ** (gamma / 2)
#         sigma_bar = sigma * torch.sqrt(2 * len(self.dataset)) / (len(self.dataset) - 1) ** (1 - gamma / 2)
#         for name, param in self.model.named_parameters():
#             if not 'bias' in name:
#                 mean = torch.mean(importance_weights * param ** 2)
#                 var = torch.var(importance_weights * param ** 2)
#                 ewc_loss = mu_bar * mean + sigma_bar * var
#                 loss += ewc_loss * torch.sum(importance_weights * torch.exp((param ** 2 - mean) ** 2 / (2 * var)))
#         loss.backward()
#         return loss.item()
