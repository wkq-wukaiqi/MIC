from copy import deepcopy

import torch
from timm.models.layers import DropPath
from torch import nn
from torch.nn.modules.dropout import _DropoutNd
import torch.nn.functional as F


class EMATeacher(nn.Module):

    def __init__(self, model, alpha, pseudo_label_weight):
        super(EMATeacher, self).__init__()
        self.ema_model = deepcopy(model)
        self.alpha = alpha
        self.pseudo_label_weight = pseudo_label_weight
        if self.pseudo_label_weight == 'None':
            self.pseudo_label_weight = None

    def _init_ema_weights(self, model):
        for param in self.ema_model.parameters():
            param.detach_()
        mp = list(model.parameters())
        mcp = list(self.ema_model.parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, model, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.ema_model.parameters(),
                                    model.parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def update_weights(self, model, iter):
        # Init/update ema model
        if iter == 0:
            self._init_ema_weights(model)
        if iter > 0:
            self._update_ema(model, iter)

    @torch.no_grad()
    def forward(self, target_img):
        # Generate pseudo-label
        for m in self.ema_model.modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        logits, _ = self.ema_model(target_img)

        ema_softmax = torch.softmax(logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)

        if self.pseudo_label_weight is None:
            pseudo_weight = torch.tensor(1., device=logits.device)
        elif self.pseudo_label_weight == 'prob':
            pseudo_weight = pseudo_prob
        else:
            raise NotImplementedError(self.pseudo_label_weight)

        return pseudo_label, pseudo_weight, ema_softmax
    
class EMATeacherPrototype(nn.Module):

    def __init__(self, model, alpha, pseudo_label_weight, threshold):
        super(EMATeacherPrototype, self).__init__()
        self.ema_model = deepcopy(model)
        # 按照ProDA论文，softmax软标签是固定的
        self.ema_model_fix = deepcopy(model)
        self.alpha = alpha
        self.pseudo_label_weight = pseudo_label_weight
        if self.pseudo_label_weight == 'None':
            self.pseudo_label_weight = None

        self.threshold = threshold

        self.num_classes = self.ema_model.num_classes
        self.features_dim = self.ema_model._features_dim

        self.prototypes = nn.Parameter(
            torch.zeros((self.num_classes, self.features_dim)),
            requires_grad=False
        )
        self.momentum = 0.9

        self.init_mode = False
        self.len_dataset = 0
        self.confidence_count = [0 for _ in range(12)]

    def _init_ema_weights(self, model):
        for param in self.ema_model.parameters():
            param.detach_()
        mp = list(model.parameters())
        mcp = list(self.ema_model.parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, model, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.ema_model.parameters(),
                                    model.parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def update_weights(self, model, iter):
        # Init/update ema model
        if iter == 0:
            self._init_ema_weights(model)
        if iter > 0:
            self._update_ema(model, iter)

    @torch.no_grad()
    def forward(self, target_img):
        # Generate pseudo-label
        for m in self.ema_model.modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

        for m in self.ema_model_fix.modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

        _, features = self.ema_model(target_img)
        logits, _ = self.ema_model_fix(target_img)
        # logits, features = self.ema_model(target_img)
        ema_softmax = torch.softmax(logits.detach(), dim=1)

        # 计算w
        distances = torch.cdist(features, self.prototypes, p=2)
        # 文中公式没写，开源代码里写了还要减去最小距离
        nearest_distance, _ = distances.min(dim=1, keepdim=True)
        distances = distances - nearest_distance
        w = F.softmax(-distances, dim=1)

        # ProDA的不同点是在这里给ema_softmax乘上了一个权重w，然后再做argmax产生硬伪标签
        ema_softmax = ema_softmax * w
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)

        return pseudo_prob, pseudo_label, ema_softmax, features

    def init_begin(self):
        self.init_mode = True

    def init_end(self):
        self.init_mode = False
        for i in range(self.num_classes):
            self.prototypes[i] = self.prototypes[i] / self.confidence_count[i]
        self.confidence_count = [0 for _ in range(self.num_classes)]

    @torch.no_grad()
    def init_prototypes(self, target_img):
        # Generate pseudo-label
        for m in self.ema_model.modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        logits, features = self.ema_model(target_img)

        ema_softmax = torch.softmax(logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)

        self.update_prototypes(features, pseudo_prob, pseudo_label)

        high_confidence_idx = torch.nonzero(pseudo_prob > self.threshold).squeeze()

        return len(high_confidence_idx)

    @torch.no_grad()
    def update_prototypes(self, features, pseudo_prob, pseudo_label):
        if self.init_mode:
            # 去掉置信度小于阈值的
            high_confidence_idx = torch.nonzero(pseudo_prob > self.threshold).squeeze()
        else:
            # 更新时全用
            high_confidence_idx = torch.nonzero(pseudo_prob > 0).squeeze()
        # high_confidence_idx = torch.nonzero(pseudo_prob > self.threshold).squeeze()
        
        if high_confidence_idx.size() != torch.Size([0]):
            # 更新原型
            pseudo_label_usage = pseudo_label[high_confidence_idx]
            # pseudo_label_usage = pseudo_label
            if pseudo_label_usage.size() == torch.Size([]):
                pseudo_label_usage = pseudo_label_usage.unsqueeze(0)
            for i in range(self.num_classes):
                # 查找数据类型编号为i的样本在batch里的下标
                class_index = torch.nonzero(pseudo_label_usage == i).squeeze()
                if class_index.size() == torch.Size([0]):
                    # 如果没有第i类的样本，则跳过
                    continue
                if class_index.size() == torch.Size([]):
                    class_index = class_index.unsqueeze(0)
                # 取出第i类样本的feature
                prototype = features[class_index]
                # 第i类样本的feature
                if prototype.size() == torch.Size([]):
                    prototype = prototype.unsqueeze(0)
                prototype = torch.mean(prototype, dim=0)
                if self.init_mode:
                    # 初始化聚类中心，求均值
                    self.prototypes[i] = self.prototypes[i] + prototype
                    self.confidence_count[i] += len(class_index)
                else:
                    # 动量更新聚类中心
                    if torch.sum(self.prototypes[i]) == 0:
                        self.prototypes[i] = prototype
                    else:
                        self.prototypes[i] = self.prototypes[i] * self.momentum + prototype * (
                                1 - self.momentum)
