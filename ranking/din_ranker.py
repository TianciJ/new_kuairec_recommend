"""PyTorch DIN 精排：候选相关的局部激活、未归一化兴趣池化、完播二分类。"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ranking.neural_ranker import TorchFineRanker, resolve_device, batch_inputs, train_epoch


@dataclass
class DINConfig:
    max_length: int = 50
    embedding_dim: int = 32
    hidden_dim: int = 128
    dropout: float = 0.2
    batch_size: int = 512
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    epochs: int = 10
    patience: int = 3
    seed: int = 42

    def __post_init__(self):
        for name in ('max_length', 'embedding_dim', 'hidden_dim', 'batch_size', 'epochs', 'patience'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} 必须为正整数')
        if not 0 <= self.dropout < 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError('非法 dropout / learning_rate / weight_decay')


class DINInterestEncoder(nn.Module):
    """用户/视频向量与候选相关历史聚合，两个精排模型使用同一实现。"""
    def __init__(self, num_users, num_items, config):
        super().__init__()
        dim = config.embedding_dim
        self.user_embedding = nn.Embedding(num_users + 1, dim, padding_idx=0)
        self.item_embedding = nn.Embedding(num_items + 2, dim, padding_idx=0)
        for embedding in (self.user_embedding, self.item_embedding):
            nn.init.normal_(embedding.weight, std=0.02)
            with torch.no_grad():
                embedding.weight[0].zero_()
        with torch.no_grad():
            self.item_embedding.weight[1].zero_()
        self.activation = nn.Sequential(nn.Linear(4 * dim, 64), nn.Sigmoid(),
                                        nn.Linear(64, 32), nn.Sigmoid(), nn.Linear(32, 1))

    def interest(self, target, history_ids):
        history = self.item_embedding(history_ids)
        query = target.unsqueeze(1).expand_as(history)
        interactions = torch.cat([query, history, query - history, query * history], dim=-1)
        weights = self.activation(interactions).squeeze(-1)
        weights = weights.masked_fill(history_ids.eq(0), 0)
        # 无 softmax；全 padding 用户的兴趣向量严格为零。
        return (weights.unsqueeze(-1) * history).sum(dim=1)


class DINNetwork(DINInterestEncoder):
    def __init__(self, num_users, num_items, dense_dim, config):
        super().__init__(num_users, num_items, config)
        dim = config.embedding_dim
        # PReLU 基础版，不声称复现论文的 Dice 和 mini-batch aware regularization。
        self.output = nn.Sequential(nn.Linear(3 * dim + dense_dim, config.hidden_dim), nn.PReLU(),
                                    nn.Dropout(config.dropout), nn.Linear(config.hidden_dim, 64), nn.PReLU(),
                                    nn.Dropout(config.dropout), nn.Linear(64, 1))

    def forward(self, users, items, histories, dense):
        target = self.item_embedding(items)
        interest = self.interest(target, histories)
        return self.output(torch.cat([self.user_embedding(users), target, interest, dense], dim=-1)).squeeze(-1)


class DINFineRanker(TorchFineRanker):
    model_name = 'din'
    config_type = DINConfig

    def build_network(self):
        return DINNetwork(len(self.encoder.users), len(self.encoder.items),
                          2 * len(self.encoder.features), self.config)
