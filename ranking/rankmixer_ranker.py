"""Dense RankMixer 的 KuaiRec 适配：语义 token、多头混合、独立 PFFN。

依据 https://arxiv.org/html/2507.15551v3 §3.1–3.3；不包含 Sparse-MoE。
"""
from dataclasses import dataclass
import math

import torch
from torch import nn

from ranking.din_ranker import DINConfig, DINInterestEncoder
from ranking.neural_ranker import TorchFineRanker


@dataclass
class RankMixerConfig(DINConfig):
    token_dim: int = 64
    layers: int = 2
    expansion: int = 2

    def __post_init__(self):
        super().__post_init__()
        if self.token_dim <= 0 or self.token_dim % 8 or self.layers <= 0 or self.expansion <= 0:
            raise ValueError('token_dim 必须为 8 的正整数倍，layers/expansion 必须为正')


class TokenMixing(nn.Module):
    """论文式(3)(4)：[B,T,H,D/H] 交换 T/H 轴；H=T，无可训练参数。"""
    def forward(self, tokens):
        batch, count, width = tokens.shape
        if width % count:
            raise ValueError('token 宽度必须能被 token 数量整除')
        return tokens.reshape(batch, count, count, width // count).transpose(1, 2).reshape(batch, count, width)


class PerTokenFFN(nn.Module):
    """每个 token 的权重独立，用 batched GEMM 并行计算。"""
    def __init__(self, tokens, width, expansion):
        super().__init__()
        hidden = width * expansion
        self.w1 = nn.Parameter(torch.empty(tokens, width, hidden))
        self.b1 = nn.Parameter(torch.zeros(tokens, 1, hidden))
        self.w2 = nn.Parameter(torch.empty(tokens, hidden, width))
        self.b2 = nn.Parameter(torch.zeros(tokens, 1, width))
        nn.init.uniform_(self.w1, -1 / math.sqrt(width), 1 / math.sqrt(width))
        nn.init.uniform_(self.w2, -1 / math.sqrt(hidden), 1 / math.sqrt(hidden))

    def forward(self, tokens):
        hidden = nn.functional.gelu(torch.bmm(tokens.transpose(0, 1), self.w1) + self.b1)
        return (torch.bmm(hidden, self.w2) + self.b2).transpose(0, 1)


class RankMixerBlock(nn.Module):
    def __init__(self, tokens, width, expansion, dropout):
        super().__init__()
        self.mixing = TokenMixing()
        self.ffn = PerTokenFFN(tokens, width, expansion)
        self.mix_norm = nn.LayerNorm(width)
        self.ffn_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        # 论文式(1)的 post-LN 残差结构，额外 dropout 与现有训练正则一致。
        tokens = self.mix_norm(tokens + self.mixing(tokens))
        return self.ffn_norm(tokens + self.dropout(self.ffn(tokens)))


def semantic_groups(features):
    """五组数值字段各含数值与对应缺失标记，全部特征恰好分配一次。"""
    groups = {name: [] for name in ('recall', 'user', 'item', 'category', 'cross')}
    for i, name in enumerate(features):
        if name.startswith(('hot_', 'itemcf_', 'gsasrec_', 'from_', 'recall_')):
            group = 'recall'
        elif name.startswith('u_'):
            group = 'user'
        elif name.startswith('i_category_'):
            group = 'category'
        elif name.startswith('i_'):
            group = 'item'
        elif name.startswith(('uc_', 'ui_', 'ua_')):
            group = 'cross'
        else:
            raise ValueError(f'未定义 RankMixer 特征分组：{name}')
        groups[group].append(i)
    if any(not indices for indices in groups.values()):
        raise ValueError('RankMixer 的五个数值语义分组均需要至少一个字段')
    return {name: indices + [i + len(features) for i in indices] for name, indices in groups.items()}


class RankMixerNetwork(nn.Module):
    def __init__(self, num_users, num_items, features, config):
        super().__init__()
        # 相同输入和序列编码；替换 DIN 的最终 MLP 交互部分。
        self.behavior = DINInterestEncoder(num_users, num_items, config)
        width = config.token_dim
        self.id_projections = nn.ModuleList([nn.Linear(config.embedding_dim, width) for _ in range(3)])
        self.numeric_projections = nn.ModuleDict()
        for name, indices in semantic_groups(features).items():
            self.register_buffer(f'{name}_indices', torch.tensor(indices, dtype=torch.long))
            self.numeric_projections[name] = nn.Linear(len(indices), width)
        self.blocks = nn.Sequential(*[RankMixerBlock(8, width, config.expansion, config.dropout)
                                      for _ in range(config.layers)])
        self.output = nn.Linear(width, 1)

    def tokenize(self, users, items, histories, dense):
        user = self.behavior.user_embedding(users)
        target = self.behavior.item_embedding(items)
        interest = self.behavior.interest(target, histories)
        tokens = [projection(value) for projection, value in zip(self.id_projections, (user, target, interest))]
        tokens.extend(self.numeric_projections[name](dense.index_select(1, getattr(self, f'{name}_indices')))
                      for name in self.numeric_projections)
        return torch.stack(tokens, dim=1)

    def forward(self, users, items, histories, dense):
        tokens = self.blocks(self.tokenize(users, items, histories, dense))
        return self.output(tokens.mean(dim=1)).squeeze(-1)


class RankMixerFineRanker(TorchFineRanker):
    model_name = 'rankmixer'
    config_type = RankMixerConfig

    def build_network(self):
        return RankMixerNetwork(len(self.encoder.users), len(self.encoder.items), self.encoder.features, self.config)
