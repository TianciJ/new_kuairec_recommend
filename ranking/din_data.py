"""DIN 的时间快照、稀疏 ID 映射和仅由训练集拟合的数值预处理。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ranking.features import INTERACTION_COLUMNS, local_time


def positive_histories(path, cutoff, max_length=50, chunksize=250_000):
    """最近 L 次完播，严格 timestamp < cutoff；不同时间的重复观看保留。

    返回原始用户 ID 和右填充 -1 的原始视频 ID 矩阵，原始视频 0 不是 padding。
    """
    if max_length <= 0:
        raise ValueError('max_length 必须为正')
    boundary = local_time(cutoff).timestamp()
    tail = pd.DataFrame(columns=INTERACTION_COLUMNS)
    for chunk in pd.read_csv(path, usecols=INTERACTION_COLUMNS, chunksize=chunksize):
        chunk = chunk.loc[chunk.timestamp < boundary]
        if chunk.empty:
            continue
        if (not np.isfinite(chunk.to_numpy()).all()
                or (chunk[['user_id', 'video_id']] < 0).any().any()
                or (chunk.video_duration <= 0).any() or (chunk.play_duration < 0).any()):
            raise ValueError('历史交互包含非法值')
        chunk = chunk.loc[chunk.play_duration >= chunk.video_duration]
        if chunk.empty:
            continue
        tail = chunk if tail.empty else pd.concat([tail, chunk], ignore_index=True)
        tail = (tail.drop_duplicates(INTERACTION_COLUMNS)
                .sort_values(['user_id', 'timestamp'], kind='stable')
                .groupby('user_id', sort=False).tail(max_length))
    users = np.sort(tail.user_id.unique()).astype('int64')
    history = np.full((len(users), max_length), -1, dtype='int64')
    for i, (_, group) in enumerate(tail.groupby('user_id', sort=True)):
        history[i, :len(group)] = group.video_id.to_numpy(dtype='int64')
    return users, history


def lookup(vocabulary, values, offset=1, unknown=0):
    vocabulary = np.asarray(vocabulary, dtype='int64')
    values = np.asarray(values, dtype='int64')
    if not len(vocabulary):
        return np.full(values.shape, unknown, dtype='int64')
    positions = np.searchsorted(vocabulary, values)
    found = (positions < len(vocabulary)) & (vocabulary[np.minimum(positions, len(vocabulary) - 1)] == values)
    return np.where(found, positions + offset, unknown).astype('int64')


class DINEncoder:
    """ID 词表、signed-log 标准化和缺失标记；验证/测试不能更新状态。"""

    def fit(self, snapshots):
        if not snapshots or not sum(len(s['y']) for s in snapshots):
            raise ValueError('没有 DIN 训练样本')
        # NPZ 中的列名是 numpy.str_；checkpoint 只保存原生字符串以兼容 weights_only=True。
        self.features = [str(name) for name in snapshots[0]['features']]
        if any(list(s['features']) != self.features for s in snapshots):
            raise ValueError('时间快照特征列不一致')
        self.users = np.unique(np.concatenate([s['user_id'] for s in snapshots]))
        self.items = np.unique(np.concatenate(
            [a for s in snapshots for a in (s['video_id'], s['history'][s['history'] >= 0])]))
        # 分块累积统计，避免复制整个百万行特征矩阵。
        total = np.zeros(len(self.features), dtype='float64')
        squared = total.copy()
        count = total.copy()
        for snapshot in snapshots:
            for start in range(0, len(snapshot['X']), 20_000):
                raw = snapshot['X'][start:start + 20_000]
                good = np.isfinite(raw)
                values = self.log_values(raw).astype('float64')
                total += values.sum(axis=0)
                squared += np.square(values).sum(axis=0)
                count += good.sum(axis=0)
        self.mean = (total / np.maximum(count, 1)).astype('float32')
        self.scale = np.sqrt(np.maximum(squared / np.maximum(count, 1) - self.mean.astype('float64') ** 2, 0))
        self.scale = np.where(self.scale < 1e-6, 1, self.scale).astype('float32')
        return self

    @staticmethod
    def log_values(raw):
        raw = np.where(np.isfinite(raw), raw, 0).astype('float32')
        return np.sign(raw) * np.log1p(np.abs(raw))

    def dense(self, raw, features):
        if list(features) != self.features:
            raise ValueError('DIN 特征名称或顺序与训练不一致')
        missing = ~np.isfinite(raw)
        values = (self.log_values(raw) - self.mean) / self.scale
        values[missing] = 0
        return np.concatenate([np.clip(values, -10, 10), missing.astype('float32')], axis=1)

    def item_ids(self, values):
        # 0=padding, 1=未知视频，原始视频 0 使用独立的正常词表项。
        values = np.asarray(values)
        return np.where(values < 0, 0, lookup(self.items, values, offset=2, unknown=1))

    def state(self):
        return {name: getattr(self, name).tolist() if name != 'features' else self.features
                for name in ('features', 'users', 'items', 'mean', 'scale')}

    @classmethod
    def from_state(cls, state):
        obj = cls()
        obj.features = list(state['features'])
        for name in ('users', 'items', 'mean', 'scale'):
            setattr(obj, name, np.array(state[name], dtype='int64' if name in ('users', 'items') else 'float32'))
        return obj


def encode_snapshots(encoder, snapshots):
    """历史按用户/截点存一次，样本只存行号，不展开 N×L 序列缓存。"""
    fields = {name: [] for name in ('users', 'items', 'dense', 'history_row', 'labels', 'histories')}
    offset = 0
    for snapshot in snapshots:
        history = np.vstack([np.full((1, snapshot['history'].shape[1]), -1), snapshot['history']])
        fields['histories'].append(encoder.item_ids(history))
        fields['history_row'].append(lookup(snapshot['history_users'], snapshot['user_id']) + offset)
        fields['users'].append(lookup(encoder.users, snapshot['user_id']))
        fields['items'].append(encoder.item_ids(snapshot['video_id']))
        fields['dense'].append(encoder.dense(snapshot['X'], snapshot['features']))
        fields['labels'].append(np.asarray(snapshot['y'], dtype='float32'))
        offset += len(history)
        if len(snapshot['X']) != len(snapshot['y']):
            raise ValueError('样本和标签数量不一致')
    return {key: np.concatenate(parts) for key, parts in fields.items()}
