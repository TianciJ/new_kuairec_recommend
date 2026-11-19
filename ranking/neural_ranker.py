"""神经精排共用的批量推理、模型持久化与训练步骤。"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from ranking.din_data import DINEncoder, lookup


def resolve_device(device):
    resolved = torch.device(device)
    if resolved.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用；如需 CPU，请显式指定 --device cpu')
    return resolved


class TorchFineRanker:
    """共享评分、推荐、训练和 checkpoint 契约；子类只提供网络与配置。"""
    model_name = None
    config_type = None

    def __init__(self, encoder, config=None, device='cpu'):
        self.encoder = encoder
        self.config = config or self.config_type()
        self.device = resolve_device(device)
        self.model = self.build_network().to(self.device)
        self.metadata = {}

    def predict_encoded(self, data):
        self.model.eval()
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(data['users']), self.config.batch_size):
                index = slice(start, start + self.config.batch_size)
                inputs = batch_inputs(data, index, self.device)
                outputs.append(self.model(*inputs).sigmoid().cpu().numpy())
        scores = np.concatenate(outputs) if outputs else np.empty(0, dtype='float32')
        if not np.isfinite(scores).all():
            raise ValueError('精排产生非有限分数')
        return scores

    def predict(self, candidates, features, history_users, history):
        if len(candidates) != len(features):
            raise ValueError('候选与特征行数不一致')
        if history.shape != (len(history_users), self.config.max_length):
            raise ValueError('历史矩阵尺寸与模型配置不一致')
        data = {
            'users': lookup(self.encoder.users, candidates.user_id.to_numpy()),
            'items': self.encoder.item_ids(candidates.video_id.to_numpy()),
            'dense': self.encoder.dense(features.to_numpy(), features.columns),
            'histories': self.encoder.item_ids(np.vstack([np.full((1, self.config.max_length), -1), history])),
            'history_row': lookup(history_users, candidates.user_id.to_numpy()),
        }
        return self.predict_encoded(data)

    def recommend(self, user_id, candidates, feature_store, history_users, history, top_k=20, coarse_ranker=None, coarse_k=200):
        if top_k < 0 or coarse_k <= 0:
            raise ValueError('top_k 不能为负，coarse_k 必须为正')
        frame = candidates.loc[candidates.user_id == user_id]
        if frame.empty or top_k == 0:
            return []
        if frame.video_id.duplicated().any():
            raise ValueError('候选视频重复')
        features = feature_store.transform(frame)
        if coarse_ranker is not None:
            scores = coarse_ranker.predict(features)
            order = np.lexsort((frame.video_id.to_numpy(), -scores))[:coarse_k]
            frame, features = frame.iloc[order], features.iloc[order]
        scores = self.predict(frame, features, history_users, history)
        ids = frame.video_id.to_numpy()
        order = np.lexsort((ids, -scores))[:top_k]
        return [(int(ids[i]), float(scores[i])) for i in order]

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError(f'拒绝覆盖已有 {self.model_name} checkpoint：{path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {'schema': 1, 'model_name': self.model_name, 'config': asdict(self.config), 'encoder': self.encoder.state(),
                   'model': {k: v.detach().cpu() for k, v in self.model.state_dict().items()}, 'metadata': self.metadata}
        temporary = path.with_suffix(path.suffix + '.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, device='cpu'):
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state['schema'] != 1 or state.get('model_name', 'din') != cls.model_name:
            raise ValueError('checkpoint schema 或模型类型不匹配')
        obj = cls(DINEncoder.from_state(state['encoder']), cls.config_type(**state['config']), device)
        obj.model.load_state_dict(state['model'])
        obj.model.eval()
        obj.metadata = state['metadata']
        return obj


def batch_inputs(data, index, device):
    histories = data['histories'][data['history_row'][index]]
    return tuple(torch.as_tensor(a, device=device) for a in
                 (data['users'][index], data['items'][index], histories, data['dense'][index]))


def train_epoch(ranker, data, optimizer, rng):
    ranker.model.train()
    indices = rng.permutation(len(data['labels']))
    total = 0.0
    for start in range(0, len(indices), ranker.config.batch_size):
        batch = indices[start:start + ranker.config.batch_size]
        labels = torch.as_tensor(data['labels'][batch], device=ranker.device)
        optimizer.zero_grad(set_to_none=True)
        logits = ranker.model(*batch_inputs(data, batch, ranker.device))
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError('精排训练 loss 非有限')
        loss.backward()
        nn.utils.clip_grad_norm_(ranker.model.parameters(), 5.0)
        optimizer.step()
        total += float(loss.detach()) * len(batch)
    return total / len(indices)
