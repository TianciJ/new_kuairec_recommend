"""gSASRec 第三路召回：因果序列编码 + 多负样本 + gBCE。

参考论文 https://arxiv.org/abs/2308.07192 和作者 asash/gSASRec-pytorch。
项目适配实现，不声称复现论文指标。只训练 cutoff 之前的正反馈序列，
保留重复正反馈事件；全部观看用于已看过滤。无 JSON/哈希依赖。
"""
from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    max_length: int = 100
    embedding_dim: int = 64
    blocks: int = 2
    heads: int = 1
    dropout: float = 0.2
    negatives: int = 256
    calibration: float = 0.75
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 20
    patience: int = 4
    validation_days: int = 2
    positive_threshold: float = 1.0
    cutoff: str = "2020-08-29"
    seed: int = 42

    def __post_init__(self):
        for key in ("max_length", "embedding_dim", "blocks", "heads", "negatives",
                    "batch_size", "epochs", "patience", "validation_days"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} 必须大于 0")
        if self.embedding_dim % self.heads:
            raise ValueError("embedding_dim 必须能被 heads 整除")
        if not 0 <= self.dropout < 1 or not 0 <= self.calibration <= 1:
            raise ValueError("dropout 应在 [0,1)，calibration 应在 [0,1]")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate 必须为有限正数")
        if not math.isfinite(self.positive_threshold) or self.positive_threshold <= 0:
            raise ValueError("positive_threshold 必须为有限正数")


def local_time(value):
    value = pd.Timestamp(value)
    return value.tz_localize("Asia/Shanghai") if value.tzinfo is None else value.tz_convert("Asia/Shanghai")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 固定种子提升可复现性，但不承诺跨设备/版本逐位一致。
    torch.backends.cudnn.benchmark = False


def get_device(value):
    value = ("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else value
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；请使用已安装 GPU PyTorch 的 recommend 环境")
    return device


def gbce_loss(positive_logits, negative_logits, beta):
    """论文式 (8) 的稳定等价形式；对负样本求和，再除以 k+1。

    log(sigmoid(s)^beta) = beta * log(sigmoid(s))。
    不需要显式 sigmoid/pow/log，也不需要 float64 的概率裁剪。
    """
    return (beta * F.softplus(-positive_logits.float())
            + F.softplus(negative_logits.float()).sum(dim=-1)).mean() / (negative_logits.shape[-1] + 1)


def sample_negatives(targets, num_items, count, generator=None):
    """在 1..N 中排除当前目标，有放回均匀采样；不使用测试历史。

    不排除用户其他历史物品，以保持 alpha=k/(N-1) 的采样空间定义。
    采样负例不是用户明确的负反馈。
    """
    if num_items < 2:
        raise ValueError("至少需要两个训练正反馈视频")
    sampled = torch.randint(1, num_items, (*targets.shape, count),
                            device=targets.device, generator=generator)
    return sampled + (sampled >= targets.unsqueeze(-1)).long()


class AttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.embedding_dim
        self.heads = config.heads
        self.dropout = config.dropout
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.feedforward = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(config.dropout),
                                         nn.Linear(d, d), nn.Dropout(config.dropout))

    def forward(self, x, valid):
        batch, length, dim = x.shape
        query = self.norm1(x)
        def split(tensor):
            return tensor.view(batch, length, self.heads, dim // self.heads).transpose(1, 2)
        # 布尔 mask 的 True 表示允许注意。填充 query 只开放对角线，避免全遮挡 NaN。
        causal = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
        allowed = causal[None, :, :] & valid[:, None, :]
        allowed = allowed | ((~valid[:, :, None]) & torch.eye(length, dtype=torch.bool, device=x.device))
        attended = F.scaled_dot_product_attention(
            split(self.q(query)), split(self.k(x)), split(self.v(x)),
            attn_mask=allowed[:, None, :, :], dropout_p=self.dropout if self.training else 0.0,
        ).transpose(1, 2).reshape(batch, length, dim)
        x = self.norm2(query + attended)
        return (x + self.feedforward(x)) * valid.unsqueeze(-1)


class SequenceModel(nn.Module):
    def __init__(self, num_items, config):
        super().__init__()
        self.items = nn.Embedding(num_items + 1, config.embedding_dim, padding_idx=0)
        # 按作者 gSASRec 默认设置使用独立输出 embedding。
        self.output_items = nn.Embedding(num_items + 1, config.embedding_dim, padding_idx=0)
        self.positions = nn.Embedding(config.max_length, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([AttentionBlock(config) for _ in range(config.blocks)])
        self.norm = nn.LayerNorm(config.embedding_dim)
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()

    def forward(self, sequences):
        valid = sequences != 0
        position = torch.arange(sequences.shape[1], device=sequences.device)
        x = self.dropout(self.items(sequences) + self.positions(position)) * valid.unsqueeze(-1)
        for layer in self.layers:
            x = layer(x, valid)
        return self.norm(x) * valid.unsqueeze(-1)


def read_interactions(path, cutoff):
    columns = ["user_id", "video_id", "timestamp", "play_duration", "video_duration"]
    parts = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=500_000):
        if chunk[columns].isna().any().any() or not np.isfinite(chunk[columns].to_numpy()).all():
            raise ValueError("交互表包含缺失/非有限数值")
        ids = chunk[["user_id", "video_id"]].to_numpy()
        if (ids < 0).any() or (ids != np.floor(ids)).any():
            raise ValueError("用户/视频 ID 必须是非负整数")
        if (chunk.video_duration <= 0).any() or (chunk.play_duration < 0).any():
            raise ValueError("视频时长必须为正，播放时长必须非负")
        chunk = chunk.loc[chunk.timestamp < local_time(cutoff).timestamp()].copy()
        if not chunk.empty:
            chunk[["user_id", "video_id"]] = chunk[["user_id", "video_id"]].astype("int64")
            parts.append(chunk)
    if not parts:
        raise ValueError("截止前没有训练交互")
    return pd.concat(parts, ignore_index=True)


class PreparedData:
    def __init__(self, frame, config):
        positives = frame.loc[frame.play_duration >= config.positive_threshold * frame.video_duration]
        # 只删除完全重复事件；重复观看不同于重复数据，不按用户-视频去重。
        positives = positives.drop_duplicates().sort_values(["user_id", "timestamp"], kind="stable")
        self.video_ids = np.sort(positives.video_id.unique()).astype(np.int64)
        if len(self.video_ids) < 2:
            raise ValueError("至少需要两个正反馈视频才能训练")
        # 原始 video_id=0 是真实视频；模型内部使用紧凑 1..N，0 专用于 padding。
        self.sequences = {}
        self.train_windows = []
        for user, group in positives.groupby("user_id", sort=True):
            sequence = np.searchsorted(self.video_ids, group.video_id.to_numpy()) + 1
            window = np.zeros(config.max_length + 1, dtype=np.int64)
            tail = sequence[-len(window):]
            window[-len(tail):] = tail
            self.sequences[int(user)] = window[1:].copy()
            if len(sequence) >= 2:
                self.train_windows.append(window)
        if not self.train_windows:
            raise ValueError("没有至少两次正反馈的用户，无法构造 next-item 训练目标")
        self.train_windows = np.stack(self.train_windows)
        user_ids = frame.user_id.to_numpy(dtype=np.int64)
        item_ids = frame.video_id.to_numpy(dtype=np.int64)
        self.seen = sparse.csr_matrix((np.ones(len(frame), dtype=bool), (user_ids, item_ids)),
                                     shape=(int(user_ids.max()) + 1, int(item_ids.max()) + 1))
        self.seen.sort_indices()


class GSASRecRecall:
    def __init__(self, config=None, device="auto"):
        self.config = config or Config()
        self.device = get_device(device)
        self.model = None
        self.training_info = {}

    def _initialize(self, prepared):
        self.video_ids = prepared.video_ids
        self.sequences = prepared.sequences
        self.seen = prepared.seen
        seed_all(self.config.seed)
        self.model = SequenceModel(len(self.video_ids), self.config).to(self.device)

    def _train_epoch(self, windows, optimizer, rng):
        self.model.train()
        cfg = self.config
        total, positions = 0.0, 0
        beta = 1.0 - cfg.calibration + cfg.calibration * cfg.negatives / (len(self.video_ids) - 1)
        for start in range(0, len(windows), cfg.batch_size):
            window = torch.as_tensor(windows[start:start + cfg.batch_size], device=self.device)
            inputs, targets = window[:, :-1], window[:, 1:]
            mask = (inputs != 0) & (targets != 0)
            hidden = self.model(inputs)[mask]
            targets = targets[mask]
            negatives = sample_negatives(targets, len(self.video_ids), cfg.negatives, rng)
            positive_logits = (hidden * self.model.output_items(targets)).sum(-1)
            negative_logits = torch.einsum("bd,bkd->bk", hidden, self.model.output_items(negatives))
            loss = gbce_loss(positive_logits, negative_logits, beta)
            if not torch.isfinite(loss):
                raise RuntimeError("训练 loss 非有限值，停止以避免保存损坏模型")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            count = int(mask.sum())
            total += loss.item() * count
            positions += count
        return total / positions

    def fit(self, train_path=ROOT / "data/processed/train.csv"):
        """内部时间验证选轮数，然后从头在完整 train 上训练相同轮数。

        不读取 test，不改变磁盘上的 train/test，不将验证期动作注入验证历史。
        """
        cfg = self.config
        frame = read_interactions(train_path, cfg.cutoff)
        validation_cutoff = local_time(cfg.cutoff) - pd.Timedelta(days=cfg.validation_days)
        fitting = frame.loc[frame.timestamp < validation_cutoff.timestamp()]
        validation = frame.loc[frame.timestamp >= validation_cutoff.timestamp()]
        if fitting.empty or validation.empty:
            raise ValueError("内部时间验证两侧都必须有数据；请调整 validation_days/cutoff")
        targets = self._targets(validation)
        if not targets:
            raise ValueError("内部验证期没有正反馈")
        prepared = PreparedData(fitting, cfg)
        self._initialize(prepared)
        device_name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"
        print(f"device={device_name}; train_users={len(prepared.train_windows)}; "
              f"items={len(self.video_ids)}; validation_users={len(targets)}; "
              f"validation_cutoff={validation_cutoff.isoformat()}", flush=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        random_order = np.random.default_rng(cfg.seed)
        sampler = torch.Generator(device=self.device).manual_seed(cfg.seed)
        best_score, best_epoch, stale, history = -1.0, 0, 0, []
        for epoch in range(1, cfg.epochs + 1):
            start = time.perf_counter()
            order = random_order.permutation(len(prepared.train_windows))
            loss = self._train_epoch(prepared.train_windows[order], optimizer, sampler)
            metrics = self.evaluate_targets(targets, top_k=200)
            score = metrics["macro_recall"]
            history.append({"epoch": epoch, "loss": loss, "validation_recall_200": score})
            print(f"select epoch={epoch}/{cfg.epochs} loss={loss:.6f} "
                  f"val_recall@200={score:.6%} seconds={time.perf_counter()-start:.1f}", flush=True)
            if score > best_score + 1e-10:
                best_score, best_epoch, stale = score, epoch, 0
            else:
                stale += 1
                if stale >= cfg.patience:
                    break
        # 固定已选轮数后，完整 train 重训；不用 test 决定停止时机。
        del optimizer, prepared, fitting, validation
        prepared = PreparedData(frame, cfg)
        self._initialize(prepared)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        random_order = np.random.default_rng(cfg.seed)
        sampler = torch.Generator(device=self.device).manual_seed(cfg.seed)
        final_losses = []
        print(f"refit epochs={best_epoch}; train_users={len(prepared.train_windows)}; "
              f"items={len(self.video_ids)}; train_rows={len(frame)}", flush=True)
        for epoch in range(1, best_epoch + 1):
            start = time.perf_counter()
            order = random_order.permutation(len(prepared.train_windows))
            loss = self._train_epoch(prepared.train_windows[order], optimizer, sampler)
            final_losses.append(loss)
            print(f"refit epoch={epoch}/{best_epoch} loss={loss:.6f} "
                  f"seconds={time.perf_counter()-start:.1f}", flush=True)
        self.model.eval()
        self.training_info = {"selected_epochs": best_epoch, "best_validation_recall_200": best_score,
                              "validation_cutoff": validation_cutoff.isoformat(), "selection_history": history,
                              "refit_losses": final_losses, "train_rows": len(frame), "device": device_name}
        if self.device.type == "cuda":
            self.training_info["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated(self.device) / 1024**2
        return self

    @torch.inference_mode()
    def recommend_batch(self, user_ids, top_k=200, *, candidate_ids=None, exclude_ids=()):
        if self.model is None:
            raise RuntimeError("请先 fit 或 load")
        users = [int(u) for u in user_ids]
        if top_k < 0 or any(u < 0 for u in users):
            raise ValueError("user_id 和 top_k 必须非负")
        result = {u: [] for u in users}
        active = [u for u in dict.fromkeys(users) if u in self.sequences]
        if not active or not top_k:
            return result
        allowed = None if candidate_ids is None else set(candidate_ids)
        excluded = set(exclude_ids)
        self.model.eval()
        for start in range(0, len(active), self.config.batch_size):
            batch_users = active[start:start + self.config.batch_size]
            sequences = torch.as_tensor(np.stack([self.sequences[u] for u in batch_users]), device=self.device)
            hidden = self.model(sequences)[:, -1, :]
            scores = (hidden @ self.model.output_items.weight[1:].T).float().cpu().numpy()
            if not np.isfinite(scores).all():
                raise RuntimeError("召回分数包含非有限值")
            for user, row in zip(batch_users, scores):
                seen = set(self.seen.getrow(user).indices.tolist()) if user < self.seen.shape[0] else set()
                blocked = seen | excluded
                valid = np.fromiter((v not in blocked and (allowed is None or v in allowed)
                                     for v in self.video_ids), dtype=bool, count=len(self.video_ids))
                ids, values = self.video_ids[valid], row[valid]
                # logits 可以为负；不按 >0 过滤，不把它们当成观看概率。
                order = np.lexsort((ids, -values))[:top_k]
                result[user] = [(int(ids[i]), float(values[i])) for i in order]
        return result

    def recommend(self, user_id, top_k=200, *, candidate_ids: Iterable[int] | None = None,
                  exclude_ids: Iterable[int] = ()):
        return self.recommend_batch([user_id], top_k, candidate_ids=candidate_ids,
                                    exclude_ids=exclude_ids)[int(user_id)]

    def _targets(self, frame):
        positive = frame.loc[frame.play_duration >= self.config.positive_threshold * frame.video_duration]
        return {int(u): set(g.video_id.tolist()) for u, g in positive.groupby("user_id")}

    def evaluate_targets(self, targets, top_k=200):
        targets = {u: set(v) for u, v in targets.items() if len(v)}
        if not targets:
            raise ValueError("没有可评估正例")
        candidates = self.recommend_batch(targets, top_k)
        hits, total, recall_sum = 0, 0, 0.0
        for user, target in targets.items():
            found = len({v for v, _ in candidates[user]} & target)
            hits += found
            total += len(target)
            recall_sum += found / len(target)
        return {"users": len(targets), "positive_pairs": total, "hits": hits,
                "macro_recall": recall_sum / len(targets), "micro_recall": hits / total,
                "mean_candidates": sum(map(len, candidates.values())) / len(targets)}

    def evaluate(self, test_path, top_k=200):
        # 仅显式调用时读取 test；固定训练截止时刻的历史，不滚动添加测试行为。
        frame = pd.read_csv(test_path, usecols=["user_id", "video_id", "timestamp", "play_duration", "video_duration"])
        if frame.empty or frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
            raise ValueError("测试数据为空或包含缺失/非有限值")
        if frame.timestamp.min() < local_time(self.config.cutoff).timestamp():
            raise ValueError("测试集包含训练截止时间之前的交互")
        if frame.video_duration.min() <= 0 or frame.play_duration.min() < 0:
            raise ValueError("测试集包含非法时长")
        return self.evaluate_targets(self._targets(frame), top_k)

    def save(self, path):
        if self.model is None:
            raise RuntimeError("请先 fit 或 load")
        users = np.array(sorted(self.sequences), dtype=np.int64)
        payload = {"kind": "gsasrec_recall_v1", "config": asdict(self.config),
                   "weights": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                   "video_ids": torch.from_numpy(self.video_ids.copy()), "users": torch.from_numpy(users),
                   "sequences": torch.from_numpy(np.stack([self.sequences[u] for u in users])),
                   "seen_indices": torch.from_numpy(self.seen.indices.astype(np.int64)),
                   "seen_indptr": torch.from_numpy(self.seen.indptr.astype(np.int64)),
                   "seen_shape": list(self.seen.shape), "training_info": self.training_info}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            torch.save(payload, stream)

    @classmethod
    def load(cls, path, device="auto"):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["kind"] != "gsasrec_recall_v1":
            raise ValueError("不是 gSASRec 召回文件")
        instance = cls(Config(**payload["config"]), device)
        instance.video_ids = payload["video_ids"].numpy()
        instance.sequences = dict(zip(payload["users"].tolist(), payload["sequences"].numpy()))
        indices = payload["seen_indices"].numpy()
        instance.seen = sparse.csr_matrix((np.ones(len(indices), dtype=bool), indices,
                                          payload["seen_indptr"].numpy()), shape=tuple(payload["seen_shape"]))
        instance.model = SequenceModel(len(instance.video_ids), instance.config).to(instance.device)
        instance.model.load_state_dict(payload["weights"])
        instance.model.eval()
        instance.training_info = payload["training_info"]
        return instance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data/processed/train.csv")
    parser.add_argument("--device", default="cuda", help="cuda / cpu / auto；默认 CUDA，不静默退回 CPU")
    parser.add_argument("--save-index", type=Path)
    parser.add_argument("--load-index", type=Path)
    parser.add_argument("--test", type=Path, help="可选：训练结束/加载后做一次测试，不用于选模型")
    parser.add_argument("--user-id", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--output", type=Path)
    defaults = asdict(Config())
    for name, default in defaults.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    args = parser.parse_args()
    if args.save_index and args.save_index.exists():
        parser.error("save-index 已存在，请选择新文件名")
    if args.load_index:
        model = GSASRecRecall.load(args.load_index, args.device)
    else:
        model = GSASRecRecall(Config(**{k: getattr(args, k) for k in defaults}), args.device).fit(args.train)
    if args.save_index:
        model.save(args.save_index)
        print(f"saved={args.save_index}", flush=True)
    if args.test:
        print("TEST", model.evaluate(args.test, args.top_k), flush=True)
    results = pd.DataFrame(model.recommend(args.user_id, args.top_k), columns=["video_id", "score"])
    results.insert(0, "user_id", args.user_id)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
