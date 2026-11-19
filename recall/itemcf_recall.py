"""ItemCF 召回：从训练期正反馈计算视频共现余弦相似度。

不训练神经网络；fit 离线构建相似度索引。默认观看比例 >= 1 为正反馈，
每用户-视频只计一次，按最近一次正反馈保留每用户最近 200 个正反馈视频。
相似度保留每个视频最相似的 100 个邻居，推荐时累加用户正反馈视频的相似度。
完整观看历史用于排除已看视频。不会读取测试集、来源 JSON 或进行哈希校验。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "data/processed/train.csv"
DEFAULT_CUTOFF = "2020-08-29"


def _recent_positives(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    """先按用户-视频去重，再按最近正反馈时间截断，避免重复播放占满窗口。"""
    frame = frame.groupby(["user_id", "video_id"], as_index=False)["timestamp"].max()
    if limit:
        frame = frame.sort_values(
            ["user_id", "timestamp", "video_id"], ascending=[True, False, True], kind="stable"
        ).groupby("user_id", sort=False).head(limit)
    return frame


def _top_indices(ids: np.ndarray, scores: np.ndarray, count: int) -> np.ndarray:
    """分数降序、视频 ID 升序，分界同分也保持确定性。"""
    selected = np.arange(len(ids))
    if len(ids) > count:
        threshold = np.partition(scores, len(scores) - count)[len(scores) - count]
        selected = selected[scores >= threshold]
    order = np.lexsort((ids[selected], -scores[selected]))
    return selected[order[:count]]


class ItemCFRecall:
    def __init__(
        self, positive_threshold: float = 1.0, max_history: int = 200,
        similarity_top_k: int = 100, cutoff: str = DEFAULT_CUTOFF,
    ):
        if not np.isfinite(positive_threshold) or positive_threshold <= 0:
            raise ValueError("positive_threshold 必须大于 0 且有限")
        if max_history < 0 or similarity_top_k <= 0:
            raise ValueError("max_history 不能为负，similarity_top_k 必须大于 0")
        self.positive_threshold = positive_threshold
        self.max_history = max_history  # 0 表示不限历史长度，真实数据上可能计算较慢。
        self.similarity_top_k = similarity_top_k
        cutoff_time = pd.Timestamp(cutoff)
        self.cutoff = cutoff_time.tz_localize("Asia/Shanghai") if cutoff_time.tzinfo is None else cutoff_time.tz_convert("Asia/Shanghai")
        self._fitted = False

    def fit(
        self, train_path: str | Path = DEFAULT_TRAIN,
        chunksize: int = 500_000, block_size: int = 256,
    ) -> "ItemCFRecall":
        if block_size <= 0:
            raise ValueError("block_size 必须大于 0")
        self._fitted = False
        cutoff = self.cutoff.timestamp()
        seen_parts, positive_parts = [], []
        columns = ["user_id", "video_id", "timestamp", "play_duration", "video_duration"]
        for chunk in pd.read_csv(
            train_path, usecols=columns,
            dtype={"user_id": "int64", "video_id": "int64", "timestamp": "float64",
                   "play_duration": "float64", "video_duration": "float64"},
            chunksize=chunksize,
        ):
            if not np.isfinite(chunk["timestamp"]).all():
                raise ValueError("训练数据包含无效 timestamp")
            chunk = chunk.loc[chunk["timestamp"] < cutoff]
            if chunk.empty:
                continue
            if (chunk[["user_id", "video_id"]] < 0).any().any():
                raise ValueError("用户和视频 ID 必须为非负整数")
            durations = chunk[["play_duration", "video_duration"]].to_numpy()
            if not np.isfinite(durations).all() or (chunk["video_duration"] <= 0).any() or (chunk["play_duration"] < 0).any():
                raise ValueError("观看时长必须非负、视频时长必须大于 0，且不能缺失")
            seen_parts.append(chunk[["user_id", "video_id"]].to_numpy())
            # 从时长统一计算比例，避免原 watch_ratio 的少量精度差异。
            positive = chunk.loc[
                chunk["play_duration"] >= self.positive_threshold * chunk["video_duration"],
                ["user_id", "video_id", "timestamp"],
            ]
            if not positive.empty:
                # 每块先保留 top-N，最后全局再次去重取 top-N；结果不依赖 CSV 排序。
                positive_parts.append(_recent_positives(positive, self.max_history))
        if seen_parts:
            pairs = np.concatenate(seen_parts)
            shape = (int(pairs[:, 0].max()) + 1, int(pairs[:, 1].max()) + 1)
            self.seen = sparse.csr_matrix(
                (np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])), shape=shape,
            )
            del pairs, seen_parts
        else:
            shape = (0, 0)
            self.seen = sparse.csr_matrix(shape, dtype=bool)
        if positive_parts:
            positives = _recent_positives(pd.concat(positive_parts, ignore_index=True), self.max_history)
            self.positive = sparse.csr_matrix(
                (np.ones(len(positives), dtype=np.float32),
                 (positives["user_id"].to_numpy(), positives["video_id"].to_numpy())), shape=shape,
            )
            del positives, positive_parts
        else:
            self.positive = sparse.csr_matrix(shape, dtype=np.float32)

        # S(i,j) = 共同正反馈用户数 / sqrt(正反馈用户数(i) * 正反馈用户数(j))。
        # 用户历史上限同时应用于共现和分母，保证矩阵定义一致。
        counts = np.asarray(self.positive.sum(axis=0)).ravel()
        inverse_norm = np.zeros_like(counts, dtype=np.float32)
        np.divide(1.0, np.sqrt(counts), out=inverse_norm, where=counts > 0)
        normalized = self.positive.multiply(inverse_norm).tocsr()
        item_vectors = normalized.T.tocsr()
        indices, values, indptr = [], [], [0]
        for start in range(0, shape[1], block_size):
            stop = min(start + block_size, shape[1])
            block = item_vectors[start:stop] @ normalized
            for local_row in range(stop - start):
                left, right = block.indptr[local_row:local_row + 2]
                ids, scores = block.indices[left:right], block.data[left:right]
                valid = ids != start + local_row
                ids, scores = ids[valid], scores[valid]
                chosen = _top_indices(ids, scores, self.similarity_top_k)
                indices.append(ids[chosen])
                values.append(scores[chosen])
                indptr.append(indptr[-1] + len(chosen))
        self.similarity = sparse.csr_matrix(
            (np.concatenate(values) if values else np.array([], dtype=np.float32),
             np.concatenate(indices) if indices else np.array([], dtype=np.int32),
             np.asarray(indptr)), shape=(shape[1], shape[1]),
        )
        self.similarity.sort_indices()
        self._fitted = True
        return self

    def recommend(
        self, user_id: int, top_k: int = 100, *,
        candidate_ids: Iterable[int] | None = None,
        exclude_ids: Iterable[int] = (),
    ) -> list[tuple[int, float]]:
        """返回 [(video_id, 相似度累加分数)]；新用户或无正反馈时返回 []。

        上游可用热门召回补足不足的候选；不把两路分数当成同一种概率。
        """
        if not self._fitted:
            raise RuntimeError("请先 fit 或 load 索引")
        if user_id < 0 or top_k < 0:
            raise ValueError("user_id 和 top_k 不能为负数")
        if top_k == 0 or user_id >= self.positive.shape[0]:
            return []
        scores = self.positive.getrow(user_id) @ self.similarity
        ids, values = scores.indices, scores.data
        blocked = set(self.seen.getrow(user_id).indices) | set(exclude_ids)
        allowed = None if candidate_ids is None else set(candidate_ids)
        valid = np.fromiter(
            (i not in blocked and (allowed is None or i in allowed) for i in ids),
            dtype=bool, count=len(ids),
        )
        ids, values = ids[valid], values[valid]
        chosen = _top_indices(ids, values, top_k)
        return [(int(ids[i]), float(values[i])) for i in chosen]

    def save(self, path: str | Path) -> None:
        if not self._fitted:
            raise RuntimeError("请先 fit")
        payload = {
            "kind": np.array("itemcf_recall_v1"), "cutoff": np.array(self.cutoff.isoformat()),
            "positive_threshold": self.positive_threshold, "max_history": self.max_history,
            "similarity_top_k": self.similarity_top_k,
        }
        for name in ("seen", "positive", "similarity"):
            matrix = getattr(self, name)
            payload.update({f"{name}_data": matrix.data, f"{name}_indices": matrix.indices,
                            f"{name}_indptr": matrix.indptr, f"{name}_shape": np.array(matrix.shape)})
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez_compressed(stream, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "ItemCFRecall":
        with np.load(path, allow_pickle=False) as data:
            if str(data["kind"].item()) != "itemcf_recall_v1":
                raise ValueError("不是 ItemCF 召回索引")
            model = cls(float(data["positive_threshold"].item()), int(data["max_history"].item()),
                        int(data["similarity_top_k"].item()), str(data["cutoff"].item()))
            for name in ("seen", "positive", "similarity"):
                matrix = sparse.csr_matrix(
                    (data[f"{name}_data"], data[f"{name}_indices"], data[f"{name}_indptr"]),
                    shape=tuple(data[f"{name}_shape"]),
                )
                setattr(model, name, matrix)
        model._fitted = True
        return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    parser.add_argument("--positive-threshold", type=float, default=1.0)
    parser.add_argument("--max-history", type=int, default=200, help="每用户最近正反馈视频上限；0 表示不限")
    parser.add_argument("--similarity-top-k", type=int, default=100)
    parser.add_argument("--user-id", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--save-index", type=Path)
    parser.add_argument("--load-index", type=Path, help="直接加载索引，不再读取 CSV 或计算相似度")
    parser.add_argument("--output", type=Path, help="可选：保存当前用户的召回结果 CSV")
    args = parser.parse_args()
    model = ItemCFRecall.load(args.load_index) if args.load_index else ItemCFRecall(
        args.positive_threshold, args.max_history, args.similarity_top_k, args.cutoff
    ).fit(args.train)
    if args.save_index:
        model.save(args.save_index)
    result = pd.DataFrame(model.recommend(args.user_id, args.top_k), columns=["video_id", "score"])
    result.insert(0, "user_id", args.user_id)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
