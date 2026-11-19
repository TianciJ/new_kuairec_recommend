"""热门召回：截止日前 N 天的每日播放人数之和，过滤用户全部已看视频。

无需梯度训练。跨天相加的 play_user_num 是累计人日，不是跨天去重人数。
只读取 CSV，不依赖来源 JSON，不做哈希校验。可用 NPZ 保存、加载离线索引。
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
DEFAULT_DAILY = ROOT / "data/raw/KuaiRec 2.0/data/item_daily_features.csv"
DEFAULT_CUTOFF = "2020-08-29"


class HotRecall:
    """fit 一次后，可为多个用户调用 recommend；分数不是概率。"""

    def __init__(self, window_days: int = 7, cutoff: str = DEFAULT_CUTOFF):
        if window_days <= 0:
            raise ValueError("window_days 必须大于 0")
        end = pd.Timestamp(cutoff)
        end = end.tz_localize("Asia/Shanghai") if end.tzinfo is None else end.tz_convert("Asia/Shanghai")
        if end != end.normalize():
            raise ValueError("每日统计只能按北京时间零点截断，不能使用未结束当天的统计")
        self.window_days = window_days
        self.cutoff = end
        self._fitted = False

    def fit(
        self,
        train_path: str | Path = DEFAULT_TRAIN,
        daily_path: str | Path = DEFAULT_DAILY,
        chunksize: int = 500_000,
    ) -> "HotRecall":
        self._fitted = False
        daily = pd.read_csv(
            daily_path, usecols=["video_id", "date", "play_user_num"],
            dtype={"video_id": "int64", "date": "int64", "play_user_num": "float64"},
        )
        end_day = int(self.cutoff.strftime("%Y%m%d"))
        start_day = int((self.cutoff - pd.Timedelta(days=self.window_days)).strftime("%Y%m%d"))
        daily = daily.loc[(daily["date"] >= start_day) & (daily["date"] < end_day)]
        if daily.duplicated(["video_id", "date"]).any():
            raise ValueError("热度窗口内有重复的视频-日期记录，不能重复累加")
        if (daily["video_id"] < 0).any():
            raise ValueError("video_id 必须为非负整数")
        counts = daily["play_user_num"].to_numpy()
        if not np.isfinite(counts).all() or (counts < 0).any():
            raise ValueError("play_user_num 必须为非负、非缺失数值")
        ranking = daily.groupby("video_id", as_index=False)["play_user_num"].sum()
        ranking = ranking.loc[ranking["play_user_num"] > 0].sort_values(
            ["play_user_num", "video_id"], ascending=[False, True], kind="stable"
        )
        self.video_ids = ranking["video_id"].to_numpy(dtype=np.int64)
        self.scores = ranking["play_user_num"].to_numpy(dtype=np.float64)

        # 所有观看都进入排除集合，不以是否完播区分，重复观看只占一个位置。
        parts = []
        cutoff = self.cutoff.timestamp()
        for chunk in pd.read_csv(
            train_path, usecols=["user_id", "video_id", "timestamp"],
            dtype={"user_id": "int64", "video_id": "int64", "timestamp": "float64"}, chunksize=chunksize,
        ):
            if not np.isfinite(chunk["timestamp"]).all():
                raise ValueError("训练数据包含无效 timestamp")
            chunk = chunk.loc[chunk["timestamp"] < cutoff]
            if (chunk[["user_id", "video_id"]] < 0).any().any():
                raise ValueError("用户和视频 ID 必须为非负整数")
            if not chunk.empty:
                parts.append(chunk[["user_id", "video_id"]].to_numpy())
        if parts:
            pairs = np.concatenate(parts)
            self.seen = sparse.csr_matrix(
                (np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])),
                shape=(int(pairs[:, 0].max()) + 1, int(pairs[:, 1].max()) + 1),
            )
        else:
            self.seen = sparse.csr_matrix((0, 0), dtype=bool)
        self._fitted = True
        return self

    def recommend(
        self, user_id: int, top_k: int = 100, *,
        candidate_ids: Iterable[int] | None = None,
        exclude_ids: Iterable[int] = (),
    ) -> list[tuple[int, float]]:
        """返回 [(video_id, 热度分数)]；新用户直接得到热门榜。

        candidate_ids 可由上游提供公开状态、广告等业务过滤后的候选集合。
        """
        if not self._fitted:
            raise RuntimeError("请先 fit 或 load 索引")
        if user_id < 0 or top_k < 0:
            raise ValueError("user_id 和 top_k 不能为负数")
        if top_k == 0:
            return []
        blocked = set(exclude_ids)
        if user_id < self.seen.shape[0]:
            blocked.update(self.seen.getrow(user_id).indices)
        allowed = None if candidate_ids is None else set(candidate_ids)
        result = []
        for video_id, score in zip(self.video_ids, self.scores):
            if video_id in blocked or (allowed is not None and video_id not in allowed):
                continue
            result.append((int(video_id), float(score)))
            if len(result) == top_k:
                break
        return result

    def save(self, path: str | Path) -> None:
        """保存一个 NPZ 文件；拒绝覆盖已有文件。"""
        if not self._fitted:
            raise RuntimeError("请先 fit")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez_compressed(
                stream, kind=np.array("hot_recall_v1"), window_days=self.window_days,
                cutoff=np.array(self.cutoff.isoformat()), video_ids=self.video_ids, scores=self.scores,
                seen_data=self.seen.data, seen_indices=self.seen.indices,
                seen_indptr=self.seen.indptr, seen_shape=np.array(self.seen.shape),
            )

    @classmethod
    def load(cls, path: str | Path) -> "HotRecall":
        with np.load(path, allow_pickle=False) as data:
            if str(data["kind"].item()) != "hot_recall_v1":
                raise ValueError("不是热门召回索引")
            model = cls(int(data["window_days"].item()), str(data["cutoff"].item()))
            model.video_ids, model.scores = data["video_ids"], data["scores"]
            model.seen = sparse.csr_matrix(
                (data["seen_data"], data["seen_indices"], data["seen_indptr"]),
                shape=tuple(data["seen_shape"]),
            )
        model._fitted = True
        return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--daily", type=Path, default=DEFAULT_DAILY)
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF, help="北京时间截断日期，不包含该日")
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--user-id", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--save-index", type=Path)
    parser.add_argument("--load-index", type=Path, help="直接加载索引，不再读取 CSV 或计算热度")
    parser.add_argument("--output", type=Path, help="可选：保存当前用户的召回结果 CSV")
    args = parser.parse_args()
    model = HotRecall.load(args.load_index) if args.load_index else HotRecall(
        args.window_days, args.cutoff
    ).fit(args.train, args.daily)
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
