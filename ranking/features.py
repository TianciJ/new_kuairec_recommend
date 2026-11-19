"""只在截止时间之前计算特征；未来交互仅供独立的标签函数读取。"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / 'data/processed/train.csv'
DAILY = ROOT / 'data/raw/KuaiRec 2.0/data/item_daily_features.csv'
CATEGORIES = ROOT / 'data/raw/KuaiRec 2.0/data/item_categories.csv'
INTERACTION_COLUMNS = ['user_id', 'video_id', 'timestamp', 'play_duration', 'video_duration']
SIGNAL_COLUMNS = [f'{route}_{kind}' for route in ('hot', 'itemcf', 'gsasrec') for kind in ('score', 'rank')]
COUNT_COLUMNS = ['play_cnt', 'show_cnt', 'complete_play_cnt', 'short_time_play_cnt',
                 'valid_play_cnt', 'like_cnt', 'follow_cnt']
CATEGORY_COUNT = 31


def local_time(value):
    value = pd.Timestamp(value)
    return value.tz_localize('Asia/Shanghai') if value.tzinfo is None else value.tz_convert('Asia/Shanghai')


def read_window(path, start, end):
    """分块加载 [start, end)，不依赖源文件的行顺序。时长保留毫秒。"""
    start, end = local_time(start).timestamp(), local_time(end).timestamp()
    if start >= end:
        raise ValueError('时间窗口必须满足 start < end')
    parts = []
    dtypes = {'user_id': 'int32', 'video_id': 'int32', 'timestamp': 'float64',
              'play_duration': 'float32', 'video_duration': 'float32'}
    for chunk in pd.read_csv(path, usecols=INTERACTION_COLUMNS, dtype=dtypes, chunksize=250_000):
        selected = chunk.loc[(chunk.timestamp >= start) & (chunk.timestamp < end)]
        if selected.empty:
            continue
        if not np.isfinite(selected.to_numpy()).all() or (selected[['user_id', 'video_id']] < 0).any().any():
            raise ValueError('交互包含非法 ID 或非有限值')
        if (selected.video_duration <= 0).any() or (selected.play_duration < 0).any():
            raise ValueError('交互包含非法时长')
        parts.append(selected)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame({k: pd.Series(dtype=v) for k, v in dtypes.items()})


def first_feedback(path, start, end):
    """窗口内首次已观测观看作为二分类标签；不为未观测候选生成负例。"""
    frame = read_window(path, start, end)
    frame = frame.sort_values('timestamp', kind='stable').drop_duplicates(['user_id', 'video_id'])
    frame['label'] = (frame.play_duration >= frame.video_duration).astype('int8')
    return frame[['user_id', 'video_id', 'label']]


def positive_targets(path, start, end):
    """评估保持现有协议：窗口内任意一次完播，按用户视频去重，包含已看和冷启动正例。"""
    frame = read_window(path, start, end)
    frame = frame.loc[frame.play_duration >= frame.video_duration, ['user_id', 'video_id']].drop_duplicates()
    return {int(u): set(g.video_id.tolist()) for u, g in frame.groupby('user_id', sort=True)}


def _ratio(a, b):
    return a / b.where(b > 0)


def recent_history(path, cutoff, limit=200):
    """逐块保留每用户最近 limit 次观看，并准确累加 1/7 天计数。"""
    cutoff = local_time(cutoff).timestamp()
    tail = None
    totals = None
    for chunk in pd.read_csv(path, usecols=INTERACTION_COLUMNS, chunksize=250_000,
                             dtype={'user_id': 'int32', 'video_id': 'int32', 'timestamp': 'float64',
                                    'play_duration': 'float32', 'video_duration': 'float32'}):
        chunk = chunk.loc[chunk.timestamp < cutoff].copy()
        if chunk.empty:
            continue
        if not np.isfinite(chunk.to_numpy()).all() or (chunk.video_duration <= 0).any() or (chunk.play_duration < 0).any():
            raise ValueError('历史交互包含非法数值')
        chunk['positive'] = (chunk.play_duration >= chunk.video_duration).astype('float32')
        values = pd.DataFrame({'user_id': chunk.user_id})
        for days in (1, 7):
            mask = chunk.timestamp >= cutoff - days * 86400
            values[f'u_watch_{days}d'] = mask.astype('float32')
            values[f'u_complete_{days}d'] = (mask & (chunk.positive == 1)).astype('float32')
        partial = values.groupby('user_id').sum()
        totals = partial if totals is None else totals.add(partial, fill_value=0)
        tail = chunk if tail is None else pd.concat([tail, chunk], ignore_index=True)
        tail = tail.sort_values(['user_id', 'timestamp'], kind='stable').groupby('user_id', sort=False).tail(limit).copy()
    if tail is None:
        raise ValueError('截点之前没有用户历史')
    return tail, totals.astype('float32')


class FeatureStore:
    """一份不可变时间快照；transform 仅查表，可以分批用于训练和推理。"""

    def __init__(self, cutoff, train_path=TRAIN, daily_path=DAILY, category_path=CATEGORIES):
        self.cutoff = local_time(cutoff)
        day = int(self.cutoff.strftime('%Y%m%d'))
        columns = ['video_id', 'date', 'author_id', 'video_duration', 'video_type', 'upload_dt'] + COUNT_COLUMNS
        daily = pd.read_csv(daily_path, usecols=columns)
        daily = daily.loc[daily.date < day]
        if daily.duplicated(['video_id', 'date']).any():
            raise ValueError('日统计表存在重复的 video_id/date')
        latest = daily.sort_values('date', kind='stable').drop_duplicates('video_id', keep='last').set_index('video_id')
        self.authors = latest.author_id
        self.items = pd.DataFrame(index=latest.index)
        self.items['i_duration_s'] = latest.video_duration / 1000.0
        upload = pd.to_datetime(latest.upload_dt, errors='coerce', format='mixed')
        self.items['i_age_days'] = (self.cutoff.tz_localize(None).normalize() - upload).dt.total_seconds() / 86400
        self.items['i_is_ad'] = latest.video_type.map({'AD': 1.0, 'NORMAL': 0.0})
        self.items['i_stats_age_days'] = (self.cutoff.tz_localize(None) - pd.to_datetime(latest.date.astype(str), format='%Y%m%d')).dt.total_seconds() / 86400
        for days in (1, 7):
            first_day = int((self.cutoff - pd.Timedelta(days=days)).strftime('%Y%m%d'))
            window = daily.loc[daily.date >= first_day]
            totals = window.groupby('video_id')[COUNT_COLUMNS].sum(min_count=1)
            self.items[f'i_stats_days_{days}d'] = window.groupby('video_id').size().reindex(self.items.index).fillna(0)
            for name in COUNT_COLUMNS:
                self.items[f'i_{name}_{days}d'] = totals[name]
            for name, numerator in [('complete_rate', 'complete_play_cnt'), ('short_rate', 'short_time_play_cnt'), ('valid_rate', 'valid_play_cnt')]:
                self.items[f'i_{name}_{days}d'] = _ratio(totals[numerator], totals.play_cnt)
        self.items['i_play_trend'] = self.items.i_play_cnt_1d / (self.items.i_play_cnt_7d / 7 + 1)
        self.items = self.items.astype('float32')
        self.eligible_ids = set(self.items.index[self.items.i_age_days.ge(0)])
        del daily, latest

        category_frame = pd.read_csv(category_path)
        if category_frame.video_id.duplicated().any():
            raise ValueError('类别表应每个视频一行')
        labels = [set(ast.literal_eval(v)) for v in category_frame.feat]
        if any(any(not isinstance(c, int) or not 0 <= c < CATEGORY_COUNT for c in cs) for cs in labels):
            raise ValueError('类别必须是 0..30 的整数列表')
        self.categories = pd.DataFrame(0.0, index=category_frame.video_id, columns=[f'i_category_{i}' for i in range(CATEGORY_COUNT)], dtype='float32')
        for video, cs in zip(category_frame.video_id, labels):
            if cs:
                self.categories.loc[video, [f'i_category_{i}' for i in cs]] = 1

        history, counts_by_window = recent_history(train_path, self.cutoff)
        history['ratio_clip'] = np.minimum(history.play_duration / history.video_duration, 1)
        history['duration_s'] = history.video_duration / 1000.0
        grouped = history.groupby('user_id')
        self.users = grouped.agg(u_watch_recent=('video_id', 'size'), u_complete_recent=('positive', 'sum'),
                                 u_ratio_clip_recent=('ratio_clip', 'mean'), u_duration_s_recent=('duration_s', 'mean'),
                                 u_last_timestamp=('timestamp', 'max'))
        self.users['u_complete_rate_recent'] = _ratio(self.users.u_complete_recent, self.users.u_watch_recent)
        self.users['u_recency_hours'] = (self.cutoff.timestamp() - self.users.pop('u_last_timestamp')) / 3600
        self.users = self.users.join(counts_by_window)
        self.users['u_complete_rate_7d'] = _ratio(self.users.u_complete_7d, self.users.u_watch_7d)
        self.users = self.users.astype('float32')
        self.prior = float(history.positive.mean())

        # 用稀疏矩阵聚合多类别，避免展开几百万条交互与类别的笛卡尔式关联。
        users = np.sort(history.user_id.unique())
        videos = np.union1d(history.video_id.unique(), self.categories.index.to_numpy())
        u = np.searchsorted(users, history.user_id)
        v = np.searchsorted(videos, history.video_id)
        shape = (len(users), len(videos))
        category_matrix = sparse.csr_matrix(self.categories.reindex(videos).fillna(0).to_numpy())
        counts = sparse.csr_matrix((np.ones(len(history), dtype='float32'), (u, v)), shape=shape)
        positives = sparse.csr_matrix((history.positive.to_numpy(), (u, v)), shape=shape)
        self.category_plays = pd.DataFrame((counts @ category_matrix).toarray(), index=users)
        self.category_positives = pd.DataFrame((positives @ category_matrix).toarray(), index=users)
        history['author_id'] = history.video_id.map(self.authors)
        self.author_stats = history.groupby(['user_id', 'author_id']).agg(
            ua_watch_recent=('video_id', 'size'), ua_complete_recent=('positive', 'sum')).astype('float32')
        self.feature_names = None

    def transform(self, candidates):
        """输入包含 user_id/video_id 和六个 score/rank 列；不接收标签或观看结果。"""
        u, v = candidates.user_id.to_numpy(), candidates.video_id.to_numpy()
        x = candidates[SIGNAL_COLUMNS].reset_index(drop=True).astype('float32')
        for route in ('hot', 'itemcf', 'gsasrec'):
            x[f'from_{route}'] = x[f'{route}_rank'].notna().astype('float32')
        x['recall_source_num'] = x[['from_hot', 'from_itemcf', 'from_gsasrec']].sum(axis=1)
        uf = self.users.reindex(u).reset_index(drop=True)
        for name in ['u_watch_1d', 'u_watch_7d', 'u_complete_1d', 'u_complete_7d', 'u_watch_recent', 'u_complete_recent']:
            uf[name] = uf[name].fillna(0)
        uf['u_no_history'] = (uf.u_watch_recent == 0).astype('float32')
        it = self.items.reindex(v).reset_index(drop=True)
        cat = self.categories.reindex(v).fillna(0).reset_index(drop=True)
        masks = cat.to_numpy()
        ncat = masks.sum(axis=1)
        plays = self.category_plays.reindex(u).fillna(0).to_numpy()
        pos = self.category_positives.reindex(u).fillna(0).to_numpy()
        preference = plays / np.maximum(uf.u_watch_recent.to_numpy()[:, None], 1)
        rates = (pos + 10 * self.prior) / (plays + 10)
        cross = pd.DataFrame({
            'uc_watch_share_mean': (preference * masks).sum(axis=1) / np.maximum(ncat, 1),
            'uc_watch_share_max': (preference * masks).max(axis=1),
            'uc_complete_rate_mean': (rates * masks).sum(axis=1) / np.maximum(ncat, 1),
            'uc_watch_count_max': (plays * masks).max(axis=1),
            'ui_duration_gap_s': abs(it.i_duration_s.to_numpy() - uf.u_duration_s_recent.to_numpy()),
            'i_category_count': ncat,
        })
        author_ids = self.authors.reindex(v).to_numpy()
        author = self.author_stats.reindex(pd.MultiIndex.from_arrays([u, author_ids], names=['user_id', 'author_id'])).fillna(0).reset_index(drop=True)
        x = pd.concat([x, uf, it, cat, cross, author], axis=1).astype('float32')
        x = x.replace([np.inf, -np.inf], np.nan)
        self.feature_names = x.columns.tolist()
        return x
