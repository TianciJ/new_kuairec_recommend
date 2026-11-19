"""复用三路召回，按时间截点缓存候选；不读取标签期或测试期行为。"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ranking.features import TRAIN, DAILY, SIGNAL_COLUMNS, local_time


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def save_candidates(path, frame, cutoff, top_k):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as f:
        np.savez_compressed(f, user_id=frame.user_id.to_numpy(dtype='int32'),
                            video_id=frame.video_id.to_numpy(dtype='int32'),
                            signals=frame[SIGNAL_COLUMNS].to_numpy(dtype='float32'),
                            cutoff=np.array(local_time(cutoff).isoformat()), top_k=np.array(top_k))
    temporary.replace(path)


def load_candidates(path, cutoff, top_k):
    with np.load(path, allow_pickle=False) as data:
        if str(data['cutoff'].item()) != local_time(cutoff).isoformat() or int(data['top_k']) != top_k:
            raise ValueError('候选缓存的时间或候选数不匹配')
        frame = pd.DataFrame(data['signals'], columns=SIGNAL_COLUMNS)
        frame.insert(0, 'video_id', data['video_id'])
        frame.insert(0, 'user_id', data['user_id'])
    return frame


def merge_routes(routes):
    """单个用户的三路列表合并；未命中分数与排名保留 NaN。"""
    values = {}
    for route_no, route in enumerate(('hot', 'itemcf', 'gsasrec')):
        seen = set()
        for rank, (video, score) in enumerate(routes[route], start=1):
            video = int(video)
            if video in seen or not np.isfinite(score):
                raise ValueError('召回包含重复 ID 或非有限分数')
            seen.add(video)
            row = values.setdefault(video, np.full(6, np.nan, dtype='float32'))
            row[route_no * 2:route_no * 2 + 2] = score, rank
    ids = np.array(sorted(values), dtype='int32')
    signals = np.stack([values[v] for v in ids]) if len(ids) else np.empty((0, 6), dtype='float32')
    return ids, signals


def hot_candidates(user_id, top_k, directory, eligible_ids):
    from recall.hot_recall import HotRecall

    hot = HotRecall.load(Path(directory) / 'hot.npz')
    ids, signals = merge_routes({'hot': hot.recommend(user_id, top_k, candidate_ids=eligible_ids),
                                 'itemcf': [], 'gsasrec': []})
    frame = pd.DataFrame(signals, columns=SIGNAL_COLUMNS)
    frame.insert(0, 'video_id', ids)
    frame.insert(0, 'user_id', user_id)
    return frame


def observed_days(train_path=TRAIN):
    days = set()
    for chunk in pd.read_csv(train_path, usecols=['date'], chunksize=500_000):
        days.update(int(v) for v in chunk.date.unique())
    return sorted(days)


def internal_validation_days(cutoff, days):
    """历史 gSASRec 用最后两个有记录的自然日选轮数，允许中间日期空档。"""
    cutoff = local_time(cutoff)
    previous = [d for d in days if d < int(cutoff.strftime('%Y%m%d'))]
    if len(previous) < 3:
        raise ValueError('至少需要三个有交互的日期构建历史召回模型')
    start = local_time(pd.to_datetime(str(previous[-2]), format='%Y%m%d'))
    return int((cutoff - start).days)


def build_candidates(cutoff, directory, eligible_ids, users, *, top_k=200, train_path=TRAIN,
                     daily_path=DAILY, gsas_epochs=100, gsas_patience=10, days=None,
                     existing_gsas=None):
    """训练必要的历史召回模型，分用户合并，复用已验证缓存。"""
    from recall.hot_recall import HotRecall
    from recall.itemcf_recall import ItemCFRecall
    from recall.gsasrec_recall import GSASRecRecall, Config
    import torch

    cutoff = local_time(cutoff)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    meta_path = directory / 'candidates_meta.json'
    meta = {'version': 1, 'cutoff': cutoff.isoformat(), 'top_k': top_k,
            'gsas_epochs': gsas_epochs, 'gsas_patience': gsas_patience,
            'eligible_ids': sorted(int(v) for v in eligible_ids), 'users': sorted(int(u) for u in users)}
    if meta_path.exists():
        saved = json.loads(meta_path.read_text(encoding='utf-8'))
        if any(saved.get(key) != value for key, value in meta.items()):
            raise ValueError(f'缓存配置发生变化，请换用新的输出目录：{directory}')
    if (directory / 'candidates.npz').exists() and meta_path.exists():
        return load_candidates(directory / 'candidates.npz', cutoff, top_k)
    write_json(meta_path, meta)

    hot_path, cf_path, gsas_path = (directory / f'{name}.{suffix}' for name, suffix in [('hot', 'npz'), ('itemcf', 'npz'), ('gsasrec', 'pt')])
    hot = HotRecall.load(hot_path) if hot_path.exists() else HotRecall(cutoff=str(cutoff)).fit(train_path, daily_path)
    if not hot_path.exists():
        hot.save(hot_path)
    cf = ItemCFRecall.load(cf_path) if cf_path.exists() else ItemCFRecall(cutoff=str(cutoff)).fit(train_path)
    if not cf_path.exists():
        cf.save(cf_path)
    if existing_gsas or gsas_path.exists():
        gsas = GSASRecRecall.load(existing_gsas or gsas_path, device='cuda')
    else:
        config = Config(cutoff=str(cutoff), epochs=gsas_epochs, patience=gsas_patience,
                        validation_days=internal_validation_days(cutoff, days or observed_days(train_path)))
        print(f'TRAIN RECALL cutoff={cutoff.isoformat()} validation_days={config.validation_days}', flush=True)
        gsas = GSASRecRecall(config, device='cuda').fit(train_path)
        gsas.save(gsas_path)
    if hot.cutoff != cutoff or cf.cutoff != cutoff or local_time(gsas.config.cutoff) != cutoff:
        raise ValueError('召回模型截止时间不一致，拒绝使用未来模型')
    if gsas.config.positive_threshold != 1 or cf.positive_threshold != 1:
        raise ValueError('召回正反馈口径必须为完播')

    user_arrays, video_arrays, signal_arrays = [], [], []
    users = sorted(set(int(u) for u in users))
    for start in range(0, len(users), 64):
        batch = users[start:start + 64]
        third = gsas.recommend_batch(batch, top_k, candidate_ids=eligible_ids)
        for user in batch:
            routes = {'hot': hot.recommend(user, top_k, candidate_ids=eligible_ids),
                      'itemcf': cf.recommend(user, top_k, candidate_ids=eligible_ids), 'gsasrec': third[user]}
            ids, signals = merge_routes(routes)
            user_arrays.append(np.full(len(ids), user, dtype='int32'))
            video_arrays.append(ids)
            signal_arrays.append(signals)
        if start % 1024 == 0:
            print(f'CANDIDATES cutoff={cutoff.date()} users={min(start + 64, len(users))}/{len(users)}', flush=True)
    frame = pd.DataFrame(np.concatenate(signal_arrays), columns=SIGNAL_COLUMNS)
    frame.insert(0, 'video_id', np.concatenate(video_arrays))
    frame.insert(0, 'user_id', np.concatenate(user_arrays))
    save_candidates(directory / 'candidates.npz', frame, cutoff, top_k)
    del hot, cf, gsas, user_arrays, video_arrays, signal_arrays
    gc.collect()
    torch.cuda.empty_cache()
    return frame
