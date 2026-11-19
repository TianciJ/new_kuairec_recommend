"""LightGBM 完播粗排：历史快照训练、GPU 学习、固定截点评估与推荐。"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Windows 的 conda OpenCL 发行库依赖同环境 Library/bin 下的 Boost DLL。
_dll_handle = None
if os.name == 'nt' and (Path(sys.prefix) / 'Library/bin').is_dir():
    _dll_handle = os.add_dll_directory(str(Path(sys.prefix) / 'Library/bin'))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from ranking.features import ROOT, TRAIN, FeatureStore, first_feedback, positive_targets, local_time
from ranking.candidates import build_candidates, hot_candidates, observed_days, write_json

SCHEMA_VERSION = 1
DEFAULT_WINDOWS = [('2020-08-03', '2020-08-06'), ('2020-08-06', '2020-08-09'), ('2020-08-09', '2020-08-11')]


def metric_summary(recommendations, targets, pools, k=200):
    """与原召回一致，按用户平均 Recall；NDCG 分母包括不可召回的真实正例。"""
    targets = {u: set(v) for u, v in targets.items() if len(v)}
    if not targets:
        raise ValueError('没有可评估正例')
    hits = total = pool_hits = 0
    recalls, ndcgs, ceilings, sizes = [], [], [], []
    discounts = 1 / np.log2(np.arange(k) + 2)
    for user, target in targets.items():
        selected = recommendations.get(user, [])[:k]
        if len(selected) != len(set(selected)):
            raise ValueError('推荐列表含重复视频')
        hit = sum(v in target for v in selected)
        available = len(set(pools.get(user, [])) & target)
        hits += hit
        total += len(target)
        pool_hits += available
        recalls.append(hit / len(target))
        ceilings.append(available / len(target))
        sizes.append(len(selected))
        dcg = sum(discounts[i] for i, v in enumerate(selected) if v in target)
        ndcgs.append(float(dcg / discounts[:min(k, len(target))].sum()))
    return {'users': len(targets), 'positive_pairs': total, 'hits': hits,
            f'macro_recall@{k}': float(np.mean(recalls)), f'micro_recall@{k}': hits / total,
            f'ndcg@{k}': float(np.mean(ndcgs)), 'mean_candidates': float(np.mean(sizes)),
            'pool_hits': pool_hits, 'pool_macro_recall': float(np.mean(ceilings)),
            'candidate_positive_retention': hits / pool_hits if pool_hits else 0.0}


def classification_metrics(y, scores):
    return {'samples': len(y), 'positive_rate': float(np.mean(y)),
            'auc': float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
            'logloss': float(log_loss(y, scores, labels=[0, 1]))}


class LightGBMCoarseRanker:
    def __init__(self, booster, feature_names):
        self.booster = booster
        self.feature_names = list(feature_names)

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        metadata = json.loads((directory / 'model_meta.json').read_text(encoding='utf-8'))
        return cls(lgb.Booster(model_file=str(directory / 'model.txt')), metadata['features'])

    def predict(self, features):
        if features.columns.tolist() != self.feature_names:
            raise ValueError('推理特征的名称或顺序与训练不一致')
        scores = self.booster.predict(features, num_threads=4)
        if not np.isfinite(scores).all():
            raise ValueError('粗排产生非有限分数')
        return scores

    def recommend(self, user_id, candidates, feature_store, top_k=200):
        if top_k < 0:
            raise ValueError('top_k 不能为负')
        frame = candidates.loc[candidates.user_id == user_id]
        if not len(frame) or not top_k:
            return []
        scores = self.predict(feature_store.transform(frame))
        ids = frame.video_id.to_numpy()
        order = np.lexsort((ids, -scores))[:top_k]
        return [(int(ids[i]), float(scores[i])) for i in order]


def prepare_window(start, end, args, days):
    directory = args.output / 'recall' / local_time(start).strftime('%Y%m%d')
    labeled_path = directory / 'labeled.npz'
    label_meta = directory / 'labeled_meta.json'
    expected = {'schema': SCHEMA_VERSION, 'start': start, 'end': end, 'route_k': args.route_k,
                'gsas_epochs': args.gsas_epochs, 'gsas_patience': args.gsas_patience}
    if labeled_path.exists() and label_meta.exists():
        saved = json.loads(label_meta.read_text(encoding='utf-8'))
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError(f'训练样本缓存过期，请使用新输出目录：{directory}')
        print(f'REUSE LABELS {start}', flush=True)
        return labeled_path
    print(f'FEATURE SNAPSHOT {start}', flush=True)
    store = FeatureStore(start)
    candidates = build_candidates(start, directory, store.eligible_ids, store.users.index,
                                  top_k=args.route_k, gsas_epochs=args.gsas_epochs,
                                  gsas_patience=args.gsas_patience, days=days)
    # 先生成候选再取未来标签；未知候选不赋 0、不用于分类训练。
    labels = first_feedback(TRAIN, start, end)
    samples = candidates.merge(labels, on=['user_id', 'video_id'], how='inner', validate='one_to_one')
    if samples.label.nunique() != 2:
        raise ValueError(f'{start} 的候选交集缺少训练样本或某个标签类别')
    parts = []
    for offset in range(0, len(samples), 20_000):
        parts.append(store.transform(samples.iloc[offset:offset + 20_000]).to_numpy())
    x = np.concatenate(parts)
    with labeled_path.with_suffix('.tmp').open('wb') as stream:
        np.savez_compressed(stream, X=x, y=samples.label.to_numpy(dtype='int8'), features=np.array(store.feature_names))
    labeled_path.with_suffix('.tmp').replace(labeled_path)
    write_json(label_meta, expected)
    print(f'LABELED start={start} candidates={len(candidates)} observed={len(samples)} positives={int(samples.label.sum())} features={x.shape[1]}', flush=True)
    del store, candidates, samples, parts, x
    gc.collect()
    return labeled_path


def load_labeled(paths):
    xs, ys, names = [], [], None
    for path in paths:
        with np.load(path, allow_pickle=False) as d:
            if names is not None and d['features'].tolist() != names:
                raise ValueError('不同时间切片的特征不一致')
            names = d['features'].tolist()
            xs.append(d['X'])
            ys.append(d['y'])
    return pd.DataFrame(np.concatenate(xs), columns=names), np.concatenate(ys)


def evaluate_snapshot(ranker, cutoff, end, label_path, args, days, existing_gsas=None):
    started = time.perf_counter()
    # 仅在固定模型后读测试标签；用户 ID 用于确定评估人群，不注入特征。
    targets = positive_targets(label_path, cutoff, end)
    store = FeatureStore(cutoff)
    candidates = build_candidates(cutoff, args.output / 'recall' / local_time(cutoff).strftime('%Y%m%d'),
                                  store.eligible_ids, store.users.index, top_k=args.route_k,
                                  gsas_epochs=args.gsas_epochs, gsas_patience=args.gsas_patience,
                                  days=days, existing_gsas=existing_gsas)
    methods = {'lightgbm': {}, 'gsasrec': {}, 'rrf': {}}
    pools = {}
    feature_seconds = predict_seconds = 0.0
    export = []
    # 候选按 user_id/video_id 有序，以用户组切块，避免拆开一个用户的排序。
    users = candidates.user_id.unique()
    bounds = np.searchsorted(candidates.user_id.to_numpy(), users)
    bounds = np.append(bounds, len(candidates))
    for offset in range(0, len(users), 64):
        frame = candidates.iloc[bounds[offset]:bounds[min(offset + 64, len(users))]].copy()
        tick = time.perf_counter()
        features = store.transform(frame)
        feature_seconds += time.perf_counter() - tick
        tick = time.perf_counter()
        frame['prediction'] = ranker.predict(features)
        predict_seconds += time.perf_counter() - tick
        for user, group in frame.groupby('user_id', sort=False):
            ids = group.video_id.to_numpy()
            scores = group.prediction.to_numpy()
            selected = np.lexsort((ids, -scores))[:args.top_k]
            methods['lightgbm'][int(user)] = ids[selected].tolist()
            third = group.loc[group.gsasrec_rank.notna()].sort_values(['gsasrec_rank', 'video_id'])
            methods['gsasrec'][int(user)] = third.video_id.head(args.top_k).tolist()
            ranks = group[['hot_rank', 'itemcf_rank', 'gsasrec_rank']].to_numpy()
            fused = np.nansum(1 / (60 + ranks), axis=1)
            methods['rrf'][int(user)] = ids[np.lexsort((ids, -fused))[:args.top_k]].tolist()
            pools[int(user)] = ids.tolist()
            if int(user) == args.user_id:
                export.extend((int(user), int(ids[i]), float(scores[i])) for i in selected)
        if offset % 1024 == 0:
            print(f'SCORE cutoff={cutoff} users={min(offset + 64, len(users))}/{len(users)}', flush=True)
    # 无历史用户未包含在这次候选缓存中，评估保留在分母、记零；单次冷启动由热门接口兜底。
    metrics = {name: metric_summary(recs, targets, pools, args.top_k) for name, recs in methods.items()}
    metrics['timing'] = {'feature_seconds': feature_seconds, 'predict_seconds': predict_seconds,
                         'candidate_rows': len(candidates), 'total_seconds': time.perf_counter() - started}
    metrics['cutoff'] = cutoff
    metrics['end'] = end
    metrics['label_policy'] = 'any_complete_in_window; all_positive_pairs_in_denominator; frozen_history'
    pd.DataFrame(export, columns=['user_id', 'video_id', 'score']).to_csv(args.output / f'recommendations_{local_time(cutoff).strftime("%Y%m%d")}.csv', index=False)
    del store, candidates
    gc.collect()
    return metrics


def train(args, days):
    metadata_path = args.output / 'model_meta.json'
    if metadata_path.exists():
        print('REUSE TRAINED MODEL', flush=True)
        return LightGBMCoarseRanker.load(args.output)
    train_paths = [prepare_window(start, end, args, days) for start, end in DEFAULT_WINDOWS]
    valid_path = prepare_window('2020-08-27', '2020-08-29', args, days)
    x, y = load_labeled(train_paths)
    vx, vy = load_labeled([valid_path])
    params = {'objective': 'binary', 'metric': 'binary_logloss', 'learning_rate': 0.05,
              'num_leaves': 31, 'max_depth': 8, 'min_data_in_leaf': 200, 'max_bin': 63,
              'feature_fraction': 0.9, 'lambda_l2': 1.0, 'num_threads': args.threads,
              'seed': 42, 'verbosity': 1, 'device_type': 'gpu',
              'gpu_platform_id': args.gpu_platform_id, 'gpu_device_id': args.gpu_device_id}
    print(f'TRAIN LIGHTGBM rows={len(y)} valid_rows={len(vy)} features={x.shape[1]} device=gpu', flush=True)
    history = {}
    started = time.perf_counter()
    booster = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=args.rounds,
                        valid_sets=[lgb.Dataset(vx, label=vy)], valid_names=['valid'],
                        callbacks=[lgb.early_stopping(args.patience, first_metric_only=True),
                                   lgb.log_evaluation(25), lgb.record_evaluation(history)])
    rounds = booster.best_iteration or args.rounds
    valid_classification = classification_metrics(vy, booster.predict(vx, num_threads=args.threads))
    features = x.columns.tolist()
    booster.save_model(str(args.output / 'validation_model.txt'))
    selection_seconds = time.perf_counter() - started
    write_json(args.output / 'selection.json', {'rounds': rounds, 'history': history, 'classification': valid_classification})
    # 先释放特征矩阵，再在完整候选集上检查验证排序效果。
    del x, y, vx, vy
    gc.collect()
    valid_metrics = evaluate_snapshot(LightGBMCoarseRanker(booster, features), '2020-08-27', '2020-08-29', TRAIN, args, days)
    valid_metrics['classification'] = valid_classification
    write_json(args.output / 'validation_metrics.json', valid_metrics)
    print('VALIDATION', json.dumps(valid_metrics, ensure_ascii=False), flush=True)
    del booster
    gc.collect()
    # 固定验证选出的轮数，用训练和验证的历史样本重训，测试集不参与。
    x, y = load_labeled(train_paths + [valid_path])
    started = time.perf_counter()
    final = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=rounds)
    temporary = args.output / 'model.tmp.txt'
    final.save_model(str(temporary))
    temporary.replace(args.output / 'model.txt')
    importance = pd.DataFrame({'feature': features, 'gain': final.feature_importance(importance_type='gain'),
                               'split': final.feature_importance(importance_type='split')}).sort_values('gain', ascending=False)
    importance.to_csv(args.output / 'feature_importance.csv', index=False)
    write_json(metadata_path, {'schema': SCHEMA_VERSION, 'features': features, 'parameters': params,
                              'best_iteration': rounds, 'training_rows': len(y), 'positive_rate': float(y.mean()),
                              'lightgbm_version': lgb.__version__, 'cutoff': '2020-08-29',
                              'train_windows': DEFAULT_WINDOWS, 'validation_window': ['2020-08-27', '2020-08-29'],
                              'selection_seconds': selection_seconds,
                              'refit_seconds': time.perf_counter() - started})
    return LightGBMCoarseRanker(final, features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/ranking/lightgbm')
    parser.add_argument('--stage', choices=['all', 'train', 'test', 'recommend'], default='all')
    parser.add_argument('--route-k', type=int, default=200)
    parser.add_argument('--top-k', type=int, default=200)
    parser.add_argument('--rounds', type=int, default=1000)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--gsas-epochs', type=int, default=100)
    parser.add_argument('--gsas-patience', type=int, default=10)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--gpu-platform-id', type=int, default=0)
    parser.add_argument('--gpu-device-id', type=int, default=0)
    parser.add_argument('--gsas-final', type=Path, default=ROOT / 'artifacts/recall/gsasrec_500_patience20.pt')
    parser.add_argument('--test', type=Path, default=ROOT / 'data/processed/test.csv')
    parser.add_argument('--user-id', type=int, default=0)
    args = parser.parse_args()
    for key in ('route_k', 'top_k', 'rounds', 'patience', 'gsas_epochs', 'gsas_patience', 'threads'):
        if getattr(args, key) <= 0:
            parser.error(f'{key} 必须为正整数')
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / 'run_config.json'
    if args.stage == 'recommend':
        saved = json.loads(config_path.read_text(encoding='utf-8'))
        args.route_k = saved['route_k']
    else:
        settings = {'schema': SCHEMA_VERSION, 'route_k': args.route_k, 'top_k': args.top_k, 'rounds': args.rounds,
                    'patience': args.patience, 'gsas_epochs': args.gsas_epochs, 'gsas_patience': args.gsas_patience,
                    'gpu_platform_id': args.gpu_platform_id, 'gpu_device_id': args.gpu_device_id}
        if config_path.exists():
            saved = json.loads(config_path.read_text(encoding='utf-8'))
            if any(saved.get(key) != value for key, value in settings.items()):
                raise ValueError('输出目录包含另一份实验，请使用新的 --output 路径')
        else:
            write_json(config_path, settings)
    days = observed_days() if args.stage != 'recommend' else []
    if days:
        print('OBSERVED TRAIN DAYS', days, flush=True)
    ranker = train(args, days) if args.stage in ('all', 'train') else LightGBMCoarseRanker.load(args.output)
    if args.stage in ('all', 'test'):
        if not (args.output / 'test_metrics.json').exists():
            metrics = evaluate_snapshot(ranker, '2020-08-29', '2020-09-06', args.test, args, days, args.gsas_final)
            write_json(args.output / 'test_metrics.json', metrics)
            print('TEST', json.dumps(metrics, ensure_ascii=False), flush=True)
        else:
            prior = (args.output / 'test_metrics.json').read_text(encoding='utf-8')
            print('EXISTING TEST', prior, flush=True)
    if args.stage == 'recommend':
        from ranking.candidates import load_candidates
        store = FeatureStore('2020-08-29')
        path = args.output / 'recall/20200829/candidates.npz'
        if not path.exists():
            raise FileNotFoundError('请先完成 --stage all，以建立最终候选缓存')
        candidates = load_candidates(path, '2020-08-29', args.route_k)
        if args.user_id < 0:
            parser.error('user-id 不能为负')
        if not candidates.user_id.eq(args.user_id).any():
            # 新用户没有历史向量，使用同一截点的热门候选，再由粗排评分。
            candidates = hot_candidates(args.user_id, args.route_k, path.parent, store.eligible_ids)
        print(pd.DataFrame(ranker.recommend(args.user_id, candidates, store, args.top_k), columns=['video_id', 'score']).to_string(index=False))


if __name__ == '__main__':
    main()
