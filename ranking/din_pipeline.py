"""共享精排实验入口：三路召回 → LightGBM Top-200 → DIN/RankMixer Top-K。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from ranking.candidates import hot_candidates, load_candidates, write_json
from ranking.din_data import DINEncoder, encode_snapshots, positive_histories
from ranking.din_ranker import DINFineRanker, train_epoch
from ranking.features import ROOT, TRAIN, FeatureStore, first_feedback, local_time, positive_targets
from ranking.lightgbm_ranker import DEFAULT_WINDOWS, LightGBMCoarseRanker, classification_metrics, metric_summary

VALID_WINDOW = ('2020-08-27', '2020-08-29')
FINAL_CUTOFF = '2020-08-29'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def candidate_snapshot(coarse_dir, cutoff):
    directory = Path(coarse_dir) / 'recall' / local_time(cutoff).strftime('%Y%m%d')
    metadata = read_json(directory / 'candidates_meta.json')
    return load_candidates(directory / 'candidates.npz', cutoff, metadata['top_k'])


def prepare_snapshot(coarse_dir, output, start, end, max_length, reuse_output=None):
    """复用已验证的粗排数值特征，但不让未来训练出的粗排模型筛选历史样本。"""
    directory = Path(coarse_dir) / 'recall' / local_time(start).strftime('%Y%m%d')
    meta = read_json(directory / 'labeled_meta.json')
    if (meta['start'], meta['end']) != (start, end):
        raise ValueError('粗排标签缓存时间窗口不一致')
    settings = {'schema': 1, 'start': start, 'end': end, 'max_length': max_length}
    cache = Path(output) / 'samples' / f'{local_time(start):%Y%m%d}.npz'
    metadata_path = cache.with_suffix('.json')
    if reuse_output is not None and not cache.exists():
        reused = Path(reuse_output) / 'samples' / cache.name
        if reused.exists() and reused.with_suffix('.json').exists():
            saved = read_json(reused.with_suffix('.json'))
            if all(saved.get(key) == value for key, value in settings.items()):
                print(f'REUSE SAMPLES {reused}', flush=True)
                with np.load(reused, allow_pickle=False) as data:
                    return {k: data[k] for k in data.files}
    if cache.exists() and metadata_path.exists():
        saved = read_json(metadata_path)
        if any(saved.get(key) != value for key, value in settings.items()):
            raise ValueError('精排样本缓存已变化，请使用新输出目录')
        with np.load(cache, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    print(f'PREPARE SAMPLES {start}', flush=True)
    candidates = candidate_snapshot(coarse_dir, start)
    labels = first_feedback(TRAIN, start, end)
    samples = candidates.merge(labels, on=['user_id', 'video_id'], how='inner', validate='one_to_one')
    with np.load(directory / 'labeled.npz', allow_pickle=False) as data:
        if not np.array_equal(data['y'], samples.label.to_numpy()) or len(data['X']) != len(samples):
            raise ValueError('已有特征缓存与重建样本顺序不一致')
        result = {'X': data['X'], 'y': data['y'], 'features': data['features']}
    result['user_id'] = samples.user_id.to_numpy(dtype='int64')
    result['video_id'] = samples.video_id.to_numpy(dtype='int64')
    result['history_users'], result['history'] = positive_histories(TRAIN, start, max_length)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.with_suffix('.tmp').open('wb') as stream:
        np.savez_compressed(stream, **result)
    cache.with_suffix('.tmp').replace(cache)
    write_json(metadata_path, settings)
    print(f'SAMPLES cutoff={start} rows={len(samples)} history_users={len(result["history_users"])}', flush=True)
    return result


def new_ranker(snapshots, config, device, ranker_class=DINFineRanker):
    torch.manual_seed(config.seed)
    encoder = DINEncoder().fit(snapshots)
    ranker = ranker_class(encoder, config, device)
    optimizer = torch.optim.AdamW(ranker.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    return ranker, optimizer, np.random.default_rng(config.seed)


def train(coarse_dir, output, config, device, ranker_class=DINFineRanker, sample_cache=None):
    output = Path(output)
    model_name = ranker_class.model_name
    checkpoint = output / f'{model_name}.pt'
    windows = DEFAULT_WINDOWS + [VALID_WINDOW]
    settings = {'schema': 1, 'config': asdict(config)}
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / 'run_config.json'
    if config_path.exists():
        saved = read_json(config_path)
        if any(saved.get(key) != value for key, value in settings.items()):
            raise ValueError('输出目录已有不同的精排实验；请使用新的 --output')
    else:
        write_json(config_path, settings)
    if checkpoint.exists():
        return ranker_class.load(checkpoint, device)
    snapshots = [prepare_snapshot(coarse_dir, output, start, end, config.max_length, sample_cache) for start, end in windows]
    ranker, optimizer, rng = new_ranker(snapshots[:-1], config, device, ranker_class)
    training = encode_snapshots(ranker.encoder, snapshots[:-1])
    validation = encode_snapshots(ranker.encoder, snapshots[-1:])
    best, best_epoch, stale = float('inf'), 0, 0
    history = []
    selected_metrics = None
    print(f'TRAIN {model_name} device={device} rows={len(training["labels"])} valid={len(validation["labels"])} parameters={sum(p.numel() for p in ranker.model.parameters())}', flush=True)
    for epoch in range(1, config.epochs + 1):
        tick = time.perf_counter()
        loss = train_epoch(ranker, training, optimizer, rng)
        metrics = classification_metrics(validation['labels'], ranker.predict_encoded(validation))
        row = {'epoch': epoch, 'train_logloss': loss, 'validation': metrics, 'seconds': time.perf_counter() - tick}
        history.append(row)
        print('SELECT', json.dumps(row), flush=True)
        if metrics['logloss'] < best - 1e-6:
            best, best_epoch, stale = metrics['logloss'], epoch, 0
            selected_metrics = metrics
        else:
            stale += 1
        write_json(output / 'selection.json', {'best_epoch': best_epoch, 'history': history})
        if stale >= config.patience:
            break
    write_json(output / 'validation_metrics.json', selected_metrics)
    # 中断后允许重算选择阶段；永远不覆盖最终 checkpoint。
    del training, validation, optimizer, ranker
    gc.collect()
    ranker, optimizer, rng = new_ranker(snapshots, config, device, ranker_class)
    training = encode_snapshots(ranker.encoder, snapshots)
    del snapshots
    gc.collect()
    for epoch in range(1, best_epoch + 1):
        tick = time.perf_counter()
        loss = train_epoch(ranker, training, optimizer, rng)
        print(f'REFIT epoch={epoch}/{best_epoch} logloss={loss:.6f} seconds={time.perf_counter()-tick:.2f}', flush=True)
    ranker.metadata = {'cutoff': FINAL_CUTOFF, 'best_epoch': best_epoch,
                       'model_name': model_name, 'parameters': sum(p.numel() for p in ranker.model.parameters()),
                       'training_rows': len(training['labels']), 'validation': selected_metrics,
                       'label_policy': 'first_observed_complete; unknown_excluded; frozen_history',
                       'train_windows': DEFAULT_WINDOWS, 'validation_window': VALID_WINDOW}
    ranker.save(checkpoint)
    write_json(output / 'model_meta.json', ranker.metadata)
    print(f'SAVED {checkpoint}', flush=True)
    return ranker


def inference_context(args, ranker):
    if ranker.metadata.get('cutoff') != FINAL_CUTOFF:
        raise ValueError('精排推理截点不匹配')
    coarse_meta = read_json(args.coarse_dir / 'model_meta.json')
    if coarse_meta['cutoff'] != FINAL_CUTOFF:
        raise ValueError('粗排模型截止时间不匹配')
    coarse = LightGBMCoarseRanker.load(args.coarse_dir)
    store = FeatureStore(FINAL_CUTOFF)
    users, histories = positive_histories(TRAIN, FINAL_CUTOFF, ranker.config.max_length)
    candidates = candidate_snapshot(args.coarse_dir, FINAL_CUTOFF)
    return coarse, store, users, histories, candidates


def recommend(args, ranker):
    coarse, store, users, histories, candidates = inference_context(args, ranker)
    candidates = candidates.loc[candidates.user_id == args.user_id]
    if candidates.empty:
        candidates = hot_candidates(args.user_id, args.coarse_k, args.coarse_dir / 'recall/20200829', store.eligible_ids)
    recs = ranker.recommend(args.user_id, candidates, store, users, histories, args.top_k, coarse, args.coarse_k)
    frame = pd.DataFrame(recs, columns=['video_id', 'score'])
    path = args.output / f'recommendations_user_{args.user_id}.csv'
    frame.to_csv(path, index=False)
    print(frame.to_string(index=False), flush=True)
    print(f'RECOMMENDATIONS {path}', flush=True)


def evaluate(args, ranker, baseline=None):
    """同一粗排候选池比较 Top-K；不使用测试结果选参数。"""
    started = time.perf_counter()
    model_name = getattr(ranker, 'model_name', 'din')
    if baseline is not None:
        if baseline.model_name == model_name:
            raise ValueError('对照模型与被评估模型名称重复')
        if baseline.metadata.get('cutoff') != FINAL_CUTOFF:
            raise ValueError('对照模型截止时间不匹配')
    settings = {'top_k': args.top_k, 'coarse_k': args.coarse_k}
    target_path = args.output / f'test_metrics_k{args.top_k}.json'
    if target_path.exists():
        previous = read_json(target_path)
        if (any(previous['experiment'].get(key) != value for key, value in settings.items())
                or model_name not in previous
                or (baseline is not None and baseline.model_name not in previous)):
            raise ValueError('已有测试结果与当前候选配置不同，请使用新输出目录')
        print('EXISTING TEST', json.dumps(previous, ensure_ascii=False), flush=True)
        return previous
    coarse, store, users, histories, candidates = inference_context(args, ranker)
    rankers = {model_name: (ranker, users, histories)}
    if baseline is not None:
        base_users, base_histories = ((users, histories) if baseline.config.max_length == ranker.config.max_length
                                     else positive_histories(TRAIN, FINAL_CUTOFF, baseline.config.max_length))
        rankers[baseline.model_name] = (baseline, base_users, base_histories)
    targets = positive_targets(args.test, FINAL_CUTOFF, '2020-09-06')
    recommendations = {name: {} for name in (*rankers, 'lightgbm')}
    pools = {}
    # 缓存已按 user_id/video_id 排序；整个用户分组处理，避免分块截断同一用户。
    unique = candidates.user_id.unique()
    bounds = np.append(np.searchsorted(candidates.user_id.to_numpy(), unique), len(candidates))
    predict_seconds = {name: 0.0 for name in rankers}
    for start in range(0, len(unique), 64):
        frame = candidates.iloc[bounds[start]:bounds[min(start + 64, len(unique))]].reset_index(drop=True)
        features = store.transform(frame)
        frame['coarse_score'] = coarse.predict(features)
        selected = frame.sort_values(['user_id', 'coarse_score', 'video_id'], ascending=[True, False, True]).groupby('user_id', sort=False).head(args.coarse_k)
        selected = selected.copy()
        for name, (model, model_users, model_histories) in rankers.items():
            tick = time.perf_counter()
            selected[f'{name}_score'] = model.predict(selected, features.iloc[selected.index], model_users, model_histories)
            predict_seconds[name] += time.perf_counter() - tick
        for user, group in selected.groupby('user_id', sort=False):
            user = int(user)
            pools[user] = group.video_id.tolist()
            recommendations['lightgbm'][user] = group.video_id.head(args.top_k).tolist()
            for name in rankers:
                recommendations[name][user] = group.sort_values([f'{name}_score', 'video_id'], ascending=[False, True]).video_id.head(args.top_k).tolist()
        if start % 1024 == 0:
            print(f'EVALUATE {model_name} users={min(start + 64, len(unique))}/{len(unique)}', flush=True)
    result = {name: metric_summary(recs, targets, pools, args.top_k) for name, recs in recommendations.items()}
    result.update({f'{name}_predict_seconds': seconds for name, seconds in predict_seconds.items()})
    result.update(experiment=settings, total_seconds=time.perf_counter() - started,
                  label_policy='any_complete_in_window; all_positive_pairs_in_denominator; frozen_history')
    write_json(target_path, result)
    print('TEST', json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main(default_model='din'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['din', 'rankmixer'], default=default_model)
    parser.add_argument('--stage', choices=['train', 'recommend', 'test', 'all'], default='train')
    parser.add_argument('--coarse-dir', type=Path, default=ROOT / 'artifacts/ranking/lightgbm')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--max-length', type=int, default=50)
    parser.add_argument('--embedding-dim', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--token-dim', type=int, default=64, help='RankMixer token 宽度（8 的倍数）')
    parser.add_argument('--layers', type=int, default=2, help='RankMixer block 层数')
    parser.add_argument('--expansion', type=int, default=2, help='RankMixer PFFN 扩展倍数')
    parser.add_argument('--sample-cache', type=Path, help='只读复用样本的实验目录；不匹配时在自己的输出目录重建')
    parser.add_argument('--din-checkpoint', type=Path, default=ROOT / 'artifacts/ranking/din/din.pt', help='RankMixer 测试时同时评分的 DIN 基线')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--coarse-k', type=int, default=200)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--user-id', type=int, default=0)
    parser.add_argument('--test', type=Path, default=ROOT / 'data/processed/test.csv')
    args = parser.parse_args()
    args.output = args.output or ROOT / f'artifacts/ranking/{args.model}'
    if args.threads <= 0 or args.coarse_k <= 0 or not 0 < args.top_k <= args.coarse_k or args.user_id < 0:
        parser.error('要求 threads/coarse-k 为正、0 < top-k <= coarse-k、user-id 非负')
    torch.set_num_threads(args.threads)
    ranker_class = DINFineRanker
    options = {}
    if args.model == 'rankmixer':
        from ranking.rankmixer_ranker import RankMixerFineRanker
        ranker_class = RankMixerFineRanker
        options = dict(token_dim=args.token_dim, layers=args.layers, expansion=args.expansion)
        args.sample_cache = args.sample_cache or ROOT / 'artifacts/ranking/din'
    config = ranker_class.config_type(max_length=args.max_length, embedding_dim=args.embedding_dim, batch_size=args.batch_size,
                                      epochs=args.epochs, patience=args.patience, learning_rate=args.learning_rate, **options)
    ranker = (train(args.coarse_dir, args.output, config, args.device, ranker_class, args.sample_cache) if args.stage in ('train', 'all')
              else ranker_class.load(args.output / f'{args.model}.pt', args.device))
    if args.stage in ('test', 'all'):
        baseline = DINFineRanker.load(args.din_checkpoint, args.device) if args.model == 'rankmixer' else None
        evaluate(args, ranker, baseline)
    if args.stage == 'recommend':
        recommend(args, ranker)


if __name__ == '__main__':
    main()
