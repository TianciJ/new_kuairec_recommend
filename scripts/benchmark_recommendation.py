"""Measure cached recommendations and serial live recall with existing checkpoints."""
from __future__ import annotations

import time

IMPORT_STARTED = time.perf_counter()

import argparse
from datetime import datetime
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from ranking.candidates import hot_candidates, merge_routes, write_json
from ranking.din_data import positive_histories
from ranking.din_pipeline import FINAL_CUTOFF, candidate_snapshot
from ranking.din_ranker import DINFineRanker
from ranking.features import SIGNAL_COLUMNS, TRAIN, FeatureStore
from ranking.lightgbm_ranker import LightGBMCoarseRanker
import lightgbm
from ranking.rankmixer_ranker import RankMixerFineRanker
from recall.gsasrec_recall import GSASRecRecall
from recall.hot_recall import HotRecall
from recall.itemcf_recall import ItemCFRecall

IMPORT_MS = (time.perf_counter() - IMPORT_STARTED) * 1000


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    context, setup = load_context(args)
    first = run_request(context, 0, 'cached', 'rankmixer')
    first['phase'] = 'first_request'
    startup_ms = IMPORT_MS + sum(setup.values()) + first['total_ms']
    script_to_first_ms = (time.perf_counter() - IMPORT_STARTED) * 1000
    context['rankers']['din'] = measure(
        setup, 'din_load_ms', lambda: DINFineRanker.load(args.din, args.device), gpu=True)
    context['hot'] = measure(setup, 'hot_load_ms', lambda: HotRecall.load(context['recall_dir'] / 'hot.npz'))
    context['itemcf'] = measure(setup, 'itemcf_load_ms', lambda: ItemCFRecall.load(context['recall_dir'] / 'itemcf.npz'))
    context['gsasrec'] = measure(
        setup, 'gsasrec_load_ms', lambda: GSASRecRecall.load(args.gsasrec, args.device), gpu=True)
    verification = verify(context)
    users = sample_users(context, args.users, args.seed)
    variants = [('cached', 'din'), ('cached', 'rankmixer'), ('live', 'din'), ('live', 'rankmixer')]
    for _ in range(args.warmups):
        for mode, model in variants:
            run_request(context, 0, mode, model)
    records = [first]
    rng = np.random.default_rng(args.seed)
    for repeat in range(args.repeats):
        for index, user in enumerate(rng.permutation(users)):
            for variant in rng.permutation(len(variants)):
                mode, model = variants[variant]
                row = run_request(context, int(user), mode, model)
                row.update(phase='warm_request', repeat=repeat)
                records.append(row)
            if (index + 1) % 16 == 0 or index + 1 == len(users):
                print(f'PROGRESS repeat={repeat + 1}/{args.repeats} users={index + 1}/{len(users)}', flush=True)
    for repeat in range(args.repeats):
        for mode, model in variants:
            row = run_request(context, 999999, mode, model)
            row.update(phase='cold_user', repeat=repeat)
            records.append(row)
    frame = pd.DataFrame(records)
    summary = summarize(frame)
    result = {
        'environment': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
                        'torch': torch.__version__, 'numpy': np.__version__, 'pandas': pd.__version__,
                        'lightgbm': lightgbm.__version__, 'device': args.device,
                        'gpu': torch.cuda.get_device_name() if args.device == 'cuda' else None,
                        'torch_threads': torch.get_num_threads()},
        'settings': {'users': users.tolist(), 'repeats': args.repeats, 'warmups': args.warmups,
                     'seed': args.seed, 'route_k': 200, 'coarse_k': 200, 'top_k': 20,
                     'cutoff': FINAL_CUTOFF, 'request_execution': 'serial',
                     'checkpoints': {'din': str(args.din), 'rankmixer': str(args.rankmixer),
                                     'gsasrec': str(args.gsasrec)}},
        'import_ms': IMPORT_MS, 'setup_ms': setup,
        'initialization_and_first_cached_rankmixer_request_ms': startup_ms,
        'script_start_to_first_cached_rankmixer_request_ms': script_to_first_ms,
        'verification': verification, 'summary': summary,
        'notes': [
            'Cached mode follows the existing CLI: scan the in-memory full candidate frame for one user.',
            'Live mode really calls all three existing recall APIs serially, then merges their results.',
            'Warm requests retain models, FeatureStore, histories and candidate cache in memory.',
            'Fine scoring uses the unchanged predict API, including encoding all users histories.',
            'CUDA calls are synchronized at timed stage boundaries; fine scoring includes encoding and transfers.',
            'Startup is one process observation with uncontrolled OS disk cache, excluding interpreter launch and exit.',
            'A new user is reported separately because ItemCF and gSASRec return empty results.',
            'No models are trained, no test labels are read, and no production code or checkpoints are changed.',
        ],
    }
    frame.to_csv(args.output / 'requests.csv', index=False)
    write_json(args.output / 'summary.json', result)
    for name, item in summary.items():
        timing = item['timing_ms']['total_ms']
        print(f"SUMMARY {name} n={item['requests']} p50_ms={timing['p50']:.3f} p95_ms={timing['p95']:.3f}", flush=True)
    print(f'OUTPUT {args.output}', flush=True)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/benchmarks' / datetime.now().strftime('recommendation_%Y%m%d_%H%M%S'))
    parser.add_argument('--users', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--coarse-dir', type=Path, default=ROOT / 'artifacts/ranking/lightgbm')
    parser.add_argument('--din', type=Path, default=ROOT / 'artifacts/ranking/din/din.pt')
    parser.add_argument('--rankmixer', type=Path, default=ROOT / 'artifacts/ranking/rankmixer/rankmixer.pt')
    parser.add_argument('--gsasrec', type=Path, default=ROOT / 'artifacts/recall/gsasrec_500_patience20.pt')
    args = parser.parse_args()
    if min(args.users, args.repeats, args.warmups, args.threads) <= 0:
        parser.error('users/repeats/warmups/threads must be positive')
    return args


def measure(times, name, operation, gpu=False, announce=False):
    if gpu and torch.cuda.is_initialized():
        torch.cuda.synchronize()
    start = time.perf_counter()
    value = operation()
    if gpu and torch.cuda.is_initialized():
        torch.cuda.synchronize()
    times[name] = (time.perf_counter() - start) * 1000
    if announce:
        print(f'SETUP {name}={times[name]:.3f}', flush=True)
    return value


def load_context(args):
    times = {}
    if args.device == 'cuda':
        measure(times, 'cuda_init_ms', torch.cuda.init, gpu=True, announce=True)
    ranker = measure(times, 'rankmixer_load_ms', lambda: RankMixerFineRanker.load(args.rankmixer, args.device), gpu=True, announce=True)
    coarse = measure(times, 'lightgbm_load_ms', lambda: LightGBMCoarseRanker.load(args.coarse_dir), announce=True)
    store = measure(times, 'feature_store_build_ms', lambda: FeatureStore(FINAL_CUTOFF), announce=True)
    users, history = measure(times, 'positive_histories_build_ms',
                             lambda: positive_histories(TRAIN, FINAL_CUTOFF, ranker.config.max_length), announce=True)
    candidates = measure(times, 'candidate_cache_load_ms', lambda: candidate_snapshot(args.coarse_dir, FINAL_CUTOFF), announce=True)
    return {'rankers': {'rankmixer': ranker}, 'coarse': coarse, 'store': store,
            'history_users': users, 'history': history, 'candidates': candidates,
            'recall_dir': args.coarse_dir / 'recall/20200829'}, times


def live_candidates(context, user, times):
    routes = {}
    for name in ('hot', 'itemcf', 'gsasrec'):
        routes[name] = measure(times, f'{name}_recall_ms',
                               lambda: context[name].recommend(user, 200, candidate_ids=context['store'].eligible_ids),
                               gpu=name == 'gsasrec')
        times[f'{name}_count'] = len(routes[name])
    def merged():
        ids, signals = merge_routes(routes)
        frame = pd.DataFrame(signals, columns=SIGNAL_COLUMNS)
        frame.insert(0, 'video_id', ids)
        frame.insert(0, 'user_id', user)
        return frame
    return measure(times, 'merge_ms', merged)


def run_request(context, user, mode, model_name, return_result=False):
    times = {'user_id': user, 'mode': mode, 'model': model_name}
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
    start = time.perf_counter()
    if mode == 'live':
        candidates = live_candidates(context, user, times)
    else:
        candidates = measure(times, 'candidate_lookup_ms',
                             lambda: context['candidates'].loc[context['candidates'].user_id == user])
        if candidates.empty:
            candidates = measure(times, 'hot_fallback_load_and_recall_ms',
                                 lambda: hot_candidates(user, 200, context['recall_dir'], context['store'].eligible_ids))
    features = measure(times, 'feature_transform_ms', lambda: context['store'].transform(candidates))
    coarse_scores = measure(times, 'lightgbm_predict_ms', lambda: context['coarse'].predict(features))
    coarse_order = measure(times, 'coarse_top200_ms',
                           lambda: np.lexsort((candidates.video_id.to_numpy(), -coarse_scores))[:200])
    selected, selected_features = measure(times, 'coarse_slice_ms',
                                          lambda: (candidates.iloc[coarse_order], features.iloc[coarse_order]))
    ranker = context['rankers'][model_name]
    fine_scores = measure(times, 'fine_predict_ms',
                          lambda: profiled_predict(ranker, selected, selected_features, context, times), gpu=True)
    ids = selected.video_id.to_numpy()
    order = measure(times, 'fine_top20_ms', lambda: np.lexsort((ids, -fine_scores))[:20])
    recommendations = [(int(ids[i]), float(fine_scores[i])) for i in order]
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
    times['total_ms'] = (time.perf_counter() - start) * 1000
    times.update(candidate_count=len(candidates), coarse_count=len(selected), result_count=len(recommendations))
    return (times, recommendations, candidates) if return_result else times


def profiled_predict(ranker, candidates, features, context, times):
    original = ranker.predict_encoded
    started = time.perf_counter()

    def network(data):
        times['fine_encoding_ms'] = (time.perf_counter() - started) * 1000
        return measure(times, 'fine_network_and_transfer_ms', lambda: original(data), gpu=True)

    ranker.predict_encoded = network
    try:
        return ranker.predict(candidates, features, context['history_users'], context['history'])
    finally:
        ranker.predict_encoded = original


def sample_users(context, count, seed):
    eligible = np.intersect1d(context['candidates'].user_id.unique(), list(context['gsasrec'].sequences))
    if count > len(eligible):
        raise ValueError('requested more users than available warm users')
    others = np.setdiff1d(eligible, [0])
    return np.r_[0, np.random.default_rng(seed).choice(others, count - 1, replace=False)].astype('int64')


def verify(context):
    checks = {}
    for mode in ('cached', 'live'):
        for model_name, ranker in context['rankers'].items():
            for user in (0, 999999):
                _, results, candidates = run_request(context, user, mode, model_name, return_result=True)
                reference = ranker.recommend(user, candidates, context['store'], context['history_users'],
                                             context['history'], top_k=20, coarse_ranker=context['coarse'], coarse_k=200)
                assert results == reference, (mode, model_name, user, 'staged result differs from existing API')
                ids = [v for v, _ in results]
                assert len(ids) == len(set(ids)) == 20
                assert not (set(ids) & set(context['hot'].seen.getrow(user).indices) if user < context['hot'].seen.shape[0] else set())
                checks[f'{mode}_{model_name}_{user}'] = {'matches_existing_recommend_api': True, 'result_count': len(ids)}
    cached = context['candidates'].loc[context['candidates'].user_id == 0].reset_index(drop=True)
    live = live_candidates(context, 0, {}).reset_index(drop=True)
    pd.testing.assert_frame_equal(cached, live, check_dtype=False, check_exact=False, rtol=1e-5, atol=1e-6)
    checks['live_matches_cached_user_0_candidates'] = True
    print('VERIFICATION passed', flush=True)
    return checks


def summarize(frame):
    summaries = {}
    for (phase, mode, model), group in frame.groupby(['phase', 'mode', 'model']):
        values = {}
        for column in [c for c in frame if c.endswith('_ms')]:
            series = group[column].dropna().to_numpy()
            if len(series):
                values[column] = {'mean': float(np.mean(series)), 'p50': float(np.percentile(series, 50)),
                                  'p95': float(np.percentile(series, 95)), 'max': float(np.max(series))}
        summaries[f'{phase}/{mode}/{model}'] = {'requests': len(group), 'timing_ms': values,
            'candidate_count_mean': float(group.candidate_count.mean()),
            'coarse_count_mean': float(group.coarse_count.mean()), 'result_count_min': int(group.result_count.min())}
    return summaries


if __name__ == '__main__':
    main()
