"""时间边界、未知标签、多类别、候选融合和评估分母的行为检查。"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ranking.features import FeatureStore, first_feedback, positive_targets, local_time, COUNT_COLUMNS, SIGNAL_COLUMNS
from ranking.candidates import merge_routes, internal_validation_days, save_candidates, load_candidates
from ranking.lightgbm_ranker import LightGBMCoarseRanker, metric_summary, lgb


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.train, self.daily, self.categories = [self.root / name for name in ('train.csv', 'daily.csv', 'categories.csv')]
        start = local_time('2020-08-03').timestamp()
        # 未按时间排序；边界时刻的记录不能出现在历史中。
        pd.DataFrame([
            [0, 2, start + 1, 5, 10], [0, 0, start - 100, 10, 10],
            [1, 1, start - 200, 2, 10], [0, 1, start - 300, 5, 10],
            [0, 2, start + 10, 20, 10], [0, 3, start, 20, 10],
        ], columns=['user_id', 'video_id', 'timestamp', 'play_duration', 'video_duration']).to_csv(self.train, index=False)
        records = []
        for video in range(4):
            row = {'video_id': video, 'date': 20200802, 'author_id': video % 2,
                   'video_duration': 10000, 'video_type': 'NORMAL', 'upload_dt': '2020-08-01'}
            row.update({name: 2 for name in COUNT_COLUMNS})
            row.update(play_cnt=10, show_cnt=20, complete_play_cnt=4)
            records.append(row)
            future = dict(row, date=20200803, play_cnt=9999, video_duration=999999)
            records.append(future)
        pd.DataFrame(records).to_csv(self.daily, index=False)
        pd.DataFrame({'video_id': range(4), 'feat': ['[0]', '[1]', '[1, 0, 1]', '[2]']}).to_csv(self.categories, index=False)

    def tearDown(self):
        self.temp.cleanup()

    def candidate(self, user=0, video=2):
        return pd.DataFrame([dict(user_id=user, video_id=video, **{name: 1.0 for name in SIGNAL_COLUMNS})])

    def test_features_exclude_future_and_handle_multi_category(self):
        store = FeatureStore('2020-08-03', self.train, self.daily, self.categories)
        x = store.transform(self.candidate())
        self.assertEqual(x.u_watch_recent.iloc[0], 2)
        self.assertEqual(x.u_watch_1d.iloc[0], 2)
        self.assertEqual(x.i_duration_s.iloc[0], 10)
        self.assertEqual(x.i_play_cnt_1d.iloc[0], 10)
        self.assertAlmostEqual(x.i_complete_rate_1d.iloc[0], .4)
        self.assertEqual(x.i_category_count.iloc[0], 2)
        self.assertAlmostEqual(x.uc_watch_share_mean.iloc[0], .5)
        self.assertNotIn('label', x.columns)
        self.assertNotIn('play_duration', x.columns)
        # 任意改变未来标签和当天统计，都不应改变这份历史快照。
        raw = pd.read_csv(self.train)
        raw.loc[raw.timestamp >= local_time('2020-08-03').timestamp(), 'play_duration'] = 100000
        raw.to_csv(self.train, index=False)
        same = FeatureStore('2020-08-03', self.train, self.daily, self.categories).transform(self.candidate())
        pd.testing.assert_frame_equal(x, same)

    def test_unknown_user_and_video(self):
        store = FeatureStore('2020-08-03', self.train, self.daily, self.categories)
        x = store.transform(self.candidate(999, 999))
        self.assertEqual(x.u_no_history.iloc[0], 1)
        self.assertEqual(x.u_watch_recent.iloc[0], 0)
        self.assertTrue(np.isnan(x.i_duration_s.iloc[0]))
        self.assertEqual(x.ua_watch_recent.iloc[0], 0)

    def test_labels_first_observation_unknown_not_negative(self):
        labels = first_feedback(self.train, '2020-08-03', '2020-08-04')
        self.assertEqual(dict(zip(labels.video_id, labels.label)), {3: 1, 2: 0})
        self.assertNotIn(0, labels.video_id.tolist())
        # 评估仍使用窗口内任意完播，所以第二次完播的 video=2 也是评估正例。
        self.assertEqual(positive_targets(self.train, '2020-08-03', '2020-08-04'), {0: {2, 3}})

    def test_merge_missing_negative_scores_and_roundtrip(self):
        ids, signals = merge_routes({'hot': [(2, 10), (1, 9)], 'itemcf': [(2, .4)], 'gsasrec': [(3, -8), (2, -9)]})
        self.assertEqual(ids.tolist(), [1, 2, 3])
        self.assertTrue(np.isnan(signals[0, 2]))
        self.assertEqual(signals[2, 4], -8)
        frame = pd.DataFrame(signals, columns=SIGNAL_COLUMNS)
        frame.insert(0, 'video_id', ids)
        frame.insert(0, 'user_id', np.zeros(3, dtype='int32'))
        path = self.root / 'candidates.npz'
        save_candidates(path, frame, '2020-08-03', 200)
        pd.testing.assert_frame_equal(frame, load_candidates(path, '2020-08-03', 200))
        with self.assertRaises(ValueError):
            load_candidates(path, '2020-08-04', 200)

    def test_metrics_keep_unreachable_and_empty_users(self):
        result = metric_summary({0: [2, 1]}, {0: {2, 9}, 1: {8}}, {0: [1, 2, 3]}, k=2)
        self.assertEqual(result['users'], 2)
        self.assertEqual(result['positive_pairs'], 3)
        self.assertAlmostEqual(result['macro_recall@2'], .25)
        self.assertAlmostEqual(result['micro_recall@2'], 1 / 3)
        self.assertEqual(result['candidate_positive_retention'], 1)

    def test_validation_gap_and_duplicate_metadata(self):
        self.assertEqual(internal_validation_days('2020-08-27', [20200807, 20200808, 20200809, 20200810]), 18)
        df = pd.read_csv(self.daily)
        pd.concat([df, df.iloc[[0]]]).to_csv(self.daily, index=False)
        with self.assertRaises(ValueError):
            FeatureStore('2020-08-03', self.train, self.daily, self.categories)

    def test_model_serialization_and_feature_order(self):
        rng = np.random.default_rng(2)
        x = pd.DataFrame(rng.normal(size=(200, 3)).astype('float32'), columns=['a', 'b', 'c'])
        y = (x.a > 0).astype('int8')
        # 小规模契约检查用显式 CPU；正式训练固定 gpu，并单独做真实 GPU 检查。
        model = lgb.train({'objective': 'binary', 'device_type': 'cpu', 'num_threads': 2, 'verbosity': -1}, lgb.Dataset(x, label=y), num_boost_round=5)
        ranker = LightGBMCoarseRanker(model, x.columns)
        expected = ranker.predict(x)
        model.save_model(str(self.root / 'model.txt'))
        loaded = LightGBMCoarseRanker(lgb.Booster(model_file=str(self.root / 'model.txt')), x.columns)
        np.testing.assert_allclose(expected, loaded.predict(x))
        with self.assertRaises(ValueError):
            loaded.predict(x[['b', 'a', 'c']])


if __name__ == '__main__':
    unittest.main()
