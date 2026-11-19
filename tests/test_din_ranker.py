"""DIN 的时间隔离、注意力、训练梯度、序列映射和级联推理契约。"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from ranking.din_data import DINEncoder, encode_snapshots, positive_histories
from ranking.din_ranker import DINConfig, DINFineRanker, train_epoch
from ranking.features import INTERACTION_COLUMNS, local_time


def snapshot(n=128):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(n, 2)).astype('float32')
    x[0, 1] = np.nan
    return {'X': x, 'features': np.array(['a', 'b']), 'y': (x[:, 0] > 0).astype('float32'),
            'user_id': np.arange(n) % 2, 'video_id': np.arange(n) % 3,
            'history_users': np.array([0, 1]), 'history': np.array([[0, 1, -1], [2, -1, -1]])}


class DINTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(4)
        self.config = DINConfig(max_length=3, embedding_dim=8, hidden_dim=16,
                                batch_size=32, dropout=0, epochs=3)
        self.sample = snapshot()
        self.encoder = DINEncoder().fit([self.sample])
        self.ranker = DINFineRanker(self.encoder, self.config)

    def test_history_excludes_boundary_future_negative_and_exact_duplicates(self):
        cutoff = local_time('2020-08-03').timestamp()
        rows = [[0, 9, cutoff, 10, 10], [0, 0, cutoff - 5, 10, 10],
                [0, 1, cutoff - 4, 1, 10], [0, 2, cutoff - 3, 20, 10],
                [0, 0, cutoff - 2, 10, 10], [0, 0, cutoff - 2, 10, 10],
                [1, 8, cutoff + 1, 10, 10], [2, 3, cutoff - 1, 10, 10]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'train.csv'
            pd.DataFrame(rows, columns=INTERACTION_COLUMNS).to_csv(path, index=False)
            users, history = positive_histories(path, '2020-08-03', max_length=2, chunksize=2)
            np.testing.assert_array_equal(users, [0, 2])
            np.testing.assert_array_equal(history, [[2, 0], [3, -1]])
            _, empty = positive_histories(path, '2020-08-01', 2)
            self.assertEqual(empty.shape, (0, 2))

    def test_encoder_no_validation_fit_and_padding_unknown_id_zero(self):
        before = self.encoder.state()
        validation = snapshot(8)
        validation['X'][:] = 1e10
        validation['user_id'][:] = 999
        validation['video_id'][:] = 999
        encoded = encode_snapshots(self.encoder, [validation])
        self.assertEqual(before, self.encoder.state())
        self.assertTrue((encoded['users'] == 0).all())
        self.assertTrue((encoded['items'] == 1).all())
        self.assertTrue((encoded['histories'][encoded['history_row']] == 0).all())
        self.assertEqual(self.encoder.item_ids([-1, 999, 0]).tolist(), [0, 1, 2])
        self.assertTrue(np.isfinite(encoded['dense']).all())
        with self.assertRaises(ValueError):
            self.encoder.dense(self.sample['X'], ['b', 'a'])

    def test_histories_are_separate_between_time_snapshots(self):
        other = snapshot()
        other['history'] = np.array([[2, 2, 2], [0, -1, -1]])
        data = encode_snapshots(self.encoder, [self.sample, other])
        first = data['histories'][data['history_row'][0]]
        second = data['histories'][data['history_row'][128]]
        self.assertNotEqual(first.tolist(), second.tolist())

    def test_attention_target_conditioned_padding_and_gradients(self):
        model = self.ranker.model
        query = torch.randn(1, 8, requires_grad=True)
        history = torch.tensor([[2, 3]])
        interest = model.interest(query, history)
        padded = model.interest(query, torch.tensor([[2, 3, 0, 0]]))
        torch.testing.assert_close(interest, padded)
        torch.testing.assert_close(model.interest(query, torch.zeros(1, 3, dtype=torch.long)), torch.zeros(1, 8))
        changed = model.interest(query + 2, history)
        self.assertGreater((interest - changed).abs().max().item(), 1e-8)
        interest.sum().backward()
        self.assertGreater(query.grad.abs().sum().item(), 0)
        self.assertGreater(model.item_embedding.weight.grad[2].abs().sum().item(), 0)
        self.assertEqual(model.item_embedding.weight.grad[0].abs().sum().item(), 0)

    def test_training_reduces_loss_and_roundtrip(self):
        data = encode_snapshots(self.encoder, [self.sample])
        optimizer = torch.optim.Adam(self.ranker.model.parameters(), lr=0.01)
        rng = np.random.default_rng(4)
        losses = [train_epoch(self.ranker, data, optimizer, rng) for _ in range(8)]
        self.assertLess(losses[-1], losses[0] * .75)
        expected = self.ranker.predict_encoded(data)
        self.assertTrue(np.isfinite(expected).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'din.pt'
            self.ranker.save(path)
            loaded = DINFineRanker.load(path)
            np.testing.assert_allclose(loaded.predict_encoded(data), expected, atol=1e-7)
            with self.assertRaises(FileExistsError):
                self.ranker.save(path)

    def test_recommend_only_reranks_coarse_pool_and_handles_cold_start(self):
        class Store:
            def transform(self, candidates):
                return pd.DataFrame({'a': candidates.video_id.to_numpy(dtype='float32'), 'b': np.zeros(len(candidates), dtype='float32')})

        class Coarse:
            def predict(self, features):
                return features.a.to_numpy()

        candidates = pd.DataFrame({'user_id': [999] * 4, 'video_id': [0, 1, 2, 999]})
        recs = self.ranker.recommend(999, candidates, Store(), self.sample['history_users'], self.sample['history'],
                                     top_k=2, coarse_ranker=Coarse(), coarse_k=2)
        self.assertEqual({video for video, score in recs}, {2, 999})
        self.assertTrue(all(np.isfinite(score) for video, score in recs))
        self.assertGreaterEqual(recs[0][1], recs[1][1])
        self.assertEqual(self.ranker.recommend(3, candidates, Store(), self.sample['history_users'], self.sample['history']), [])

    def test_evaluation_aligns_rows_across_user_batches_and_uses_same_pool(self):
        from ranking.din_pipeline import evaluate

        class Store:
            def transform(self, candidates):
                return pd.DataFrame({'a': candidates.video_id.to_numpy(dtype='float32')})

        class Coarse:
            def predict(self, features):
                return features.a.to_numpy()

        class Fine:
            def predict(self, candidates, features, users, histories):
                np.testing.assert_array_equal(candidates.video_id.to_numpy(), features.a.to_numpy())
                return -features.a.to_numpy()

        candidates = pd.DataFrame({'user_id': np.repeat(np.arange(65), 3), 'video_id': np.tile([0, 1, 2], 65)})
        targets = {u: {0, 1} for u in range(65)}
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory), coarse_dir=Path(directory), test=Path(directory) / 'test.csv',
                                   top_k=1, coarse_k=2)
            with patch('ranking.din_pipeline.inference_context', return_value=(Coarse(), Store(), [], [], candidates)), \
                    patch('ranking.din_pipeline.positive_targets', return_value=targets):
                result = evaluate(args, Fine())
        self.assertEqual(result['din']['users'], 65)
        self.assertEqual(result['din']['hits'], 65)
        self.assertEqual(result['lightgbm']['hits'], 0)
        self.assertEqual(result['din']['pool_hits'], result['lightgbm']['pool_hits'])
        self.assertEqual(result['din']['pool_hits'], 65)


if __name__ == '__main__':
    unittest.main()
