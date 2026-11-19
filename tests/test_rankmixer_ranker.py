"""RankMixer 核心算子、兼容性、冷启动与训练/持久化契约。"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from ranking.din_data import DINEncoder, encode_snapshots
from ranking.din_ranker import DINConfig, DINFineRanker
from ranking.neural_ranker import train_epoch
from ranking.rankmixer_ranker import RankMixerConfig, RankMixerFineRanker, TokenMixing, PerTokenFFN, semantic_groups


def make_snapshot(n=128):
    x = np.random.default_rng(7).normal(size=(n, 5)).astype('float32')
    x[0, 2] = np.nan
    return {'X': x, 'features': np.array(['hot_score', 'u_watch_recent', 'i_age_days', 'i_category_0', 'uc_watch_share_mean']),
            'y': (x[:, 1] > 0).astype('float32'), 'user_id': np.arange(n) % 2, 'video_id': np.arange(n) % 3,
            'history_users': np.array([0, 1]), 'history': np.array([[0, 1, -1], [2, -1, -1]])}


class RankMixerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(4)
        self.config = RankMixerConfig(max_length=3, embedding_dim=8, token_dim=16, layers=2,
                                      dropout=0, batch_size=32, epochs=2, patience=1)
        self.snapshot = make_snapshot()
        self.encoder = DINEncoder().fit([self.snapshot])
        self.ranker = RankMixerFineRanker(self.encoder, self.config)

    def test_mixing_matches_head_token_permutation_and_has_no_weights(self):
        mixing = TokenMixing()
        x = torch.arange(2 * 4 * 8).reshape(2, 4, 8)
        result = mixing(x)
        expected = torch.empty_like(x)
        for b in range(2):
            for t in range(4):
                for h in range(4):
                    expected[b, h, t * 2:(t + 1) * 2] = x[b, t, h * 2:(h + 1) * 2]
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(mixing(result), x)
        self.assertEqual(sum(p.numel() for p in mixing.parameters()), 0)
        with self.assertRaises(ValueError):
            mixing(torch.zeros(1, 3, 8))

    def test_per_token_ffn_has_independent_parameters_and_matches_reference(self):
        ffn = PerTokenFFN(3, 8, 2)
        x = torch.randn(2, 3, 8)
        reference = torch.stack([torch.nn.functional.gelu(x[:, t] @ ffn.w1[t] + ffn.b1[t]) @ ffn.w2[t] + ffn.b2[t]
                                 for t in range(3)], dim=1)
        torch.testing.assert_close(ffn(x), reference)
        ffn(x)[:, 0].sum().backward()
        self.assertGreater(ffn.w1.grad[0].abs().sum().item(), 0)
        self.assertEqual(ffn.w1.grad[1:].abs().sum().item(), 0)

    def test_semantic_groups_partition_values_and_missing_indicators(self):
        groups = semantic_groups(self.encoder.features)
        indices = [i for group in groups.values() for i in group]
        self.assertEqual(sorted(indices), list(range(10)))
        with self.assertRaises(ValueError):
            semantic_groups(['unknown_feature'])
        with self.assertRaises(ValueError):
            RankMixerConfig(token_dim=63)

    def test_training_roundtrip_and_wrong_model_rejected(self):
        data = encode_snapshots(self.encoder, [self.snapshot])
        optimizer = torch.optim.AdamW(self.ranker.model.parameters(), lr=.003)
        rng = np.random.default_rng(4)
        losses = [train_epoch(self.ranker, data, optimizer, rng) for _ in range(12)]
        self.assertLess(losses[-1], .75 * losses[0])
        scores = self.ranker.predict_encoded(data)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rankmixer.pt'
            self.ranker.save(path)
            loaded = RankMixerFineRanker.load(path)
            np.testing.assert_allclose(loaded.predict_encoded(data), scores, atol=1e-7)
            with self.assertRaises(ValueError):
                DINFineRanker.load(path)
            with self.assertRaises(FileExistsError):
                self.ranker.save(path)

    def test_empty_history_cold_ids_and_batch_independence(self):
        candidates = pd.DataFrame({'user_id': [999, 0], 'video_id': [999, 0]})
        features = pd.DataFrame(np.zeros((2, 5), dtype='float32'), columns=self.encoder.features)
        users, history = np.empty(0, dtype='int64'), np.empty((0, 3), dtype='int64')
        scores = self.ranker.predict(candidates, features, users, history)
        single = self.ranker.predict(candidates.iloc[:1], features.iloc[:1], users, history)
        self.assertTrue(np.isfinite(scores).all())
        np.testing.assert_allclose(scores[:1], single, atol=1e-7)
        self.assertTrue(torch.equal(self.ranker.model.behavior.item_embedding.weight[0], torch.zeros(8)))

    def test_legacy_din_checkpoint_still_loads(self):
        din = DINFineRanker(self.encoder, DINConfig(max_length=3, embedding_dim=8, dropout=0))
        data = encode_snapshots(self.encoder, [self.snapshot])
        expected = din.predict_encoded(data)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'old_din.pt'
            din.save(path)
            payload = torch.load(path, weights_only=True)
            payload.pop('model_name')
            torch.save(payload, path)
            np.testing.assert_allclose(DINFineRanker.load(path).predict_encoded(data), expected, atol=1e-7)
            with self.assertRaises(ValueError):
                RankMixerFineRanker.load(path)

    def test_shared_training_saves_correct_model_and_refits_all_windows(self):
        from ranking.din_pipeline import train
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('ranking.din_pipeline.prepare_snapshot', return_value=self.snapshot):
                trained = train(root, root, self.config, 'cpu', RankMixerFineRanker)
                self.assertEqual(trained.metadata['training_rows'], 4 * len(self.snapshot['y']))
                loaded = train(root, root, self.config, 'cpu', RankMixerFineRanker)
                self.assertIsInstance(loaded, RankMixerFineRanker)
                self.assertFalse((root / 'din.pt').exists())


if __name__ == '__main__':
    unittest.main()
