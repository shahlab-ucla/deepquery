from __future__ import annotations

import unittest

from wormctx.poc.contracts import DatasetSplit, EpisodeFamily
from wormctx.poc.simulation import generate_episode


class TensorizeTests(unittest.TestCase):
    @unittest.skipUnless(__import__("importlib").util.find_spec("torch"), "torch not installed")
    def test_episode_tensor_shapes(self) -> None:
        from wormctx.poc.tensorize import VOCABULARY, tensorize_episode

        episode = generate_episode(
            EpisodeFamily.natural_variation,
            group_index=3,
            seed=19,
            split=DatasetSplit.test,
        )
        item = tensorize_episode(
            episode,
            max_nodes=64,
            max_edges=192,
            numeric_feature_dim=16,
            max_hypotheses=4,
            max_experiments=4,
        )
        self.assertEqual(tuple(item["numeric_features"].shape), (64, 16))
        self.assertEqual(int(item["node_mask"].sum()), len(episode.graph.nodes))
        self.assertEqual(int(item["edge_mask"].sum()), len(episode.graph.edges))
        self.assertEqual(len(VOCABULARY["operators"]), 10)


if __name__ == "__main__":
    unittest.main()
