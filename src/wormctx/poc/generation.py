"""Config-driven episode generation without a tensor-framework dependency."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from .config import PocConfig
from .contracts import BiologicalEpisode, DatasetSplit, EpisodeFamily, InvalidDesignFlag
from .simulation import generate_episode


def _red_team_flags(
    family: EpisodeFamily,
    episode_index: int,
) -> tuple[InvalidDesignFlag, ...]:
    if family is EpisodeFamily.developmental:
        return (
            (InvalidDesignFlag.database_absence_as_powered_negative,)
            if episode_index == 0
            else ()
        )
    if family is EpisodeFamily.cross_context:
        return (
            (InvalidDesignFlag.transport_without_identification,)
            if episode_index == 0
            else ()
        )
    natural_flags = (
        InvalidDesignFlag.dominance_from_homozygous_panel,
        InvalidDesignFlag.absent_locus_as_snp,
        InvalidDesignFlag.post_treatment_covariate,
    )
    return (
        (natural_flags[episode_index],)
        if episode_index < len(natural_flags)
        else ()
    )


def generate_configured_episodes(config: PocConfig) -> list[BiologicalEpisode]:
    """Generate exact group-disjoint split counts with all red-team flags represented."""
    split_counts = {
        DatasetSplit.train: config.simulation.train_groups_per_family,
        DatasetSplit.val: config.simulation.validation_groups_per_family,
        DatasetSplit.test: config.simulation.test_groups_per_family,
    }
    offsets: dict[DatasetSplit, int] = {}
    running = 0
    for split, count in split_counts.items():
        offsets[split] = running
        running += count

    episodes = []
    for family in EpisodeFamily:
        family_offset = list(EpisodeFamily).index(family) * 100_000
        for split, group_count in split_counts.items():
            for local_group in range(group_count):
                global_group = family_offset + offsets[split] + local_group
                split_group = f"sim:{family.value}:{split.value}:group-{local_group:04d}"
                for episode_index in range(config.simulation.episodes_per_group):
                    flags = _red_team_flags(family, episode_index)
                    episodes.append(
                        generate_episode(
                            family,
                            group_index=global_group,
                            episode_index=episode_index,
                            seed=config.simulation.seed,
                            split=split,
                            split_group=split_group,
                            invalid_flags=flags,
                        )
                    )
    return episodes


def split_episodes(
    episodes: Iterable[BiologicalEpisode],
) -> dict[str, list[BiologicalEpisode]]:
    result = {"train": [], "val": [], "test": []}
    group_to_split: dict[str, str] = {}
    for episode in episodes:
        split = episode.split.value
        previous = group_to_split.setdefault(episode.split_group, split)
        if previous != split:
            raise ValueError(
                f"split group {episode.split_group!r} occurs in both {previous!r} and {split!r}"
            )
        result[split].append(episode)
    for split, values in result.items():
        if not values:
            raise ValueError(f"generated dataset has an empty {split!r} split")
    groups = {
        split: {item.split_group for item in values} for split, values in result.items()
    }
    if (
        groups["train"] & groups["val"]
        or groups["train"] & groups["test"]
        or groups["val"] & groups["test"]
    ):
        raise ValueError("split groups overlap")
    return result


def dataset_summary(episodes: Iterable[BiologicalEpisode]) -> dict[str, Any]:
    values = list(episodes)
    if not values:
        raise ValueError("cannot summarize an empty episode collection")
    return {
        "episode_count": len(values),
        "simulation_only": all(item.provenance.synthetic for item in values),
        "family_counts": dict(sorted(Counter(item.family.value for item in values).items())),
        "split_counts": dict(sorted(Counter(item.split.value for item in values).items())),
        "split_group_counts": {
            split: len({item.split_group for item in values if item.split.value == split})
            for split in ("train", "val", "test")
        },
        "invalid_flag_counts": dict(
            sorted(
                Counter(
                    flag.value
                    for item in values
                    for flag in item.labels.invalid_design_flags
                ).items()
            )
        ),
        "node_count_range": [
            min(len(item.graph.nodes) for item in values),
            max(len(item.graph.nodes) for item in values),
        ],
        "edge_count_range": [
            min(len(item.graph.edges) for item in values),
            max(len(item.graph.edges) for item in values),
        ],
    }


__all__ = ["dataset_summary", "generate_configured_episodes", "split_episodes"]
