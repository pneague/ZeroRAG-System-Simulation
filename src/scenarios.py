"""Single source of truth for scenario classification sets.

Both the semantic-DHT entrypoint (``experiments/run_simulation.py``) and the
LSH-DHT entrypoint (``experiments/run_lsh_simulation.py``) need to classify
scenarios into the same families:

* ``SEEDED_SCENARIOS`` -- peers start with a precomputed inventory.
* ``RUNTIME_SEARCHABLE_SCENARIOS`` -- searchability is recomputed at runtime
  from current ownership instead of being read from a sidecar.
* ``FIFO_SCENARIOS`` -- peers cap their owned-doc set with FIFO eviction.
* ``RANDOMIZED_TIMELINE_SCENARIOS`` -- consume the randomized metadata
  artifact instead of the original chronological one.
* ``AMAZON_SUPPORTED_SCENARIOS`` / ``SYNTHETIC_SUPPORTED_SCENARIOS`` /
  ``PUBMEDQA_SUPPORTED_SCENARIOS`` -- the subset of scenarios that make sense
  on datasets with no real timestamps.

Keeping these in one place avoids drift between the two algorithms. The
existing ``src/network.py`` and ``src/amazon_dataset.py`` re-export them for
backward compatibility with already-imported call sites.
"""

from __future__ import annotations


SEEDED_SCENARIOS: frozenset[str] = frozenset(
    {
        "seeded_fixed_inventory",
        "seeded_randomized_timeline",
    }
)


RUNTIME_SEARCHABLE_SCENARIOS: frozenset[str] = frozenset(
    {
        "fifo_chronological",
        "fifo_randomized_timeline",
        "empty_start_randomized_timeline",
    }
)


FIFO_SCENARIOS: frozenset[str] = frozenset(
    {
        "fifo_chronological",
        "fifo_randomized_timeline",
    }
)


RANDOMIZED_TIMELINE_SCENARIOS: frozenset[str] = frozenset(
    {
        "seeded_randomized_timeline",
        "fifo_randomized_timeline",
        "empty_start_randomized_timeline",
    }
)


AMAZON_SUPPORTED_SCENARIOS: frozenset[str] = frozenset(
    {
        "seeded_fixed_inventory",
        "seeded_randomized_timeline",
        "empty_start_randomized_timeline",
        "fifo_randomized_timeline",
    }
)


SYNTHETIC_SUPPORTED_SCENARIOS: frozenset[str] = frozenset(
    {
        "seeded_fixed_inventory",
        "seeded_randomized_timeline",
        "empty_start_randomized_timeline",
        "fifo_randomized_timeline",
    }
)


PUBMEDQA_SUPPORTED_SCENARIOS: frozenset[str] = SYNTHETIC_SUPPORTED_SCENARIOS


ALL_SCENARIOS: tuple[str, ...] = (
    "empty_start_chronological",
    "fifo_chronological",
    "empty_start_randomized_timeline",
    "fifo_randomized_timeline",
    "seeded_fixed_inventory",
    "seeded_randomized_timeline",
)


DEFAULT_SCENARIO = "empty_start_chronological"
