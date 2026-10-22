from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.peer import PeerResponse, Referral
from src.routing import (
    run_best_first_multi_relevant_search,
    run_fanout_multi_relevant_search,
    run_fanout_search,
)


def _response(
    peer_id: int,
    *,
    found: bool = False,
    referrals: list[Referral] | None = None,
    found_relevant_doc_ids: tuple[str, ...] = (),
) -> PeerResponse:
    return PeerResponse(
        queried_peer_id=peer_id,
        found=found,
        found_doc_id="target" if found else None,
        referrals=referrals or [],
        found_relevant_doc_ids=found_relevant_doc_ids,
    )


def _ref(peer_id: int, similarity: float) -> Referral:
    return Referral(
        peer_id=peer_id,
        closest_doc_id=f"doc_{peer_id}",
        similarity=similarity,
        source="test",
    )


class FanoutSearchTests(unittest.TestCase):
    def test_multi_relevant_best_first_continues_after_first_hit(self) -> None:
        responses = {
            1: _response(1, referrals=[_ref(2, 0.9)]),
            2: _response(2, found_relevant_doc_ids=("d2",), referrals=[_ref(3, 0.8)]),
            3: _response(3, found_relevant_doc_ids=("d3",)),
        }

        result = run_best_first_multi_relevant_search(
            querier_peer_id=1,
            max_cycles=10,
            query_peer=responses.__getitem__,
        )

        self.assertEqual(result.queried_peer_ids, [1, 2, 3])
        self.assertEqual(result.result_symbol, "+")
        self.assertEqual(result.retrieved_relevant_doc_ids, ("d2", "d3"))
        self.assertEqual(result.best_contributing_peer_id, 2)
        self.assertEqual(result.requests_until_first_relevant, 2)
        self.assertEqual(result.termination_reason, "exhausted")

    def test_multi_relevant_best_first_tracks_largest_contributor(self) -> None:
        responses = {
            1: _response(1, referrals=[_ref(2, 0.9), _ref(3, 0.8)]),
            2: _response(2, found_relevant_doc_ids=("d2",)),
            3: _response(3, found_relevant_doc_ids=("d3", "d4")),
        }

        result = run_best_first_multi_relevant_search(
            querier_peer_id=1,
            max_cycles=10,
            query_peer=responses.__getitem__,
        )

        self.assertEqual(result.contributing_peer_ids, (2, 3))
        self.assertEqual(result.best_contributing_peer_id, 3)

    def test_multi_relevant_fanout_tracks_largest_contributor(self) -> None:
        responses = {
            1: _response(1, referrals=[_ref(2, 0.9), _ref(3, 0.8)]),
            2: _response(2, found_relevant_doc_ids=("d2",)),
            3: _response(3, found_relevant_doc_ids=("d3", "d4", "d5")),
        }

        result = run_fanout_multi_relevant_search(
            querier_peer_id=1,
            max_cycles=10,
            fanout=2,
            query_peer=responses.__getitem__,
        )

        self.assertEqual(result.contributing_peer_ids, (2, 3))
        self.assertEqual(result.best_contributing_peer_id, 3)

    def test_fanout_queries_top_candidates_by_similarity_then_peer_id(self) -> None:
        responses = {
            1: _response(1, referrals=[_ref(3, 0.7), _ref(2, 0.9), _ref(4, 0.8)]),
            2: _response(2, referrals=[_ref(5, 0.95)]),
            4: _response(4, referrals=[_ref(6, 0.6)]),
        }

        result = run_fanout_search(
            querier_peer_id=1,
            target_doc_id="target",
            max_cycles=1,
            fanout=2,
            query_peer=lambda peer_id: responses.get(peer_id, _response(peer_id)),
        )

        self.assertEqual(result.queried_peer_ids, [1, 2, 4])
        self.assertEqual(result.cycles_used, 1)
        self.assertEqual(result.termination_reason, "cap")
        self.assertEqual(result.best_known_peer_id, 5)

    def test_fanout_tie_breaks_by_peer_id_within_round(self) -> None:
        responses = {
            1: _response(1, referrals=[_ref(3, 0.9), _ref(2, 0.9)]),
            2: _response(2, found=True),
            3: _response(3, found=True),
        }

        result = run_fanout_search(
            querier_peer_id=1,
            target_doc_id="target",
            max_cycles=10,
            fanout=2,
            query_peer=responses.__getitem__,
        )

        self.assertEqual(result.queried_peer_ids, [1, 2, 3])
        self.assertEqual(result.cycles_used, 1)
        self.assertEqual(result.successful_peer_id, 2)
        self.assertEqual(result.termination_reason, "found")

    def test_fanout_deduplicates_candidates_and_keeps_best_similarity(self) -> None:
        responses = {
            1: _response(1, referrals=[_ref(2, 0.9), _ref(3, 0.8)]),
            2: _response(2, referrals=[_ref(4, 0.4)]),
            3: _response(3, referrals=[_ref(4, 0.95)]),
            4: _response(4, found=True),
        }

        result = run_fanout_search(
            querier_peer_id=1,
            target_doc_id="target",
            max_cycles=10,
            fanout=2,
            query_peer=responses.__getitem__,
        )

        self.assertEqual(result.queried_peer_ids, [1, 2, 3, 4])
        self.assertEqual(result.cycles_used, 2)
        self.assertEqual(result.result_symbol, "+")
        self.assertEqual(result.successful_peer_id, 4)

    def test_fanout_exhausts_when_no_candidates_remain(self) -> None:
        result = run_fanout_search(
            querier_peer_id=1,
            target_doc_id="target",
            max_cycles=10,
            fanout=10,
            query_peer=lambda peer_id: _response(peer_id),
        )

        self.assertEqual(result.queried_peer_ids, [1])
        self.assertEqual(result.cycles_used, 0)
        self.assertEqual(result.termination_reason, "exhausted")
        self.assertIsNone(result.best_known_peer_id)


if __name__ == "__main__":
    unittest.main()
