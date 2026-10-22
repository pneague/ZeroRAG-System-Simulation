from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.peer import PeerResponse, Referral


@dataclass(slots=True)
class SearchResult:
    cycles_used: int
    result_symbol: str
    termination_reason: str
    queried_peer_ids: list[int]
    successful_peer_id: int | None
    best_known_peer_id: int | None
    retrieved_relevant_doc_ids: tuple[str, ...] = ()
    contributing_peer_ids: tuple[int, ...] = ()
    requests_until_first_relevant: int | None = None
    best_contributing_peer_id: int | None = None


def _best_contributor(
    current_best_peer_id: int | None,
    current_best_count: int,
    peer_id: int,
    relevant_doc_count: int,
) -> tuple[int | None, int]:
    if relevant_doc_count > current_best_count:
        return peer_id, relevant_doc_count
    return current_best_peer_id, current_best_count


def _merge_candidate(candidate_list: dict[int, Referral], referral: Referral) -> None:
    current = candidate_list.get(referral.peer_id)
    if current is None or referral.similarity > current.similarity:
        candidate_list[referral.peer_id] = referral


def _similarity_select(unqueried: list[Referral]) -> Referral:
    return max(unqueried, key=lambda item: item.similarity)


def _similarity_best_known(candidate_list: dict[int, Referral]) -> int | None:
    if not candidate_list:
        return None
    return max(candidate_list.values(), key=lambda item: item.similarity).peer_id


def _rank_referrals(referrals: list[Referral]) -> list[Referral]:
    return sorted(referrals, key=lambda item: (-item.similarity, item.peer_id))


def run_best_first_search(
    *,
    querier_peer_id: int,
    target_doc_id: str,
    max_cycles: int,
    query_peer: Callable[[int], PeerResponse],
    select_next: Callable[[list[Referral]], Referral] = _similarity_select,
    best_known: Callable[[dict[int, Referral]], int | None] = _similarity_best_known,
) -> SearchResult:
    candidate_list: dict[int, Referral] = {}
    queried_peer_ids: list[int] = []
    queried_set: set[int] = set()
    successful_peer_id: int | None = None

    current_peer_id = querier_peer_id
    while len(queried_peer_ids) < max_cycles:
        response = query_peer(current_peer_id)
        queried_peer_ids.append(current_peer_id)
        queried_set.add(current_peer_id)

        for referral in response.referrals:
            _merge_candidate(candidate_list, referral)

        if response.found and response.found_doc_id == target_doc_id:
            successful_peer_id = current_peer_id
            return SearchResult(
                cycles_used=len(queried_peer_ids),
                result_symbol="+",
                termination_reason="found",
                queried_peer_ids=queried_peer_ids,
                successful_peer_id=successful_peer_id,
                best_known_peer_id=successful_peer_id,
            )

        unqueried = [
            referral for peer_id, referral in candidate_list.items() if peer_id not in queried_set
        ]
        if not unqueried:
            return SearchResult(
                cycles_used=len(queried_peer_ids),
                result_symbol="-",
                termination_reason="exhausted",
                queried_peer_ids=queried_peer_ids,
                successful_peer_id=None,
                best_known_peer_id=best_known(candidate_list),
            )
        current_peer_id = select_next(unqueried).peer_id

    return SearchResult(
        cycles_used=len(queried_peer_ids),
        result_symbol="-",
        termination_reason="cap",
        queried_peer_ids=queried_peer_ids,
        successful_peer_id=None,
        best_known_peer_id=best_known(candidate_list),
    )


def run_best_first_multi_relevant_search(
    *,
    querier_peer_id: int,
    max_cycles: int,
    query_peer: Callable[[int], PeerResponse],
    select_next: Callable[[list[Referral]], Referral] = _similarity_select,
    best_known: Callable[[dict[int, Referral]], int | None] = _similarity_best_known,
) -> SearchResult:
    candidate_list: dict[int, Referral] = {}
    queried_peer_ids: list[int] = []
    queried_set: set[int] = set()
    retrieved: set[str] = set()
    contributing: set[int] = set()
    requests_until_first: int | None = None
    best_contributing_peer_id: int | None = None
    best_contributing_count = 0

    current_peer_id = querier_peer_id
    while len(queried_peer_ids) < max_cycles:
        response = query_peer(current_peer_id)
        queried_peer_ids.append(current_peer_id)
        queried_set.add(current_peer_id)

        for referral in response.referrals:
            _merge_candidate(candidate_list, referral)

        if response.found_relevant_doc_ids:
            if requests_until_first is None:
                requests_until_first = len(queried_peer_ids)
            retrieved.update(response.found_relevant_doc_ids)
            contributing.add(current_peer_id)
            best_contributing_peer_id, best_contributing_count = _best_contributor(
                best_contributing_peer_id,
                best_contributing_count,
                current_peer_id,
                len(response.found_relevant_doc_ids),
            )

        unqueried = [
            referral for peer_id, referral in candidate_list.items() if peer_id not in queried_set
        ]
        if not unqueried:
            return SearchResult(
                cycles_used=len(queried_peer_ids),
                result_symbol="+" if retrieved else "-",
                termination_reason="exhausted",
                queried_peer_ids=queried_peer_ids,
                successful_peer_id=None,
                best_known_peer_id=best_known(candidate_list),
                retrieved_relevant_doc_ids=tuple(sorted(retrieved)),
                contributing_peer_ids=tuple(sorted(contributing)),
                requests_until_first_relevant=requests_until_first,
                best_contributing_peer_id=best_contributing_peer_id,
            )
        current_peer_id = select_next(unqueried).peer_id

    return SearchResult(
        cycles_used=len(queried_peer_ids),
        result_symbol="+" if retrieved else "-",
        termination_reason="cap",
        queried_peer_ids=queried_peer_ids,
        successful_peer_id=None,
        best_known_peer_id=best_known(candidate_list),
        retrieved_relevant_doc_ids=tuple(sorted(retrieved)),
        contributing_peer_ids=tuple(sorted(contributing)),
        requests_until_first_relevant=requests_until_first,
        best_contributing_peer_id=best_contributing_peer_id,
    )


def run_fanout_search(
    *,
    querier_peer_id: int,
    target_doc_id: str,
    max_cycles: int,
    fanout: int,
    query_peer: Callable[[int], PeerResponse],
    best_known: Callable[[dict[int, Referral]], int | None] = _similarity_best_known,
) -> SearchResult:
    candidate_list: dict[int, Referral] = {}
    queried_peer_ids: list[int] = []
    queried_set: set[int] = set()
    current_round_peer_ids = [querier_peer_id]
    rounds_used = -1

    while rounds_used < max_cycles and current_round_peer_ids:
        rounds_used += 1
        found_peer_id: int | None = None

        for current_peer_id in current_round_peer_ids:
            response = query_peer(current_peer_id)
            queried_peer_ids.append(current_peer_id)
            queried_set.add(current_peer_id)

            for referral in response.referrals:
                _merge_candidate(candidate_list, referral)

            if (
                found_peer_id is None
                and response.found
                and response.found_doc_id == target_doc_id
            ):
                found_peer_id = current_peer_id

        if found_peer_id is not None:
            return SearchResult(
                cycles_used=rounds_used,
                result_symbol="+",
                termination_reason="found",
                queried_peer_ids=queried_peer_ids,
                successful_peer_id=found_peer_id,
                best_known_peer_id=found_peer_id,
            )

        unqueried = [
            referral for peer_id, referral in candidate_list.items() if peer_id not in queried_set
        ]
        if not unqueried:
            return SearchResult(
                cycles_used=rounds_used,
                result_symbol="-",
                termination_reason="exhausted",
                queried_peer_ids=queried_peer_ids,
                successful_peer_id=None,
                best_known_peer_id=best_known(candidate_list),
            )

        current_round_peer_ids = [
            referral.peer_id for referral in _rank_referrals(unqueried)[: max(fanout, 1)]
        ]

    return SearchResult(
        cycles_used=rounds_used,
        result_symbol="-",
        termination_reason="cap",
        queried_peer_ids=queried_peer_ids,
        successful_peer_id=None,
        best_known_peer_id=best_known(candidate_list),
    )


def run_fanout_multi_relevant_search(
    *,
    querier_peer_id: int,
    max_cycles: int,
    fanout: int,
    query_peer: Callable[[int], PeerResponse],
    best_known: Callable[[dict[int, Referral]], int | None] = _similarity_best_known,
) -> SearchResult:
    candidate_list: dict[int, Referral] = {}
    queried_peer_ids: list[int] = []
    queried_set: set[int] = set()
    current_round_peer_ids = [querier_peer_id]
    rounds_used = 0
    retrieved: set[str] = set()
    contributing: set[int] = set()
    requests_until_first: int | None = None
    best_contributing_peer_id: int | None = None
    best_contributing_count = 0

    while rounds_used < max_cycles and current_round_peer_ids:
        rounds_used += 1

        for current_peer_id in current_round_peer_ids:
            response = query_peer(current_peer_id)
            queried_peer_ids.append(current_peer_id)
            queried_set.add(current_peer_id)

            for referral in response.referrals:
                _merge_candidate(candidate_list, referral)

            if response.found_relevant_doc_ids:
                if requests_until_first is None:
                    requests_until_first = len(queried_peer_ids)
                retrieved.update(response.found_relevant_doc_ids)
                contributing.add(current_peer_id)
                best_contributing_peer_id, best_contributing_count = _best_contributor(
                    best_contributing_peer_id,
                    best_contributing_count,
                    current_peer_id,
                    len(response.found_relevant_doc_ids),
                )

        unqueried = [
            referral for peer_id, referral in candidate_list.items() if peer_id not in queried_set
        ]
        if not unqueried:
            return SearchResult(
                cycles_used=rounds_used,
                result_symbol="+" if retrieved else "-",
                termination_reason="exhausted",
                queried_peer_ids=queried_peer_ids,
                successful_peer_id=None,
                best_known_peer_id=best_known(candidate_list),
                retrieved_relevant_doc_ids=tuple(sorted(retrieved)),
                contributing_peer_ids=tuple(sorted(contributing)),
                requests_until_first_relevant=requests_until_first,
                best_contributing_peer_id=best_contributing_peer_id,
            )

        current_round_peer_ids = [
            referral.peer_id for referral in _rank_referrals(unqueried)[: max(fanout, 1)]
        ]

    return SearchResult(
        cycles_used=rounds_used,
        result_symbol="+" if retrieved else "-",
        termination_reason="cap",
        queried_peer_ids=queried_peer_ids,
        successful_peer_id=None,
        best_known_peer_id=best_known(candidate_list),
        retrieved_relevant_doc_ids=tuple(sorted(retrieved)),
        contributing_peer_ids=tuple(sorted(contributing)),
        requests_until_first_relevant=requests_until_first,
        best_contributing_peer_id=best_contributing_peer_id,
    )
