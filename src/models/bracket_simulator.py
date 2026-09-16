"""CS2 bracket simulation with tournament Elo momentum."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from math import sqrt
from typing import Hashable, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .simulator import FrozenEloSnapshot


STAGES = ("QF", "SF", "Final", "Champion")
BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
N_ELO_FEATURES_EXPECTED_ORDER = (
    "team1_elo_global",
    "team2_elo_global",
    "team1_elo_map",
    "team2_elo_map",
    "elo_global_diff",
    "elo_map_diff",
)
N_ELO_FEATURES = len(N_ELO_FEATURES_EXPECTED_ORDER)
TEAM_DISPLAY_NAMES: dict[Hashable, str] = {
    "aurora": "Aurora",
    "betboom team": "BETBOOM",
    "9z team": "9z",
    "furia": "FURIA",
    "g2": "G2",
    "spirit": "Spirit",
    "falcons": "Falcons",
    "vitality": "Vitality",
    "mouz": "MOUZ",
    "nrg": "NRG",
    "magic": "magic",
    "natus vincere": "Natus Vincere",
    "mibr": "MIBR",
}


class FastIsotonicXGBoostPredictor:
    """Numerically equivalent fast path for a prefit isotonic XGBoost model.

    ``CalibratedClassifierCV.predict_proba`` performs substantial validation on
    every call. A tournament makes hundreds of thousands of one-row calls, so
    this adapter uses the fitted XGBoost booster and the learned isotonic knots
    directly. Construction fails closed if the fitted estimator does not have
    the expected single-model binary-isotonic structure.
    """

    def __init__(
        self,
        booster: object,
        x_thresholds: np.ndarray,
        y_thresholds: np.ndarray,
    ) -> None:
        self.booster = booster
        self.x_thresholds = np.asarray(x_thresholds, dtype=float)
        self.y_thresholds = np.asarray(y_thresholds, dtype=float)

    @classmethod
    def from_calibrated_classifier(
        cls, calibrated_model: object
    ) -> "FastIsotonicXGBoostPredictor":
        calibrated_classifiers = getattr(
            calibrated_model, "calibrated_classifiers_", None
        )
        if not calibrated_classifiers or len(calibrated_classifiers) != 1:
            raise TypeError("Expected one prefit calibrated classifier.")
        calibrated_classifier = calibrated_classifiers[0]
        calibrators = getattr(calibrated_classifier, "calibrators", None)
        if not calibrators or len(calibrators) != 1:
            raise TypeError("Expected one binary isotonic calibrator.")

        wrapped_estimator = calibrated_classifier.estimator
        base_estimator = getattr(wrapped_estimator, "estimator", wrapped_estimator)
        if not hasattr(base_estimator, "get_booster"):
            raise TypeError("Expected an XGBoost estimator.")
        calibrator = calibrators[0]
        if not hasattr(calibrator, "X_thresholds_") or not hasattr(
            calibrator, "y_thresholds_"
        ):
            raise TypeError("Expected a fitted isotonic calibrator.")
        booster = base_estimator.get_booster()
        assert booster.feature_names == list(N_ELO_FEATURES_EXPECTED_ORDER), (
            "Unsafe NumPy inference: booster feature order does not match the "
            f"locked order. Expected {list(N_ELO_FEATURES_EXPECTED_ORDER)}, "
            f"received {booster.feature_names}."
        )
        return cls(
            booster,
            calibrator.X_thresholds_,
            calibrator.y_thresholds_,
        )

    def predict_numpy(self, features: np.ndarray) -> np.ndarray:
        """Return calibrated positive-class probabilities for a NumPy matrix."""

        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != N_ELO_FEATURES:
            raise ValueError("Expected a two-dimensional six-feature matrix.")
        raw_probability = np.asarray(
            self.booster.inplace_predict(matrix), dtype=float
        )
        return np.interp(
            raw_probability,
            self.x_thresholds,
            self.y_thresholds,
            left=self.y_thresholds[0],
            right=self.y_thresholds[-1],
        )


@dataclass
class TournamentEloState:
    """Mutable ratings owned by one simulated tournament path."""

    global_ratings: dict[Hashable, float]
    map_ratings: dict[tuple[Hashable, Hashable], float]
    initial_rating: float = 1500.0
    k_factor: float = 24.0

    @classmethod
    def from_snapshot(
        cls,
        snapshot: FrozenEloSnapshot,
        teams: Sequence[Hashable],
        maps: Sequence[Hashable],
        *,
        k_factor: float = 24.0,
    ) -> "TournamentEloState":
        """Create a compact state containing only tournament teams and maps."""

        return cls(
            global_ratings={
                team: snapshot.global_rating(team) for team in teams
            },
            map_ratings={
                (team, map_name): snapshot.map_rating(team, map_name)
                for team in teams
                for map_name in maps
            },
            initial_rating=snapshot.initial_rating,
            k_factor=float(k_factor),
        )

    def global_rating(self, team: Hashable) -> float:
        return float(self.global_ratings.get(team, self.initial_rating))

    def map_rating(self, team: Hashable, map_name: Hashable) -> float:
        return float(
            self.map_ratings.get((team, map_name), self.initial_rating)
        )


@dataclass
class TournamentAnalytics:
    """Lightweight counters collected from completed tournament paths."""

    grand_final_matchups: Counter[tuple[Hashable, Hashable]] = field(
        default_factory=Counter
    )
    runner_ups: Counter[Hashable] = field(default_factory=Counter)
    exact_podiums: Counter[tuple[Hashable, Hashable, Hashable]] = field(
        default_factory=Counter
    )
    grand_final_appearances: Counter[Hashable] = field(default_factory=Counter)

    def merge(self, other: "TournamentAnalytics") -> None:
        """Merge one independently simulated batch into this tracker."""

        self.grand_final_matchups.update(other.grand_final_matchups)
        self.runner_ups.update(other.runner_ups)
        self.exact_podiums.update(other.exact_podiums)
        self.grand_final_appearances.update(other.grand_final_appearances)

    def cinderella_runs(
        self,
        champion_counts: Mapping[Hashable, int],
        n_iterations: int,
        *,
        champion_threshold: float = 0.10,
    ) -> Counter[Hashable]:
        """Return Grand Final appearances by sub-threshold champions."""

        if n_iterations <= 0:
            raise ValueError("n_iterations must be positive.")
        return Counter({
            team: appearances
            for team, appearances in self.grand_final_appearances.items()
            if champion_counts.get(team, 0) / n_iterations < champion_threshold
        })


def simulate_series_once(
    team_A: Hashable,
    team_B: Hashable,
    elo_state: TournamentEloState,
    bestOf: int,
    map_order: Sequence[Hashable],
    *,
    calibrated_model: object,
    rng: np.random.Generator,
) -> tuple[Hashable, TournamentEloState, str]:
    """Simulate one series without mutating the supplied Elo state.

    Randomness is explicit through ``rng``. The returned state is a deep copy
    containing the live global and per-map Elo updates produced by this series.
    """

    if team_A == team_B:
        raise ValueError("A team cannot play itself.")
    if bestOf not in (3, 5):
        raise ValueError("bestOf must be either 3 or 5.")
    if len(map_order) < bestOf:
        raise ValueError(f"A Bo{bestOf} requires {bestOf} ordered maps.")

    updated_state = deepcopy(elo_state)
    wins_needed = bestOf // 2 + 1
    wins_a = 0
    wins_b = 0

    for map_name in tuple(map_order)[:bestOf]:
        rating_a = updated_state.global_rating(team_A)
        rating_b = updated_state.global_rating(team_B)
        map_rating_a = updated_state.map_rating(team_A, map_name)
        map_rating_b = updated_state.map_rating(team_B, map_name)
        feature_values = (
            rating_a,
            rating_b,
            map_rating_a,
            map_rating_b,
            rating_a - rating_b,
            map_rating_a - map_rating_b,
        )
        probability_a = float(
            _predict_positive_numpy(
                calibrated_model,
                np.asarray([feature_values], dtype=np.float32),
            )[0]
        )
        if not np.isfinite(probability_a) or not 0.0 <= probability_a <= 1.0:
            raise ValueError("The calibrated model returned an invalid probability.")

        actual_a = float(rng.random() < probability_a)
        if actual_a == 1.0:
            wins_a += 1
        else:
            wins_b += 1

        global_expected = 1.0 / (
            1.0 + 10.0 ** ((rating_b - rating_a) / 400.0)
        )
        global_delta = updated_state.k_factor * (actual_a - global_expected)
        updated_state.global_ratings[team_A] = rating_a + global_delta
        updated_state.global_ratings[team_B] = rating_b - global_delta

        map_expected = 1.0 / (
            1.0 + 10.0 ** ((map_rating_b - map_rating_a) / 400.0)
        )
        map_delta = updated_state.k_factor * (actual_a - map_expected)
        updated_state.map_ratings[(team_A, map_name)] = map_rating_a + map_delta
        updated_state.map_ratings[(team_B, map_name)] = map_rating_b - map_delta

        if wins_a == wins_needed or wins_b == wins_needed:
            break

    winner = team_A if wins_a == wins_needed else team_B
    return winner, updated_state, f"{wins_a}-{wins_b}"


def simulate_tournament(
    quarterfinals: Sequence[tuple[Hashable, Hashable]],
    starting_elo_state: TournamentEloState,
    calibrated_model: object,
    active_map_pool: Sequence[Hashable],
    *,
    historical_map_orders: Mapping[
        frozenset[Hashable], Sequence[Hashable]
    ] | None = None,
    n_iterations: int = 10_000,
    random_state: int | None = 42,
    batch_size: int = 512,
    show_progress: bool = True,
) -> dict[Hashable, dict[str, int]]:
    """Simulate an eight-team single-elimination bracket.

    Quarter-finals and semi-finals are Bo3; the final is Bo5. The winners of
    QF1/QF2 meet in SF1 and the winners of QF3/QF4 meet in SF2. One Elo state is
    threaded through every match in an iteration, carrying tournament momentum.
    """

    if len(quarterfinals) != 4:
        raise ValueError("Exactly four quarter-finals are required.")
    if isinstance(n_iterations, bool) or not isinstance(n_iterations, int):
        raise TypeError("n_iterations must be an integer.")
    if n_iterations <= 0:
        raise ValueError("n_iterations must be positive.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if len(set(active_map_pool)) < 5:
        raise ValueError("The active map pool must contain at least five maps.")

    teams = [team for pairing in quarterfinals for team in pairing]
    if len(teams) != 8 or len(set(teams)) != 8:
        raise ValueError("Quarter-finals must contain eight unique teams.")

    map_orders = historical_map_orders or {}
    try:
        probability_model: object = (
            FastIsotonicXGBoostPredictor.from_calibrated_classifier(
                calibrated_model
            )
        )
    except TypeError:
        probability_model = calibrated_model
    counts = {
        team: {stage: 0 for stage in STAGES}
        for team in teams
    }
    rng = np.random.default_rng(random_state)
    completed = 0
    with tqdm(
        total=n_iterations,
        desc="Tournament Monte Carlo",
        unit="iter",
        bar_format=BAR_FORMAT,
        disable=not show_progress,
    ) as progress:
        while completed < n_iterations:
            current_batch_size = min(batch_size, n_iterations - completed)
            batch_counts = _simulate_tournament_batch(
                quarterfinals,
                starting_elo_state,
                probability_model,
                active_map_pool,
                map_orders,
                current_batch_size,
                rng,
            )
            for team in teams:
                for stage in STAGES:
                    counts[team][stage] += batch_counts[team][stage]
            completed += current_batch_size
            progress.update(current_batch_size)

    _assert_tournament_probabilities(counts, n_iterations)
    return counts


def simulate_double_elimination_tournament(
    opening_matchups: Sequence[tuple[Hashable, Hashable]],
    starting_elo_state: TournamentEloState,
    calibrated_model: object,
    active_map_pool: Sequence[Hashable],
    *,
    historical_map_orders: Mapping[
        frozenset[Hashable], Sequence[Hashable]
    ] | None = None,
    n_iterations: int = 10_000,
    random_state: int | None = 42,
    batch_size: int = 512,
    grand_final_reset: bool = False,
    track_analytics: bool = False,
    show_progress: bool = True,
) -> (
    dict[Hashable, dict[str, int]]
    | tuple[dict[Hashable, dict[str, int]], TournamentAnalytics]
):
    """Simulate a standard eight-team double-elimination bracket.

    All matches are Bo3 except the Grand Final, which is Bo5. Opening matches
    1/2 and 3/4 feed the two upper semi-finals. Lower-bracket semi-finals use
    cross-pairing to avoid an immediate rematch. ``SF`` in the compact output
    means the final four teams still alive after the lower semi-finals, and
    ``Final`` means a Grand Final appearance. A reset Bo5 can be enabled for
    formats where the lower-bracket entrant must defeat the upper-bracket
    entrant twice; StarLadder's published schedule has one Grand Final, so its
    live configuration leaves this disabled.
    """

    _validate_tournament_inputs(
        opening_matchups, active_map_pool, n_iterations, batch_size
    )
    teams = [team for pairing in opening_matchups for team in pairing]
    map_orders = historical_map_orders or {}
    try:
        probability_model: object = (
            FastIsotonicXGBoostPredictor.from_calibrated_classifier(
                calibrated_model
            )
        )
    except TypeError:
        probability_model = calibrated_model

    counts = {team: {stage: 0 for stage in STAGES} for team in teams}
    analytics = TournamentAnalytics()
    rng = np.random.default_rng(random_state)
    completed = 0
    with tqdm(
        total=n_iterations,
        desc="Double-Elimination Monte Carlo",
        unit="iter",
        bar_format=BAR_FORMAT,
        disable=not show_progress,
    ) as progress:
        while completed < n_iterations:
            current_batch_size = min(batch_size, n_iterations - completed)
            batch_counts, batch_analytics = _simulate_double_elimination_batch(
                opening_matchups,
                starting_elo_state,
                probability_model,
                active_map_pool,
                map_orders,
                current_batch_size,
                rng,
                grand_final_reset=grand_final_reset,
            )
            for team in teams:
                for stage in STAGES:
                    counts[team][stage] += batch_counts[team][stage]
            analytics.merge(batch_analytics)
            completed += current_batch_size
            progress.update(current_batch_size)

    _assert_tournament_probabilities(counts, n_iterations)
    _assert_tournament_analytics(analytics, counts, n_iterations)
    if track_analytics:
        return counts, analytics
    return counts


def _simulate_double_elimination_batch(
    opening_matchups: Sequence[tuple[Hashable, Hashable]],
    starting_elo_state: TournamentEloState,
    probability_model: object,
    active_map_pool: Sequence[Hashable],
    historical_map_orders: Mapping[
        frozenset[Hashable], Sequence[Hashable]
    ],
    n_paths: int,
    rng: np.random.Generator,
    *,
    grand_final_reset: bool,
) -> tuple[dict[Hashable, dict[str, int]], TournamentAnalytics]:
    """Simulate one standard double-elimination batch."""

    teams = [team for pairing in opening_matchups for team in pairing]
    counts = {team: {stage: 0 for stage in STAGES} for team in teams}
    analytics = TournamentAnalytics()
    states = [deepcopy(starting_elo_state) for _ in range(n_paths)]
    for team in teams:
        counts[team]["QF"] = n_paths

    upper_qf_winners: list[np.ndarray] = []
    upper_qf_losers: list[np.ndarray] = []
    for team_a, team_b in opening_matchups:
        team_as = np.full(n_paths, team_a, dtype=object)
        team_bs = np.full(n_paths, team_b, dtype=object)
        winners = _play_matchup_batch(
            team_as, team_bs, states, 3, active_map_pool,
            historical_map_orders, probability_model, rng,
        )
        upper_qf_winners.append(winners)
        upper_qf_losers.append(_opponents_of(winners, team_as, team_bs))

    upper_sf_winners: list[np.ndarray] = []
    upper_sf_losers: list[np.ndarray] = []
    for team_as, team_bs in (
        (upper_qf_winners[0], upper_qf_winners[1]),
        (upper_qf_winners[2], upper_qf_winners[3]),
    ):
        winners = _play_matchup_batch(
            team_as, team_bs, states, 3, active_map_pool,
            historical_map_orders, probability_model, rng,
        )
        upper_sf_winners.append(winners)
        upper_sf_losers.append(_opponents_of(winners, team_as, team_bs))

    lower_r1_winners: list[np.ndarray] = []
    for team_as, team_bs in (
        (upper_qf_losers[0], upper_qf_losers[1]),
        (upper_qf_losers[2], upper_qf_losers[3]),
    ):
        lower_r1_winners.append(
            _play_matchup_batch(
                team_as, team_bs, states, 3, active_map_pool,
                historical_map_orders, probability_model, rng,
            )
        )

    # Cross-pair the lower paths so an opening rematch cannot happen here.
    lower_sf_winners: list[np.ndarray] = []
    for team_as, team_bs in (
        (lower_r1_winners[0], upper_sf_losers[1]),
        (lower_r1_winners[1], upper_sf_losers[0]),
    ):
        lower_sf_winners.append(
            _play_matchup_batch(
                team_as, team_bs, states, 3, active_map_pool,
                historical_map_orders, probability_model, rng,
            )
        )

    # Exactly four teams remain alive at this point.
    for survivors in (*upper_sf_winners, *lower_sf_winners):
        _increment_counts(counts, survivors, "SF")

    upper_final_winner = _play_matchup_batch(
        upper_sf_winners[0], upper_sf_winners[1], states, 3,
        active_map_pool, historical_map_orders, probability_model, rng,
    )
    upper_final_loser = _opponents_of(
        upper_final_winner, upper_sf_winners[0], upper_sf_winners[1]
    )
    lower_final_winner = _play_matchup_batch(
        lower_sf_winners[0], lower_sf_winners[1], states, 3,
        active_map_pool, historical_map_orders, probability_model, rng,
    )
    consolidation_winner = _play_matchup_batch(
        lower_final_winner, upper_final_loser, states, 3,
        active_map_pool, historical_map_orders, probability_model, rng,
    )
    third_places = _opponents_of(
        consolidation_winner, lower_final_winner, upper_final_loser
    )

    _increment_counts(counts, upper_final_winner, "Final")
    _increment_counts(counts, consolidation_winner, "Final")
    champions = _play_matchup_batch(
        upper_final_winner, consolidation_winner, states, 5,
        active_map_pool, historical_map_orders, probability_model, rng,
    )

    if grand_final_reset:
        reset_mask = champions == consolidation_winner
        reset_indices = np.flatnonzero(reset_mask)
        if reset_indices.size:
            reset_states = [states[index] for index in reset_indices]
            reset_winners = _play_matchup_batch(
                upper_final_winner[reset_indices],
                consolidation_winner[reset_indices],
                reset_states,
                5,
                active_map_pool,
                historical_map_orders,
                probability_model,
                rng,
            )
            champions[reset_indices] = reset_winners

    runner_ups = _opponents_of(
        champions, upper_final_winner, consolidation_winner
    )
    analytics.grand_final_matchups.update(
        tuple(sorted((team_a, team_b), key=str))
        for team_a, team_b in zip(
            upper_final_winner, consolidation_winner, strict=True
        )
    )
    analytics.runner_ups.update(runner_ups.tolist())
    analytics.exact_podiums.update(
        zip(champions, runner_ups, third_places, strict=True)
    )
    analytics.grand_final_appearances.update(upper_final_winner.tolist())
    analytics.grand_final_appearances.update(consolidation_winner.tolist())
    _increment_counts(counts, champions, "Champion")
    return counts, analytics


def _play_matchup_batch(
    team_as: np.ndarray,
    team_bs: np.ndarray,
    states: list[TournamentEloState],
    best_of: int,
    active_map_pool: Sequence[Hashable],
    historical_map_orders: Mapping[
        frozenset[Hashable], Sequence[Hashable]
    ],
    probability_model: object,
    rng: np.random.Generator,
) -> np.ndarray:
    orders = _resolve_map_orders(
        team_as, team_bs, best_of, active_map_pool, historical_map_orders, rng
    )
    winners, _ = _simulate_series_batch(
        team_as, team_bs, states, best_of, orders, probability_model, rng
    )
    return winners


def _opponents_of(
    winners: np.ndarray, team_as: np.ndarray, team_bs: np.ndarray
) -> np.ndarray:
    losers = np.where(winners == team_as, team_bs, team_as)
    if np.any(losers == winners):
        raise AssertionError("Winner and loser cannot be the same team.")
    return losers


def _validate_tournament_inputs(
    matchups: Sequence[tuple[Hashable, Hashable]],
    active_map_pool: Sequence[Hashable],
    n_iterations: int,
    batch_size: int,
) -> None:
    if len(matchups) != 4:
        raise ValueError("Exactly four opening matchups are required.")
    if isinstance(n_iterations, bool) or not isinstance(n_iterations, int):
        raise TypeError("n_iterations must be an integer.")
    if n_iterations <= 0:
        raise ValueError("n_iterations must be positive.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if len(set(active_map_pool)) < 5:
        raise ValueError("The active map pool must contain at least five maps.")
    teams = [team for pairing in matchups for team in pairing]
    if len(teams) != 8 or len(set(teams)) != 8:
        raise ValueError("Opening matchups must contain eight unique teams.")


def _simulate_tournament_batch(
    quarterfinals: Sequence[tuple[Hashable, Hashable]],
    starting_elo_state: TournamentEloState,
    probability_model: object,
    active_map_pool: Sequence[Hashable],
    historical_map_orders: Mapping[
        frozenset[Hashable], Sequence[Hashable]
    ],
    n_paths: int,
    rng: np.random.Generator,
) -> dict[Hashable, dict[str, int]]:
    """Simulate one progress batch while preserving per-path Elo dictionaries."""

    teams = [team for pairing in quarterfinals for team in pairing]
    counts = {
        team: {stage: 0 for stage in STAGES}
        for team in teams
    }
    # Each Monte Carlo path receives a fresh, independent deep copy.
    states = [deepcopy(starting_elo_state) for _ in range(n_paths)]
    for team in teams:
        counts[team]["QF"] = n_paths

    quarterfinal_winners: list[np.ndarray] = []
    for team_a, team_b in quarterfinals:
        team_as = np.full(n_paths, team_a, dtype=object)
        team_bs = np.full(n_paths, team_b, dtype=object)
        orders = _resolve_map_orders(
            team_as,
            team_bs,
            3,
            active_map_pool,
            historical_map_orders,
            rng,
        )
        winners, _ = _simulate_series_batch(
            team_as, team_bs, states, 3, orders, probability_model, rng
        )
        quarterfinal_winners.append(winners)
        _increment_counts(counts, winners, "SF")

    semifinal_winners: list[np.ndarray] = []
    for team_as, team_bs in (
        (quarterfinal_winners[0], quarterfinal_winners[1]),
        (quarterfinal_winners[2], quarterfinal_winners[3]),
    ):
        orders = _resolve_map_orders(
            team_as,
            team_bs,
            3,
            active_map_pool,
            historical_map_orders,
            rng,
        )
        winners, _ = _simulate_series_batch(
            team_as, team_bs, states, 3, orders, probability_model, rng
        )
        semifinal_winners.append(winners)
        _increment_counts(counts, winners, "Final")

    final_orders = _resolve_map_orders(
        semifinal_winners[0],
        semifinal_winners[1],
        5,
        active_map_pool,
        historical_map_orders,
        rng,
    )
    champions, _ = _simulate_series_batch(
        semifinal_winners[0],
        semifinal_winners[1],
        states,
        5,
        final_orders,
        probability_model,
        rng,
    )
    _increment_counts(counts, champions, "Champion")
    return counts


def _simulate_series_batch(
    team_as: np.ndarray,
    team_bs: np.ndarray,
    states: list[TournamentEloState],
    best_of: int,
    map_orders: Sequence[Sequence[Hashable]],
    calibrated_model: object,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate independent tournament paths in estimator-efficient batches."""

    n_paths = len(states)
    if not (
        len(team_as) == len(team_bs) == len(map_orders) == n_paths
    ):
        raise ValueError("Batch inputs must have identical lengths.")
    wins_needed = best_of // 2 + 1
    wins_a = np.zeros(n_paths, dtype=np.int8)
    wins_b = np.zeros(n_paths, dtype=np.int8)
    active = np.ones(n_paths, dtype=bool)

    for map_index in range(best_of):
        active_indices = np.flatnonzero(active)
        if active_indices.size == 0:
            break
        matrix = np.empty(
            (active_indices.size, N_ELO_FEATURES), dtype=np.float32
        )
        map_names: list[Hashable] = []
        for row_index, path_index in enumerate(active_indices):
            team_a = team_as[path_index]
            team_b = team_bs[path_index]
            map_name = map_orders[path_index][map_index]
            state = states[path_index]
            rating_a = state.global_rating(team_a)
            rating_b = state.global_rating(team_b)
            map_rating_a = state.map_rating(team_a, map_name)
            map_rating_b = state.map_rating(team_b, map_name)
            matrix[row_index] = (
                rating_a,
                rating_b,
                map_rating_a,
                map_rating_b,
                rating_a - rating_b,
                map_rating_a - map_rating_b,
            )
            map_names.append(map_name)

        probabilities = _predict_positive_numpy(calibrated_model, matrix)
        if not np.isfinite(probabilities).all() or (
            (probabilities < 0.0) | (probabilities > 1.0)
        ).any():
            raise ValueError("The calibrated model returned invalid probabilities.")
        outcomes = rng.random(active_indices.size) < probabilities

        for row_index, path_index in enumerate(active_indices):
            team_a = team_as[path_index]
            team_b = team_bs[path_index]
            map_name = map_names[row_index]
            state = states[path_index]
            rating_a, rating_b, map_rating_a, map_rating_b = matrix[
                row_index, :4
            ]
            actual_a = float(outcomes[row_index])
            wins_a[path_index] += int(outcomes[row_index])
            wins_b[path_index] += int(not outcomes[row_index])

            global_expected = 1.0 / (
                1.0 + 10.0 ** ((rating_b - rating_a) / 400.0)
            )
            global_delta = state.k_factor * (actual_a - global_expected)
            state.global_ratings[team_a] = rating_a + global_delta
            state.global_ratings[team_b] = rating_b - global_delta

            map_expected = 1.0 / (
                1.0 + 10.0 ** ((map_rating_b - map_rating_a) / 400.0)
            )
            map_delta = state.k_factor * (actual_a - map_expected)
            state.map_ratings[(team_a, map_name)] = map_rating_a + map_delta
            state.map_ratings[(team_b, map_name)] = map_rating_b - map_delta

        completed = (wins_a >= wins_needed) | (wins_b >= wins_needed)
        active &= ~completed

    if active.any():
        raise AssertionError("Some batched series did not finish.")
    winners = np.where(wins_a == wins_needed, team_as, team_bs)
    scorelines = np.asarray(
        [f"{score_a}-{score_b}" for score_a, score_b in zip(wins_a, wins_b)],
        dtype=object,
    )
    return winners, scorelines


def _predict_positive_numpy(
    probability_model: object, features: np.ndarray
) -> np.ndarray:
    """Predict positive-class probabilities without constructing Pandas objects."""

    matrix = np.asarray(features, dtype=np.float32)
    direct_predictor = getattr(probability_model, "predict_numpy", None)
    if direct_predictor is not None:
        return np.asarray(direct_predictor(matrix), dtype=float)
    predict_proba = getattr(probability_model, "predict_proba", None)
    if predict_proba is None:
        raise TypeError("Probability model must expose predict_numpy or predict_proba.")
    return np.asarray(predict_proba(matrix), dtype=float)[:, 1]


def _resolve_map_orders(
    team_as: np.ndarray,
    team_bs: np.ndarray,
    best_of: int,
    active_map_pool: Sequence[Hashable],
    historical_map_orders: Mapping[
        frozenset[Hashable], Sequence[Hashable]
    ],
    rng: np.random.Generator,
) -> list[tuple[Hashable, ...]]:
    return [
        _resolve_map_order(
            team_a,
            team_b,
            best_of,
            active_map_pool,
            historical_map_orders,
            rng,
        )
        for team_a, team_b in zip(team_as, team_bs)
    ]


def _increment_counts(
    counts: MutableMapping[Hashable, MutableMapping[str, int]],
    teams: np.ndarray,
    stage: str,
) -> None:
    unique_teams, team_counts = np.unique(teams, return_counts=True)
    for team, count in zip(unique_teams, team_counts):
        counts[team][stage] += int(count)


def print_tournament_summary(
    results_dict: Mapping[Hashable, Mapping[str, int]],
    N_iterations: int,
) -> pd.DataFrame:
    """Print and return a formatted, champion-sorted tournament summary."""

    if N_iterations <= 0:
        raise ValueError("N_iterations must be positive.")

    rows: list[dict[str, str | float]] = []
    for team, counts in results_dict.items():
        champion_probability = counts["Champion"] / N_iterations
        rows.append(
            {
                "Team": TEAM_DISPLAY_NAMES.get(team, str(team)),
                "P(QF)": counts["QF"] / N_iterations,
                "P(SF)": counts["SF"] / N_iterations,
                "P(Final)": counts["Final"] / N_iterations,
                "P(Champion)": champion_probability,
                "SE(Champion)": sqrt(
                    champion_probability
                    * (1.0 - champion_probability)
                    / N_iterations
                ),
            }
        )

    numeric = pd.DataFrame(rows).sort_values(
        "P(Champion)", ascending=False, kind="stable"
    ).reset_index(drop=True)
    formatted = numeric.copy()
    probability_columns = [
        "P(QF)", "P(SF)", "P(Final)", "P(Champion)", "SE(Champion)"
    ]
    for column in probability_columns:
        formatted[column] = numeric[column].map(lambda value: f"{value:.1%}")

    print(formatted.to_string(index=False))
    return formatted


def _resolve_map_order(
    team_a: Hashable,
    team_b: Hashable,
    best_of: int,
    active_map_pool: Sequence[Hashable],
    historical_map_orders: Mapping[
        frozenset[Hashable], Sequence[Hashable]
    ],
    rng: np.random.Generator,
) -> tuple[Hashable, ...]:
    matchup = frozenset((team_a, team_b))
    if matchup in historical_map_orders:
        order = tuple(historical_map_orders[matchup])
        if len(order) < best_of:
            raise ValueError(
                f"Historical map order for {team_a}/{team_b} is incomplete."
            )
        if len(set(order[:best_of])) != best_of:
            raise ValueError("Historical map order contains duplicate maps.")
        return order[:best_of]

    sampled = rng.choice(
        np.asarray(tuple(active_map_pool), dtype=object),
        size=best_of,
        replace=False,
    )
    return tuple(sampled.tolist())


def _assert_tournament_probabilities(
    counts: Mapping[Hashable, Mapping[str, int]],
    n_iterations: int,
) -> None:
    champion_probability_sum = sum(
        team_counts["Champion"] / n_iterations
        for team_counts in counts.values()
    )
    assert np.isclose(champion_probability_sum, 1.0), (
        "Champion probabilities must sum to one."
    )

    for team, team_counts in counts.items():
        probabilities = [team_counts[stage] / n_iterations for stage in STAGES]
        assert all(0.0 <= probability <= 1.0 for probability in probabilities), (
            f"Invalid probability for {team}: {probabilities}"
        )
        assert all(
            left + 1e-12 >= right
            for left, right in zip(probabilities, probabilities[1:])
        ), f"Non-monotonic advancement probabilities for {team}: {probabilities}"

    expected_stage_totals = {
        "QF": 8 * n_iterations,
        "SF": 4 * n_iterations,
        "Final": 2 * n_iterations,
        "Champion": n_iterations,
    }
    for stage, expected_total in expected_stage_totals.items():
        actual_total = sum(team_counts[stage] for team_counts in counts.values())
        assert actual_total == expected_total, (
            f"Unexpected {stage} participant total: {actual_total}"
        )


def _assert_tournament_analytics(
    analytics: TournamentAnalytics,
    counts: Mapping[Hashable, Mapping[str, int]],
    n_iterations: int,
) -> None:
    """Prove that analytical counters reconcile with bracket outcomes."""

    assert sum(analytics.grand_final_matchups.values()) == n_iterations
    assert sum(analytics.runner_ups.values()) == n_iterations
    assert sum(analytics.exact_podiums.values()) == n_iterations
    assert sum(analytics.grand_final_appearances.values()) == 2 * n_iterations
    for team, team_counts in counts.items():
        assert analytics.grand_final_appearances[team] == team_counts["Final"], (
            f"Grand Final appearances do not reconcile for {team}."
        )
        assert analytics.runner_ups[team] + team_counts["Champion"] == (
            team_counts["Final"]
        ), f"Champion and runner-up counts do not reconcile for {team}."
