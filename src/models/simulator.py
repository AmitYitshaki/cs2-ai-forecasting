"""Monte Carlo simulation of CS2 series from calibrated map probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Hashable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd


class ProbabilityModel(Protocol):
    """Minimal interface required from a fitted probability estimator."""

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return class probabilities for each feature row."""


@dataclass(frozen=True)
class FrozenEloSnapshot:
    """Canonical Elo state frozen immediately before a series begins."""

    global_ratings: Mapping[Hashable, float]
    map_ratings: Mapping[tuple[Hashable, Hashable], float]
    initial_rating: float = 1500.0

    def global_rating(self, team: Hashable) -> float:
        """Return a team's frozen global rating, or the configured prior."""

        return float(self.global_ratings.get(team, self.initial_rating))

    def map_rating(self, team: Hashable, map_name: Hashable) -> float:
        """Return a team's frozen map rating, or the configured prior."""

        return float(
            self.map_ratings.get((team, map_name), self.initial_rating)
        )


@dataclass(frozen=True)
class SeriesSimulationResult:
    """Aggregated Monte Carlo outcomes from team A's point of view."""

    team_a: Hashable
    team_b: Hashable
    best_of: int
    n_iterations: int
    team_a_win_probability: float
    team_b_win_probability: float
    scoreline_distribution: Mapping[str, float]
    map_sequence: tuple[Hashable, ...]


class SeriesSimulator:
    """Simulate a Bo3 or Bo5 series with live in-memory Elo updates.

    The supplied snapshot is never mutated. Every Monte Carlo iteration starts
    from the same pre-series canonical ratings. After a simulated map, both the
    global Elo and that map's Elo are updated before the next map is predicted.
    The implementation evaluates all active simulations in batches, which is
    mathematically equivalent to an iteration-by-iteration loop but much faster.
    """

    FEATURE_COLUMNS: ClassVar[tuple[str, ...]] = (
        "team1_elo_global",
        "team2_elo_global",
        "team1_elo_map",
        "team2_elo_map",
        "elo_global_diff",
        "elo_map_diff",
    )

    def __init__(
        self,
        calibrated_model: ProbabilityModel,
        snapshot: FrozenEloSnapshot,
        map_sequence: Sequence[Hashable],
        *,
        k_factor: float = 24.0,
        random_state: int | None = 42,
    ) -> None:
        if k_factor <= 0:
            raise ValueError("k_factor must be positive.")
        if not map_sequence:
            raise ValueError("map_sequence cannot be empty.")

        self.calibrated_model = calibrated_model
        self.snapshot = snapshot
        self.map_sequence = tuple(map_sequence)
        self.k_factor = float(k_factor)
        self.random_state = random_state

    def simulate(
        self,
        team_A: Hashable,
        team_B: Hashable,
        best_of: int,
        n_iterations: int = 5000,
    ) -> SeriesSimulationResult:
        """Simulate a series and return win and exact-score probabilities."""

        self._validate_request(team_A, team_B, best_of, n_iterations)
        maps = self.map_sequence[:best_of]
        wins_needed = best_of // 2 + 1
        rng = np.random.default_rng(self.random_state)

        global_a = np.full(
            n_iterations, self.snapshot.global_rating(team_A), dtype=float
        )
        global_b = np.full(
            n_iterations, self.snapshot.global_rating(team_B), dtype=float
        )
        map_a = {
            map_name: np.full(
                n_iterations,
                self.snapshot.map_rating(team_A, map_name),
                dtype=float,
            )
            for map_name in maps
        }
        map_b = {
            map_name: np.full(
                n_iterations,
                self.snapshot.map_rating(team_B, map_name),
                dtype=float,
            )
            for map_name in maps
        }
        wins_a = np.zeros(n_iterations, dtype=np.int8)
        wins_b = np.zeros(n_iterations, dtype=np.int8)
        active = np.ones(n_iterations, dtype=bool)

        for map_name in maps:
            active_indices = np.flatnonzero(active)
            if active_indices.size == 0:
                break

            ratings_a = global_a[active_indices]
            ratings_b = global_b[active_indices]
            map_ratings_a = map_a[map_name][active_indices]
            map_ratings_b = map_b[map_name][active_indices]
            features = pd.DataFrame(
                {
                    "team1_elo_global": ratings_a,
                    "team2_elo_global": ratings_b,
                    "team1_elo_map": map_ratings_a,
                    "team2_elo_map": map_ratings_b,
                    "elo_global_diff": ratings_a - ratings_b,
                    "elo_map_diff": map_ratings_a - map_ratings_b,
                },
                columns=self.FEATURE_COLUMNS,
            )
            probabilities = np.asarray(
                self.calibrated_model.predict_proba(features), dtype=float
            )[:, 1]
            if not np.isfinite(probabilities).all():
                raise ValueError("The calibrated model returned non-finite values.")
            if ((probabilities < 0.0) | (probabilities > 1.0)).any():
                raise ValueError("The calibrated model returned invalid probabilities.")

            team_a_won = rng.random(active_indices.size) < probabilities
            actual = team_a_won.astype(float)
            wins_a[active_indices] += team_a_won.astype(np.int8)
            wins_b[active_indices] += (~team_a_won).astype(np.int8)

            # Elo updates use the Elo expectation, exactly as in PointInTimeEngine.
            global_expected = self.expected_score(ratings_a, ratings_b)
            global_delta = self.k_factor * (actual - global_expected)
            global_a[active_indices] = ratings_a + global_delta
            global_b[active_indices] = ratings_b - global_delta

            map_expected = self.expected_score(map_ratings_a, map_ratings_b)
            map_delta = self.k_factor * (actual - map_expected)
            map_a[map_name][active_indices] = map_ratings_a + map_delta
            map_b[map_name][active_indices] = map_ratings_b - map_delta

            completed = (wins_a >= wins_needed) | (wins_b >= wins_needed)
            active &= ~completed

        if active.any():
            raise AssertionError("Some simulations did not reach a valid scoreline.")

        scoreline_counts: dict[str, int] = {}
        for score_a, score_b in zip(wins_a, wins_b, strict=True):
            scoreline = f"{int(score_a)}-{int(score_b)}"
            scoreline_counts[scoreline] = scoreline_counts.get(scoreline, 0) + 1

        valid_scorelines = self._valid_scorelines(best_of)
        distribution = {
            scoreline: scoreline_counts.get(scoreline, 0) / n_iterations
            for scoreline in valid_scorelines
        }
        if not np.isclose(sum(distribution.values()), 1.0):
            raise AssertionError("Scoreline probabilities do not sum to one.")

        team_a_probability = float((wins_a == wins_needed).mean())
        return SeriesSimulationResult(
            team_a=team_A,
            team_b=team_B,
            best_of=best_of,
            n_iterations=n_iterations,
            team_a_win_probability=team_a_probability,
            team_b_win_probability=1.0 - team_a_probability,
            scoreline_distribution=distribution,
            map_sequence=maps,
        )

    def _validate_request(
        self,
        team_a: Hashable,
        team_b: Hashable,
        best_of: int,
        n_iterations: int,
    ) -> None:
        if team_a == team_b:
            raise ValueError("team_A and team_B must be different teams.")
        if best_of not in (3, 5):
            raise ValueError("best_of must be either 3 or 5.")
        if isinstance(n_iterations, bool) or not isinstance(n_iterations, int):
            raise TypeError("n_iterations must be an integer.")
        if n_iterations <= 0:
            raise ValueError("n_iterations must be positive.")
        if len(self.map_sequence) < best_of:
            raise ValueError(
                f"A Bo{best_of} requires at least {best_of} maps in map_sequence."
            )

    @staticmethod
    def expected_score(
        rating_a: np.ndarray, rating_b: np.ndarray
    ) -> np.ndarray:
        """Return the standard Elo expectation for vectorized ratings."""

        return 1.0 / (1.0 + np.power(10.0, (rating_b - rating_a) / 400.0))

    @staticmethod
    def _valid_scorelines(best_of: int) -> tuple[str, ...]:
        wins_needed = best_of // 2 + 1
        team_a_wins = tuple(f"{wins_needed}-{losses}" for losses in range(wins_needed))
        team_b_wins = tuple(f"{losses}-{wins_needed}" for losses in range(wins_needed))
        return team_a_wins + team_b_wins
