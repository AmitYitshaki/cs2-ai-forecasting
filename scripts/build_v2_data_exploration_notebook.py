"""Build the bilingual Version 2.0 player/team feature exploration notebook."""

from __future__ import annotations

from pathlib import Path
import re
import unicodedata

import nbformat as nbf
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "2.0_v2_data_exploration.ipynb"


def normalize_text(value: object) -> str:
    """Return the exact normalized token used by the V2 identity pipeline."""

    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def normalize_team(value: object) -> str:
    """Normalize a team label exactly as the V2 training pipeline does."""

    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().casefold()
    text = re.sub(r"\b(team|esports|gaming|club|cs2)\b", " ", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def normalize_player(value: object) -> str:
    """Normalize a player label exactly as the V2 training pipeline does."""

    if pd.isna(value):
        return ""
    text = str(value).split(" (", 1)[0].split(",", 1)[0]
    return normalize_text(text)


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    markdown(
        """
# Version 2.0 — Point-in-Time Player Analytics and Rating Decay

This notebook implements the first Version 2.0 data pipeline. It enriches the full-coverage map box scores with the coverage-capped rich player table, builds player Elo without leaking the current map into its own features, applies explicit inactivity decay to team and player ratings, and preserves the locked calendar split.
"""
    ),
    markdown(
        """
# גרסה 2.0 — אנליטיקת שחקנים בזמן אמת ודעיכת דירוג

מחברת זו מממשת את צינור הנתונים הראשון של גרסה 2.0. היא מעשירה את נתוני המפות המלאים באמצעות טבלת השחקנים העשירה אך המוגבלת בזמן, בונה Elo לשחקנים בלי להדליף את תוצאת המפה הנוכחית אל הפיצ'רים שלה, מפעילה דעיכה מפורשת לאחר חוסר פעילות, ושומרת על החלוקה הכרונולוגית הנעולה.
"""
    ),
    markdown(
        """
## Reproducibility contract

- `final_tournament_features.csv` is the full-timeline source of map results and basic player statistics.
- `cs2_newestcombinedmatches.csv` is enrichment only; it ends in October 2025 and is never required for a row to survive.
- Every feature row is emitted before the current map updates team or player state.
- Current-map ADR, KAST, K/D difference, Rating, KPR, and DPR are update signals, not model inputs for that same map.
- The half-lives below are explicit exploratory defaults. They must later be selected on Validation only.
"""
    ),
    markdown(
        """
## חוזה שחזור ובטיחות

- הקובץ `final_tournament_features.csv` הוא מקור האמת לכל ציר הזמן של תוצאות מפות וסטטיסטיקות שחקנים בסיסיות.
- הקובץ `cs2_newestcombinedmatches.csv` משמש להעשרה בלבד; הוא מסתיים באוקטובר 2025 ולעולם אינו תנאי להשארת שורה.
- כל שורת פיצ'רים נפלטת לפני שהמפה הנוכחית מעדכנת את מצב הקבוצה או השחקנים.
- ADR, KAST, הפרש הריגות/מיתות, Rating, KPR ו־DPR של המפה הנוכחית הם אותות עדכון, ולא פיצ'רים לחיזוי אותה מפה.
- זמני מחצית החיים המוגדרים כאן הם ברירות מחדל ניסיוניות ושקופות. בהמשך מותר לבחור אותם רק על Validation.
"""
    ),
    code(
        r'''
from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 180)

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "data").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

PRIMARY_PATH = PROJECT_ROOT / "data" / "final_tournament_features.csv"
RICH_PATH = PROJECT_ROOT / "data" / "cs2_newestcombinedmatches.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "v2_player_team_features.parquet"
PLAYER_EVENTS_PATH = PROJECT_ROOT / "data" / "v2_player_elo_events.parquet"

MU = 1500.0
K_TEAM = 24.0
K_PERF = 0.0
TEAM_HALF_LIFE_DAYS = 180.0
PLAYER_HALF_LIFE_DAYS = 1095.0
RICH_IMPACT_WEIGHT = 1.0
RICH_MATCH_TOLERANCE_HOURS = 10.0

TRAIN_END = pd.Timestamp("2025-12-31 23:59:59")
VAL_END = pd.Timestamp("2026-03-31 23:59:59")

print(f"Project root: {PROJECT_ROOT}")
print(f"Team/player prior: {MU:.0f}; K_team={K_TEAM:.0f}; K_perf={K_PERF:.0f}")
print(f"Half-lives: team={TEAM_HALF_LIFE_DAYS:.0f} days, player={PLAYER_HALF_LIFE_DAYS:.0f} days")
'''
    ),
    markdown(
        """
## 1. Load and lock valid map rows

The phantom-row rule is applied before any history is constructed: a row must have a map name, must not be a series-total row unless it is a genuine Bo1, and must have a positive played score. The deterministic sort keys are timestamp, match ID, and game ID.
"""
    ),
    markdown(
        """
## 1. טעינה ונעילת שורות מפה תקינות

כלל סינון שורות הרפאים מופעל לפני בניית היסטוריה כלשהי: לשורה חייב להיות שם מפה, אסור לה להיות שורת סיכום סדרה אלא אם מדובר ב־Bo1 אמיתי, וסכום התוצאה חייב להיות חיובי. מפתחות המיון הדטרמיניסטיים הם חותמת זמן, מזהה משחק ומזהה מפה.
"""
    ),
    code(
        r'''
primary_raw = pd.read_csv(PRIMARY_PATH, low_memory=False)
rich_raw = pd.read_csv(RICH_PATH, low_memory=False)

primary_raw["datetime"] = pd.to_datetime(primary_raw["datetime"], errors="raise")
is_total = primary_raw["is_total"].fillna(False).astype(bool)
best_of_one = pd.to_numeric(primary_raw["bestOf"], errors="coerce").eq(1)
score_sum = (
    pd.to_numeric(primary_raw["score1_game"], errors="coerce")
    + pd.to_numeric(primary_raw["score2_game"], errors="coerce")
)
valid_mask = primary_raw["map_name"].notna() & (~is_total | best_of_one) & score_sum.gt(0)

maps = (
    primary_raw.loc[valid_mask]
    .sort_values(["datetime", "match_id", "game_id"], kind="stable")
    .reset_index(drop=True)
)
maps["_row_id"] = np.arange(len(maps), dtype=np.int64)

assert len(maps) == 6_700, f"Expected 6,700 valid maps, found {len(maps):,}."
assert maps["datetime"].is_monotonic_increasing
assert not maps.duplicated(["match_id", "game_id"]).any()
assert not (maps["is_total"].fillna(False).astype(bool) & maps["bestOf"].ne(1)).any()
assert (maps["score1_game"] + maps["score2_game"]).gt(0).all()

print(f"Raw primary rows: {len(primary_raw):,}")
print(f"Valid map rows: {len(maps):,}")
print(f"Rich-source rows: {len(rich_raw):,}")
print(f"Primary coverage: {maps['datetime'].min()} → {maps['datetime'].max()}")
print(f"Rich coverage: {rich_raw['date'].min()} → {rich_raw['date'].max()}")
'''
    ),
    markdown(
        """
## 2. Conservative enrichment linkage

The two files do not share a reliable match identifier and their timestamps use different clock conventions. We therefore link unique matches only when the normalized unordered team pair agrees and the timestamps are within ten hours. Candidate links are greedily assigned by the smallest time gap, one-to-one. Player enrichment then requires an exact normalized player-name match inside that linked match. Uncertain rows deliberately fall back to the full-coverage basic statistics.
"""
    ),
    markdown(
        """
## 2. חיבור העשרה שמרני

לשני הקבצים אין מזהה משחק משותף אמין, וחותמות הזמן שלהם משתמשות במוסכמות שעון שונות. לכן אנו מחברים משחקים ייחודיים רק כאשר צמד הקבוצות המנורמל והלא־מסודר זהה והפרש הזמנים קטן מעשר שעות. המועמדים משויכים אחד־לאחד לפי הפרש הזמן הקטן ביותר. לאחר מכן העשרת שחקן דורשת התאמה מדויקת של שם שחקן מנורמל בתוך המשחק המקושר. שורות לא ודאיות חוזרות במכוון לסטטיסטיקות הבסיס בעלות הכיסוי המלא.
"""
    ),
    code(
        r'''
from scripts.build_v2_data_exploration_notebook import (
    normalize_player,
    normalize_team,
    normalize_text,
)


primary_matches = (
    maps[["match_id", "datetime", "team1", "team2"]]
    .drop_duplicates("match_id")
    .copy()
)
primary_matches["pair_key"] = primary_matches.apply(
    lambda row: "|".join(sorted((normalize_team(row["team1"]), normalize_team(row["team2"])))),
    axis=1,
)

rich_matches = rich_raw[["hltv_match_id", "date", "team1_name", "team2_name"]].copy()
rich_matches["rich_datetime"] = (
    pd.to_datetime(rich_matches["date"], utc=True, errors="raise").dt.tz_localize(None)
)
rich_matches["pair_key"] = rich_matches.apply(
    lambda row: "|".join(sorted((normalize_team(row["team1_name"]), normalize_team(row["team2_name"])))),
    axis=1,
)

candidates = primary_matches.merge(
    rich_matches[["hltv_match_id", "rich_datetime", "pair_key"]], on="pair_key", how="inner"
)
candidates["gap_hours"] = (
    candidates["datetime"] - candidates["rich_datetime"]
).abs().dt.total_seconds().div(3600.0)
candidates = candidates.loc[candidates["gap_hours"].le(RICH_MATCH_TOLERANCE_HOURS)].sort_values(
    ["gap_hours", "match_id", "hltv_match_id"], kind="stable"
)

used_primary: set[object] = set()
used_rich: set[object] = set()
accepted_links: list[dict[str, object]] = []
for row in candidates.itertuples(index=False):
    if row.match_id in used_primary or row.hltv_match_id in used_rich:
        continue
    used_primary.add(row.match_id)
    used_rich.add(row.hltv_match_id)
    accepted_links.append(
        {"match_id": row.match_id, "hltv_match_id": row.hltv_match_id, "gap_hours": row.gap_hours}
    )

match_links = pd.DataFrame(accepted_links)
assert not match_links["match_id"].duplicated().any()
assert not match_links["hltv_match_id"].duplicated().any()
assert match_links["gap_hours"].le(RICH_MATCH_TOLERANCE_HOURS).all()

maps = maps.merge(match_links[["match_id", "hltv_match_id"]], on="match_id", how="left")
print(f"Conservative one-to-one match links: {len(match_links):,}")
display(match_links["gap_hours"].describe().to_frame("hours"))
'''
    ),
    markdown(
        """
## 3. Player-level Impact score with graceful fallback

Each valid map is reshaped to ten player rows. The basic impact score is the sum of within-map z-scores for ADR, KAST, and kill/death difference. The rich table is match-level rather than map-level, so its z(Rating) + z(KPR) − z(DPR) term is applied exactly once: only in the update after the final map of a safely linked match. This prevents later maps from leaking into earlier-map state. After the rich source ends, `has_rich_stats` becomes false and the formula automatically uses only the basic term.

For the player update, Impact is centered within each five-player side. The resulting relative-performance terms sum to zero, so the performance adjustment rewards an above-average player on both wins and losses without changing the team's conserved Elo movement.
"""
    ),
    markdown(
        """
## 3. ציון Impact ברמת שחקן עם מנגנון גיבוי

כל מפה תקינה מומרת לעשר שורות שחקן. ציון ההשפעה הבסיסי הוא סכום ציוני התקן בתוך המפה עבור ADR, KAST והפרש הריגות/מיתות. הטבלה העשירה היא ברמת משחק שלם ולא ברמת מפה, ולכן האיבר z(Rating) + z(KPR) − z(DPR) מוחל פעם אחת בלבד: בעדכון שלאחר המפה האחרונה של משחק שחובר בבטחה. כך מידע ממפות מאוחרות אינו דולף למצב של מפה מוקדמת יותר. לאחר שמקור ההעשרה מסתיים, `has_rich_stats` הופך לשקר והנוסחה משתמשת אוטומטית רק באיבר הבסיסי.

לצורך עדכון השחקן, ציון ה־Impact ממורכז בתוך כל צד של חמישה שחקנים. רכיבי הביצוע היחסיים המתקבלים מסתכמים באפס, ולכן התאמת הביצועים מתגמלת שחקן מעל הממוצע הן בניצחון והן בהפסד, בלי לשנות את תנועת ה־Elo הכוללת שנשמרת עבור הקבוצה.
"""
    ),
    code(
        r'''
last_row_by_match = maps.groupby("match_id", sort=False)["_row_id"].max()
primary_player_rows: list[dict[str, object]] = []
for values in maps.to_dict(orient="records"):
    for side in (1, 2):
        for slot in range(1, 6):
            prefix = f"team{side}_player{slot}"
            player_id = values.get(f"{prefix}_id")
            player_name = values.get(prefix)
            if pd.notna(player_id):
                player_key = f"id:{int(float(player_id))}"
            elif normalize_player(player_name):
                player_key = f"name:{normalize_player(player_name)}"
            else:
                player_key = f"unknown:{values['match_id']}:{side}:{slot}"
            primary_player_rows.append(
                {
                    "_row_id": values["_row_id"],
                    "match_id": values["match_id"],
                    "hltv_match_id": values.get("hltv_match_id"),
                    "datetime": values["datetime"],
                    "is_final_map_of_match": values["_row_id"] == last_row_by_match.loc[values["match_id"]],
                    "side": side,
                    "slot": slot,
                    "player_key": player_key,
                    "player_name": player_name,
                    "player_norm": normalize_player(player_name),
                    "adr": values.get(f"{prefix}_adr"),
                    "kast": values.get(f"{prefix}_kast"),
                    "kddiff": values.get(f"{prefix}_kddiff"),
                }
            )

players = pd.DataFrame(primary_player_rows)
assert len(players) == len(maps) * 10

rich_player_rows: list[dict[str, object]] = []
for row in rich_raw.itertuples(index=False):
    values = row._asdict()
    for side in (1, 2):
        for slot in range(1, 6):
            prefix = f"team{side}_player_{slot}"
            rich_player_rows.append(
                {
                    "hltv_match_id": values["hltv_match_id"],
                    "player_norm": normalize_player(values.get(f"{prefix}_name")),
                    "rich_rating": values.get(f"{prefix}_RATING"),
                    "rich_kpr": values.get(f"{prefix}_KPR"),
                    "rich_dpr": values.get(f"{prefix}_DPR"),
                }
            )

rich_players = pd.DataFrame(rich_player_rows)
rich_players = rich_players.loc[rich_players["player_norm"].ne("")].drop_duplicates(
    ["hltv_match_id", "player_norm"], keep="first"
)
players = players.merge(
    rich_players,
    on=["hltv_match_id", "player_norm"],
    how="left",
    validate="many_to_one",
)

def safe_z(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    std = numeric.std(ddof=0)
    if pd.isna(std) or std <= 1e-12:
        return pd.Series(0.0, index=series.index)
    return (numeric - numeric.mean()) / std


for column in ("adr", "kast", "kddiff"):
    players[f"z_{column}"] = players.groupby("_row_id", sort=False)[column].transform(safe_z)
players["impact_basic"] = players[["z_adr", "z_kast", "z_kddiff"]].sum(axis=1, min_count=1).fillna(0.0)

for column in ("rich_rating", "rich_kpr", "rich_dpr"):
    players[f"z_{column}"] = players.groupby("_row_id", sort=False)[column].transform(safe_z)
players["has_rich_stats"] = (
    players[["rich_rating", "rich_kpr", "rich_dpr"]].notna().all(axis=1)
    & players["is_final_map_of_match"]
)
players["impact_rich"] = players["z_rich_rating"] + players["z_rich_kpr"] - players["z_rich_dpr"]
players["impact_p"] = players["impact_basic"] + np.where(
    players["has_rich_stats"], RICH_IMPACT_WEIGHT * players["impact_rich"], 0.0
)

def centered_performance(series: pd.Series) -> pd.Series:
    filled = series.fillna(series.median()).fillna(0.0)
    return filled - filled.mean()


players["relative_performance_basic"] = players.groupby(
    ["_row_id", "side"], sort=False
)["impact_basic"].transform(centered_performance)
players["relative_performance_rich"] = players.groupby(
    ["_row_id", "side"], sort=False
)["impact_p"].transform(centered_performance)

for relative_column in ("relative_performance_basic", "relative_performance_rich"):
    centered_sums = players.groupby(["_row_id", "side"])[relative_column].sum()
    assert np.allclose(centered_sums.to_numpy(), 0.0, atol=1e-10)
assert not players.loc[players["datetime"].gt(pd.Timestamp("2025-10-31")), "has_rich_stats"].any()

print(f"Player-map rows: {len(players):,}")
print(f"Rich player-map matches: {int(players['has_rich_stats'].sum()):,} ({players['has_rich_stats'].mean():.2%})")
print("Rich enrichment correctly falls back to basic ADR/KAST/KD after October 2025.")
'''
    ),
    markdown(
        """
## 4. Stateful Team Elo, Player Elo, and exponential inactivity decay

Immediately before a team or player is used again, its stored rating regresses toward 1500:

`R_effective = 1500 + (R_stored - 1500) × 2 ** (-gap_days / half_life)`

The pre-map team and roster features are recorded first. Only then is the outcome applied. The team delta is `K_team × (actual − expected)`. Each player receives one fifth of that team delta plus `K_perf × relative_performance`. Because relative performance is centered within the side, the five player deltas sum exactly to the team delta. Exact-lineup history is also counted strictly before the current map.
"""
    ),
    markdown(
        """
## 4. Elo מצבי לקבוצה ולשחקנים ודעיכה מעריכית לאחר חוסר פעילות

מיד לפני שימוש חוזר בקבוצה או בשחקן, הדירוג השמור נסוג לכיוון 1500:

`R_effective = 1500 + (R_stored - 1500) × 2 ** (-gap_days / half_life)`

תחילה נרשמים פיצ'רי הקבוצה והסגל שלפני המפה, ורק לאחר מכן מוחלת התוצאה. שינוי הקבוצה הוא `K_team × (actual − expected)`. כל שחקן מקבל חמישית משינוי הקבוצה ועוד `K_perf × relative_performance`. מאחר שהביצוע היחסי ממורכז בתוך הצד, סכום חמשת עדכוני השחקנים שווה בדיוק לשינוי הקבוצה. גם היסטוריית ההרכב המדויק נספרת רק עד לפני המפה הנוכחית.
"""
    ),
    code(
        r'''
import sys

FEATURE_MODULE_DIR = PROJECT_ROOT / "src" / "features"
if str(FEATURE_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURE_MODULE_DIR))

from player_elo_state import PlayerEloReplayEngine

replay_engine = PlayerEloReplayEngine(
    maps,
    players,
    initial_rating=MU,
    k_team=K_TEAM,
    team_half_life_days=TEAM_HALF_LIFE_DAYS,
    player_half_life_days=PLAYER_HALF_LIFE_DAYS,
    k_perf=K_PERF,
    team_normalizer=normalize_team,
    player_normalizer=normalize_player,
)
exclusive_end = maps["datetime"].max() + pd.Timedelta(nanoseconds=1)
replay_result = replay_engine.replay_until(exclusive_end)
engineered = replay_result.features.drop(columns="_row_id")

assert len(engineered) == len(maps)
probability_columns = [
    column for column in engineered if column.startswith("raw_") and column.endswith("prob")
]
assert engineered[probability_columns].apply(
    lambda column: column.between(0.0, 1.0).all()
).all()
assert engineered["datetime"].is_monotonic_increasing
assert engineered[["team1_rich_stats_final_map", "team2_rich_stats_final_map"]].dtypes.eq(bool).all()
assert np.allclose(
    engineered["player_agg_elo_diff"],
    engineered["team1_player_agg_elo"] - engineered["team2_player_agg_elo"],
)

print(f"Engineered point-in-time rows: {len(engineered):,}")
print(f"Engineered columns: {engineered.shape[1]}")
print("Replay implementation: src/features/player_elo_state.py")
'''
    ),
    markdown(
        """
## 5. Locked calendar split and artifact

The split is anchored to the calendar, not recomputed from row percentages: Train ends on 31 December 2025, Validation is 2026 Q1, and Test begins on 1 April 2026. This retains the complete 2026 Cologne event in Test. The resulting approximately 81/9/9 distribution is a consequence of those event-safe boundaries.
"""
    ),
    markdown(
        """
## 5. חלוקה כרונולוגית נעולה וארטיפקט

החלוקה מעוגנת בלוח השנה ואינה מחושבת מחדש לפי אחוזי שורות: Train מסתיים ב־31 בדצמבר 2025, Validation הוא רבעון 1 של 2026, ו־Test מתחיל ב־1 באפריל 2026. כך אירוע Cologne של 2026 נשאר בשלמותו ב־Test. ההתפלגות בקירוב 81/9/9 היא תוצאה של גבולות בטוחים לאירועים אלה.
"""
    ),
    code(
        r'''
engineered["split"] = np.select(
    [engineered["datetime"].le(TRAIN_END), engineered["datetime"].le(VAL_END)],
    ["train", "validation"],
    default="test",
)

split_order = ["train", "validation", "test"]
split_counts = engineered["split"].value_counts().reindex(split_order)
split_report = pd.DataFrame(
    {"rows": split_counts, "share": (split_counts / len(engineered)).map(lambda x: f"{x:.2%}")}
)

cologne_2026 = engineered[
    engineered["tournament"].astype(str).str.contains("Cologne", case=False, na=False)
    & engineered["datetime"].dt.year.eq(2026)
]

assert not cologne_2026.empty, "The expected 2026 Cologne event was not found."
assert cologne_2026["split"].eq("test").all(), "The 2026 Cologne event was split or left Test."
assert split_counts.to_dict() == {"train": 5472, "validation": 633, "test": 595}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
engineered.to_parquet(OUTPUT_PATH, index=False)
player_events = players[
    [
        "_row_id", "match_id", "datetime", "side", "slot", "player_key",
        "relative_performance_basic", "relative_performance_rich", "has_rich_stats",
    ]
].sort_values(["_row_id", "side", "slot"], kind="stable")
player_events.to_parquet(PLAYER_EVENTS_PATH, index=False)

display(split_report)
print(f"2026 Cologne rows kept intact in Test: {len(cologne_2026):,}")
print(f"Saved integrated point-in-time dataset: {OUTPUT_PATH}")
print(f"Saved deterministic player-update events: {PLAYER_EVENTS_PATH}")
'''
    ),
    markdown(
        """
## 6. Final engineered feature preview

The preview below contains identifiers, the target, and the point-in-time modeling features. Post-map player box scores and the internal Impact allocation are intentionally absent. Two exported `*_rich_stats_final_map` booleans remain as audit labels for whether the completed map supplied a rich update; because that fact is learned after the match, the tuning notebook explicitly excludes those two labels and instead uses the pre-match `*_has_prior_rich_stats` reliability signals.
"""
    ),
    markdown(
        """
## 6. תצוגה מקדימה של הפיצ'רים הסופיים

התצוגה שלהלן כוללת מזהים, יעד ופיצ'רי מידול בזמן אמת. נתוני השחקנים שלאחר המפה והקצאת ה־Impact הפנימית נעדרים בכוונה. שני הדגלים `*_rich_stats_final_map` נשמרים כתוויות ביקורת בלבד ומציינים אם המפה שהסתיימה סיפקה עדכון עשיר; מאחר שמידע זה נודע לאחר המשחק, מחברת הכוונון מוציאה אותם מפורשות מהמודל ומשתמשת במקום זאת באותות האמינות הקדם־משחקיים `*_has_prior_rich_stats`.
"""
    ),
    code(
        r'''
preview_columns = [
    "datetime", "match_id", "game_id", "tournament", "map_name", "team1", "team2",
    "team1_win", "split", "team1_elo_decay", "team2_elo_decay", "team_elo_diff",
    "team1_player_agg_elo", "team2_player_agg_elo", "player_agg_elo_diff",
    "team1_player_cold_starts", "team2_player_cold_starts",
    "team1_lineup_prior_maps", "team2_lineup_prior_maps",
    "team1_rich_stats_final_map", "team2_rich_stats_final_map",
    "team1_has_prior_rich_stats", "team2_has_prior_rich_stats",
    "raw_team_elo_prob", "raw_player_elo_prob",
]
display(engineered[preview_columns].head())
'''
    ),
    markdown(
        """
## Conclusion

Version 2.0 now has a reproducible, leakage-safe integrated Team/Player dataset. Rich statistics improve historical player updates where confidently available, while the full-coverage basic formula carries the system through Validation and Test. The next stage should ablate decay and enrichment one mechanism at a time on Validation; Test remains untouched for model selection.
"""
    ),
    markdown(
        """
## מסקנה

לגרסה 2.0 יש כעת מערך נתונים משולב של קבוצה ושחקנים, ניתן לשחזור וללא דליפה. הסטטיסטיקות העשירות משפרות עדכוני שחקנים היסטוריים כאשר החיבור בטוח, בעוד שנוסחת הבסיס בעלת הכיסוי המלא ממשיכה דרך Validation ו־Test. בשלב הבא יש לבצע Ablation לדעיכה ולהעשרה, מנגנון אחד בכל פעם, על Validation בלבד; Test נשאר מחוץ לבחירת המודל.
"""
    ),
]

def build_notebook() -> Path:
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(f"Wrote {OUTPUT}")
    return OUTPUT


if __name__ == "__main__":
    build_notebook()
