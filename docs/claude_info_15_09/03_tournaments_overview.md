# Upcoming Tournaments — Overview & StarLadder Deep Dive

Compiled 2026-09-15 via web search. Confidence is noted per event — this project has already hit low-quality/self-contradictory esports content in prior searches (see the shelved bookmaker-odds task), so every claim below is cross-checked against at least two independent sources before being stated as confirmed, and flagged clearly where it isn't.

---

## Brief overview

| Event | Dates | Location | Format | Simulator compatibility |
|---|---|---|---|---|
| StarLadder StarSeries Fall 2026 | Sep 17–20, 2026 | Barcelona, Spain | **Double elimination**, Bo3 (Bo5 final), no group stage — straight to an 8-team playoff bracket | **Not compatible as-is** — see below |
| PGL Masters Bucharest 2026 | Oct 24–31, 2026 | Bucharest, Romania | Swiss group stage → single-elimination playoff, Bo5 final | Compatible with current architecture |
| IEM Beijing 2026 | Nov 2–8, 2026 | Beijing, China | 16 teams, Bo3 group stage → Bo3 playoffs (QF/SF + 3rd-place decider) → Bo5 final | Compatible with current architecture; note the extra 3rd-place match, which the current module doesn't produce (non-blocking — champion/finalist probabilities are unaffected) |

*(Note: there is also a "PGL Bucharest 2026" in April with a different, larger field — not the event referenced in the Project Brief's Q4 target list. The October "PGL Masters Bucharest 2026" is the correct one.)*

---

## Deep dive: StarLadder StarSeries Fall 2026 (Barcelona)

### Format — confirmed, and this is a blocker

**This is a double-elimination bracket, not single-elimination.** Confirmed independently by two sources: a web search synthesis and a direct fetch of NAVI's own tournament page (a participating team's own event page, [navi.gg](https://navi.gg/en/tournaments/starladder-starseries-fall-2026/playoff)):

> "The competition uses a 'Double Elimination bracket' structure. The grand final is best-of-5, while all other matches are best of 3." — navi.gg
> "The entire tournament features a double-elimination bracket with every match played as a best-of-three." — [tips.gg](https://tips.gg/article/starladder-starseries-fall-2026-upper-bracket-quarterfinals-betting-predictions-tournament-winner-event-story/)

There is no group/Swiss stage — all 8 teams enter the playoff bracket directly.

**This directly matches Finding 2 in `01_review_findings.md`: `bracket_simulator.py` only supports single elimination.** It cannot represent a lower bracket, cannot track which bracket a team is in, and has no redemption-match logic. Pointing the current code at this event is not a hardcoding exercise — it requires new engineering (double-elimination progression logic) before any live run against StarLadder is possible. **This should be raised with the Team Lead as a scheduling/scope decision**: either prioritize building double-elimination support in the ~36 hours before Sept 17, or defer the first live run to PGL Bucharest (Oct 24, single-elimination, already compatible) and treat StarLadder as observation-only this cycle.

### Confirmed team list (8 teams)

MOUZ, NRG, Aurora, NAVI, FURIA, MIBR, magic, Vitality.

- MOUZ, Vitality, FURIA, and NAVI entered via direct/Global VRS invites.
- Aurora and magic advanced through the European qualifier.
- MIBR advanced through the South American qualifier; NRG through the North American qualifier.

### Confirmed upper-bracket Round 1 (Quarterfinal) matchups — Sept 17, 2026

| Match | Teams | Time (as reported) |
|---|---|---|
| UB-QF1 | MOUZ vs NRG | ~12:00–13:00 CEST (minor discrepancy between sources, see note) |
| UB-QF2 | magic vs Vitality | ~14:30–15:30 CEST |
| UB-QF3 | Aurora vs NAVI | ~17:00–18:00 CEST |
| UB-QF4 | FURIA vs MIBR | ~19:30–20:30 CEST |

**Note on times:** the two independent sources agree exactly on match order and pairings, but differ by ~1 hour on kickoff times for each match (e.g. 12:00 vs 13:00 for MOUZ/NRG) — most likely a timezone-labeling inconsistency between sources rather than a real scheduling conflict, but **verify exact start times against the official StarLadder/HLTV schedule immediately before the event**, don't hardcode either of these times as authoritative.

### What is NOT yet knowable

Because this is double elimination, only the **upper-bracket Round 1** pairings are determined pre-event — the full bracket path (who a team faces after a loss, lower-bracket seeding, redemption matches) depends on Round 1 results and is not something that can be "verified" in advance the way the Cologne backtest's already-completed bracket was. Any bracket topology fed to the simulator beyond upper-bracket Round 1 will need to be filled in live, round by round, as results come in — this is a structural difference from the Cologne backtest (fully known in advance) and from a hypothetical single-elimination event (fully seeded in advance).

### Confidence assessment

- **Format (double elimination) and team list**: high confidence — corroborated by two independent, differently-sourced pages with no contradictions.
- **Round 1 pairings**: high confidence on matchups/order — corroborated identically across sources. Medium confidence on exact kickoff times — worth a final check before the event.
- **Anything about Round 2 onward**: not yet determinable, by definition of the format — do not treat any later-round claim from a search result as reliable; none was found stated as fact for this reason.

---

## Sources

- [StarLadder StarSeries Fall 2026 Upper Bracket Quarterfinals Betting Predictions | Tips.GG](https://tips.gg/article/starladder-starseries-fall-2026-upper-bracket-quarterfinals-betting-predictions-tournament-winner-event-story/)
- [StarSeries Fall 2026: Teams, Rosters, Bracket & Schedule | Tips.GG](https://tips.gg/article/starladder-starseries-fall-2026-teams-rosters-bracket-event/)
- [StarLadder StarSeries Fall 2026 Play-off | Natus Vincere (navi.gg)](https://navi.gg/en/tournaments/starladder-starseries-fall-2026/playoff)
- [StarLadder StarSeries Fall teams, format, schedule, prizes, talent, fantasy | HLTV.org](https://www.hltv.org/news/45519/starladder-starseries-fall-teams-format-schedule-prizes-talent-fantasy)
- [StarLadder StarSeries Fall 2026 overview | HLTV.org](https://www.hltv.org/events/8057/starladder-starseries-fall-2026)
- [PGL Masters Bucharest 2026 overview | HLTV.org](https://www.hltv.org/events/8050/pgl-masters-bucharest-2026)
- [IEM Beijing 2026 - ESL Pro Tour](https://pro.eslgaming.com/tour/cs/beijing/)
- [IEM Beijing 2026 Team List Complete After Global Qualifiers | Hotspawn](https://www.hotspawn.com/counter-strike/news/iem-beijing-2026-team-list-complete)
