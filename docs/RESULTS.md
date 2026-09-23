# תוצאות הפרויקט — תמונת מצב סופית

עודכן: 16 בספטמבר 2026.

## החלטת המודל

המודל הייצורי הוא XGBoost על שישה פיצ'רי Elo קנוניים בלבד: דירוג Global ודירוג Map לכל אחת משתי הקבוצות ושני ההפרשים ביניהם. המודל אומן על Train בלבד, והכיול האיזוטוני הותאם על Validation בלבד.

מחקרי ההסרה ב־Notebooks 02–04 בדקו DNA, מפגשים ישירים, כושר חמש מפות אחרונות וכוונון Optuna. השיפורים הנקודתיים לא עברו את רווחי הסמך שנקבעו מראש, ולכן אותן תוספות לא נכנסו לארטיפקט הייצורי.

## מדד Test הנעול

Notebook 1.5 טוען את `canonical_elo_isotonic.joblib` עצמו ומעריך את אותו מודל שמשרת את הסימולטור.

| שכבה | מספר שורות מסומטרות | דיוק | Brier |
|---|---:|---:|---:|
| כלל Test | 1,190 | 0.666387 | 0.214199 |
| מפה 1 | 588 | 0.641156 | 0.222295 |
| מפה 2+ | 602 | 0.691030 | 0.206291 |

לוג־לוס Test הכולל הוא **0.646857**. המספרים הישנים שנוצרו ממודל גולמי שאומן מחדש על Train+Validation מבוטלים; הם אינם מתארים את הארטיפקט שנפרס.

## סימולציית StarLadder — מיליון מסלולים

שתי ריצות עצמאיות בוצעו עם seeds ‏42 ו־99, מיליון טורנירים בכל ריצה. Seed ‏42 משמש טבלת ייחוס ו־seed ‏99 משמש בקרת יציבות.

### הסתברות אליפות — seed 42

| קבוצה | P(Champion) מעוגל |
|---|---:|
| Vitality | 44.4% |
| Natus Vincere | 20.5% |
| FURIA | 14.8% |
| Aurora | 9.4% |
| MOUZ | 8.9% |
| magic | 1.3% |
| MIBR | 0.6% |
| NRG | 0.0% |

הערכים המלאים נשמרים ב־metadata; הטבלה מעוגלת לעשירית האחוז ולכן אינה מיועדת לסכימה ידנית.

### שלושת האירועים המובילים בכל קטגוריה — seed 42

| קטגוריה | אירוע | הסתברות |
|---|---|---:|
| Grand Final matchup | Natus Vincere vs Vitality | 24.9907% |
| Grand Final matchup | FURIA vs Vitality | 19.3101% |
| Grand Final matchup | Aurora vs Vitality | 12.6870% |
| סגנית | Vitality | 23.4931% |
| סגנית | Natus Vincere | 21.8461% |
| סגנית | FURIA | 19.6830% |
| פודיום מדויק | Vitality > Natus Vincere > FURIA | 5.6651% |
| פודיום מדויק | Vitality > FURIA > Natus Vincere | 4.5328% |
| פודיום מדויק | Vitality > Natus Vincere > MOUZ | 3.6989% |
| הגעת Cinderella לגמר | MOUZ | 23.6374% |
| הגעת Cinderella לגמר | Aurora | 22.7636% |
| הגעת Cinderella לגמר | magic | 5.4792% |

קבוצת Cinderella מוגדרת לאחר הריצה כקבוצה עם `P(Champion) < 10%`. המקום השלישי בפודיום הוא המפסיד ב־Consolidation Final, המשחק האחרון בנתיב ה־Lower Bracket שקובע את העולה השנייה לגמר הגדול.

### יציבות בין seeds

| התפלגות | הפרש מוחלט מרבי |
|---|---:|
| P(Champion) | 0.1275% |
| Grand Final matchup | 0.0425% |
| סגנית | 0.0695% |
| פודיום מדויק | 0.0447% |
| הגעת Cinderella לגמר | 0.0410% |

כל הקטגוריות עברו את תנאי הקבלה של 0.5 נקודת אחוז.

## בדיקות ייצור

- בדיקת קדם־טיסה של 5,000 מסלולים הפיקה זהות מלאה בין נתיב NumPy/Booster לבין נתיב Pandas/CalibratedClassifierCV.
- סדר ששת הפיצ'רים נבדק מול `booster.feature_names` לפני הפעלת הנתיב המהיר.
- מוני matchups, סגניות, פודיומים והופעות בגמר מתיישבים בדיוק עם מוני השלבים.
- כל export מכיל SHA־256 של ארטיפקט המודל, seed, מספר איטרציות, זמן ריצה וגרסת schema.
- כל הקבצים נכתבים אטומית ואין קובצי `.tmp` שנותרו לאחר הריצה.
- 12 בדיקות היחידה והאינטגרציה עוברות.

## מגבלות פרשנות

תחזית הטורניר אינה יודעת veto עתידי, שינויי רוסטר, זמינות שחקנים או תוצאות לאחר 21 ביוני 2026. סטיית התקן של Monte Carlo מודדת את רעש הסימולציה בלבד, ולא את אי־הוודאות של המודל או של המידע החסר.

<!-- V2_FINAL_RESULTS_START -->
## Version 2.0 final locked Test / מבחן סופי נעול של גרסה 2.0

### English

Model selection was completed before Test. The match-cluster bootstrap 95% CI for tuned-minus-untuned Validation row loss was `[-0.004206, 0.000516]`; therefore `untuned_approach_b` was locked. Test was scored exactly once on 595 natural-order map rows.

The no-decay boundary probe produced Validation Log-loss `0.657833` versus `0.657036` at 1,095 days. Disabling decay did not improve the selected Player-Elo setting, confirming that the boundary had plateaued without changing the locked model.

| Version | Accuracy | Log-loss | Brier |
|---|---:|---:|---:|
| V1 deployed calibrated model | 0.666387 | 0.646857 | 0.214199 |
| V2 Dynamic Hybrid final | 0.680672 | 0.609913 | 0.209964 |

| V2 Test stratum | Rows | Accuracy (95% cluster CI) | Log-loss (95% cluster CI) | Brier (95% cluster CI) |
|---|---:|---:|---:|---:|
| Map 1 | 294 | 0.615646 (0.557823, 0.673469) | 0.633270 (0.607737, 0.658056) | 0.221194 (0.209162, 0.233105) |
| Map 2+ | 301 | 0.744186 (0.679868, 0.806454) | 0.587100 (0.556989, 0.618393) | 0.198994 (0.184630, 0.213995) |

IEM Cologne 2026 (187 rows): Accuracy `0.689840`, Log-loss `0.608109`, Brier `0.209204`. Falcons' mean map-win probability in the isolated Falcons–FURIA rows is `0.650915`; the V1 series-win reference was `0.703800`, so these are distinct estimands.

Artifact SHA-256: `e3628145a51f831842556e2c68c709a685524b332ca6bcfebf4169f23bd73bbb`.

### עברית

המודל נבחר לפני פתיחת Test. רווח הסמך 95% ב־Bootstrap לפי אשכולות משחק עבור הפרש הפסד השורה המכוונן פחות הלא־מכוונן ב־Validation היה `[-0.004206, 0.000516]`; לכן ננעלה התצורה `untuned_approach_b`. Test חושב פעם אחת בלבד על 595 שורות מפה בסדר הטבעי.

בדיקת הגבול ללא דעיכה הפיקה Validation Log-loss של `0.657833` לעומת `0.657036` ב־1,095 ימים. ביטול הדעיכה לא שיפר את תצורת Player Elo שנבחרה, ולכן הגבול אכן הגיע לרוויה בלי לשנות את המודל הנעול.

הטבלה באנגלית לעיל היא הרשומה הקנונית. IEM Cologne 2026 כולל 187 שורות: דיוק `0.689840`, Log-loss `0.608109` ו־Brier `0.209204`. הסתברות הניצחון הממוצעת של Falcons ברמת מפה בשורות Falcons–FURIA היא `0.650915`; תחזית V1 הייתה הסתברות סדרה של `0.703800`, ולכן אלו אומדים שונים.
<!-- V2_FINAL_RESULTS_END -->

<!-- V2_STARLADDER_MC_START -->
## V2 StarLadder Barcelona Monte Carlo — 1,000,000 paths

Two independent production runs were completed with seeds 42 and 99, with one million full double-elimination tournaments per seed. Seed 42 is the reporting run; seed 99 is the stability control. The historical replay used the hard exclusive cutoff `2026-09-17T00:00:00`.

### P(Champion) — seed 42

| Team | Rounded P(Champion) |
|---|---:|
| Vitality | 38.5% |
| FURIA | 20.4% |
| Natus Vincere | 15.6% |
| MOUZ | 10.7% |
| Aurora | 9.5% |
| magic | 3.8% |
| MIBR | 1.4% |
| NRG | 0.1% |

The exact values are stored in the production metadata; the table is rounded to one decimal place to match the V1 report.

### Top three events per category — seed 42

| Category | Event | Probability |
|---|---|---:|
| Grand Final matchup | FURIA vs Vitality | 20.8930% |
| Grand Final matchup | Natus Vincere vs Vitality | 16.0469% |
| Grand Final matchup | Aurora vs Vitality | 10.2092% |
| Runner-up | FURIA | 21.5064% |
| Runner-up | Vitality | 20.9442% |
| Runner-up | Natus Vincere | 17.4545% |
| Exact podium | Vitality > FURIA > Natus Vincere | 4.3648% |
| Exact podium | Vitality > Natus Vincere > FURIA | 3.5828% |
| Exact podium | Vitality > FURIA > MOUZ | 3.1315% |
| Cinderella Grand Final reach | Aurora | 21.9347% |
| Cinderella Grand Final reach | magic | 11.2672% |
| Cinderella Grand Final reach | MIBR | 5.0457% |

The Cinderella set was recomputed from V2 seed 42 as teams with `P(Champion) < 10%`: Aurora, magic, MIBR, and NRG.

### Seed stability

| Distribution | Maximum absolute difference | Acceptance |
|---|---:|---:|
| P(Champion) | 0.0865 pp | Pass |
| Grand Final matchup | 0.0935 pp | Pass |
| Runner-up | 0.1319 pp | Pass |
| Exact podium | 0.0501 pp | Pass |
| Cinderella Grand Final reach | 0.0760 pp | Pass |
| Most Probable Bracket | Seed-42 #1 appears in seed-99 Top-3 | Pass |

All five probability-distribution checks are below the predeclared 0.5 percentage-point threshold. MPB uses a rank-membership check rather than the absolute-percentage-point rule because each complete path is a rare joint event.

### Top five Most Probable Brackets — seed 42

| Rank | Full winner sequence | Exact probability |
|---:|---|---:|
| 1 | UB QF: MOUZ, Vitality, Natus Vincere, FURIA → UB SF: Vitality, FURIA → UB Final: Vitality → LB R1: magic, Aurora → LB SF: Natus Vincere, Aurora → LB R3: Natus Vincere → LB Final: FURIA → GF: Vitality | 0.2103% |
| 2 | UB QF: MOUZ, Vitality, Aurora, FURIA → UB SF: Vitality, FURIA → UB Final: Vitality → LB R1: magic, Natus Vincere → LB SF: Aurora, Natus Vincere → LB R3: Natus Vincere → LB Final: FURIA → GF: Vitality | 0.1985% |
| 3 | UB QF: MOUZ, Vitality, Natus Vincere, FURIA → UB SF: Vitality, FURIA → UB Final: Vitality → LB R1: magic, Aurora → LB SF: Natus Vincere, Aurora → LB R3: Natus Vincere → LB Final: Natus Vincere → GF: Vitality | 0.1858% |
| 4 | UB QF: MOUZ, Vitality, Natus Vincere, FURIA → UB SF: Vitality, FURIA → UB Final: Vitality → LB R1: magic, Aurora → LB SF: Natus Vincere, MOUZ → LB R3: Natus Vincere → LB Final: FURIA → GF: Vitality | 0.1805% |
| 5 | UB QF: MOUZ, Vitality, Aurora, FURIA → UB SF: Vitality, FURIA → UB Final: Vitality → LB R1: magic, Natus Vincere → LB SF: Aurora, Natus Vincere → LB R3: Natus Vincere → LB Final: Natus Vincere → GF: Vitality | 0.1738% |

MPB #1 is **not** a clear standout. Its lead over #2 is only `0.0118` percentage points, which does not clear the conservative 95% Monte Carlo separation rule. The one million paths contained 11,964 distinct complete bracket routes.

### Production checks

- The extracted replay engine reproduced the locked training features with maximum absolute difference `0.000e+00`.
- All eight teams and all 40 roster slots resolved to historical canonical identities; there were zero player cold starts. NRG slot 5 resolved as `jeorgesnorts → id:3274` with 86 prior maps and last seen `2026-04-06 17:50:00`.
- Fast NumPy inference and calibrated sklearn inference differed by at most `2.785523e-08` on opening probabilities, and their 5,000-path preflight outputs matched exactly.
- The deployed calibration artifact SHA-256 is `a3ab81428571b5147cf944fd14fd7c1aba9e0abbb5b56bf4db12e64a23ebcfd0`.
- Both runs were written atomically to `results/starladder/`, appended to `runs_index.csv`, and left no `.tmp` files.
- Seed 42 ran in `4810.693` seconds; seed 99 ran in `4915.110` seconds.

### Interpretation limits

The forecast cannot know future vetoes, roster availability, substitutions, or form after the last historical map on 2026-06-21. Monte Carlo standard error measures simulation noise only; it does not capture model, calibration, roster, or missing-data uncertainty. The MPB tracker records match winners, not exact series scorelines.

### Post-hoc comparison — loaded only after export lock

The confirmed result was Vitality defeating Aurora 3-1. V2 ranked Vitality first with `P(Champion) = 38.4921%`, so the champion was the model's top choice. None of the Top-5 MPB paths produced a Vitality-over-Aurora Grand Final. The highest-ranked complete route that did so was rank 11 at `0.1476%`: six ranks outside the Top-5 and `0.0262` percentage points below the rank-5 cutoff. The 3-1 scoreline is not evaluated because the MPB contract stores match winners rather than exact series scores.
<!-- V2_STARLADDER_MC_END -->
