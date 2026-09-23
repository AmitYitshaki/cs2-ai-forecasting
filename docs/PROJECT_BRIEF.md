# CS2 AI Model — Project Brief

> **Historical planning document.** This brief preserves the project's original V1 scope and data discoveries. It is not the current production specification. For the deployed V2 system, start with [Architecture](ARCHITECTURE.md), [Results](RESULTS.md), and the [documentation index](README.md).

מסמך זה הוא מסמך הרקע (Context) של הפרויקט הספציפי הזה. הוא **משלים** את `Kaggle_AI_Agents_Playbook.md` (שמגדיר איך אנחנו עובדים — תפקידים, סטנדרטים, ה-Pipeline הכללי) ולא מחליף אותו. מטרת המסמך: כל סוכן/אדם שמצטרף לפרויקט, גם בלי היסטוריית השיחה, יבין תוך 5 דקות קריאה מה בונים, למה, ומה כבר ידוע על הדאטה.

> **עדכון מצב — 23.09.2026:** המסמך משמר את נקודת הפתיחה והנחות התכנון של V1. גרסה 2.0 הושלמה לאחר מכן ומוסיפה שכבת Player Elo מודעת־סגל, דעיכת זמן נפרדת לקבוצה ולשחקן, XGBoost וכיול איזוטוני. המפרט העדכני נמצא ב־`ARCHITECTURE.md`, התוצאות הקנוניות ב־`RESULTS.md`, והלקחים ב־`V2_POST_MORTEM.md`.

---

## 1. מה אנחנו בונים

**Tournament Simulator ל-CS2** — מערכת שמקבלת רשימת משתתפים וברקט של טורניר Tier-1, ומחזירה הסתברויות: מי מנצח כל מאצ', ומי מנצח את הטורניר כולו.

הארכיטקטורה היא **שתי שכבות נפרדות**, לא מודל אחד:

1. **Map Classifier** (XGBoost) — `P(team_A wins map | map_name, team_A_features, team_B_features)`. זה המודל שמאומן ישירות על הדאטה הטבלאי שלנו.
2. **Series/Bracket Simulator** — שכבה נפרדת (לא ML) שלוקחת את הסתברויות המפה, מדמה (Monte Carlo) סדרת Bo1/Bo3/Bo5, ואז מקדמת את המנצחת בעץ הטורניר. זו השכבה שהופכת "מודל שמנחש מפה" ל"סימולטור טורניר".

### אירועי בדיקה (Validation Targets) ל-2026
| אירוע | תאריכים | מיקום | הערה |
|---|---|---|---|
| StarLadder StarSeries Fall | 17–20 בספט' | ברצלונה | הכי קרוב — כנראה מוקדם מדי ל-live prediction |
| PGL Masters Bucharest | 24–31 באוק' | בוקרשט | חלון זמן ריאלי ל-live prediction |
| IEM Beijing | 2–8 בנוב' | בייג'ינג | חלון זמן ריאלי ל-live prediction |
| IEM Cologne Major 2026 | 2–21 ביוני | קלן | **כבר בדאטה שלנו** — Falcons ניצחו את FURIA 3-0. נמצא בתוך חלון ה-Test שלנו (2026 Q2) |
| PGL Major Singapore | נוב'-דצמ' (תאריך התחלה לא מאומת סופית) | סינגפור | היעד הסופי — ה-Major השני של השנה |

---

## 2. פריימר CS2 (לרענון מהיר)

- **פורמט מאצ':** Bo1 / Bo3 / Bo5 (best-of). כל "map" הוא יחידת התחרות הבסיסית.
- **מבנה סיבוב (Regulation):** MR12 — כל חצי מקסימום 12 סיבובים, קבוצה מנצחת מפה כשהיא מגיעה ל-**13 ניצחונות**. סה"כ עד 24 סיבובי רגולציה. **סיבוב 1 וסיבוב 13 הם הפיסטולים** (פתיחת כל חצי, נשק בסיס בלבד — קריטי למומנטום/כלכלה).
- **Overtime (OT):** אם 12-12 בתום הרגולציה — MR3, ראשון ל-4 מתוך עד 6 סיבובים, עם reset כלכלי. **⚠️ העמודות בדאטה הגולמי שלנו לא כוללות OT rounds בפועל** (ראו סעיף 3).
- **החלפת צד:** אחרי סיבוב 12 (וכל 3 סיבובים ב-OT).
- **מאגר המפות הפעיל** (Active Duty Pool) כפי שמופיע בדאטה שלנו: Mirage, Nuke, Anubis, Inferno, Dust2, Overpass, Ancient, Train, Vertigo.
- **מבנה טורניר טיפוסי:** שלב בתים (לרוב Swiss) → פלייאוף (single/double elimination). **לא אימתנו** את הפורמט המדויק לכל טורניר ספציפי — יש לבדוק פר-אירוע כשבונים את סימולטור הברקט.
- **שני Majors בשנה** (מבנה Valve הנוכחי, לפי מה שאימתנו ל-2026: Cologne ביוני, Singapore בנוב'/דצמ').

---

## 3. מצב הדאטה — מה יש לנו ומה למדנו

### מבנה התיקיות — שתי תיקיות, לא אחת

| תיקייה | תפקיד |
|---|---|
| `Raw Data bases/` | **הארכיון הגולמי** — כל 20 הקבצים כפי שהורדו/נוצרו, כולל כפילויות וקבצים לא בשימוש. לא נמחק ממנה כלום. |
| `data/` | **תיקיית העבודה** — 9 קבצים בלבד, אלה שבאמת בשימוש/מתוכננים לשימוש קרוב. **מכאן טוענים ב-notebooks**, לא מ-`Raw Data bases/`. |

תוכן `data/` נכון לעכשיו (הועתק, לא הוזז — המקור עדיין ב-`Raw Data bases/`):

| קובץ | סטטוס |
|---|---|
| `final_tournament_features.csv` | ✅ בשימוש — טבלת v1 |
| `team_dna_features.csv` | ✅ בשימוש — פיצ'רים גזורים |
| `cs2_tier1_games.csv` | ✅ בשימוש — מקור ל-v1 |
| `combined_round_by_round_with_map_names_cleaned.csv` | ✅ בשימוש — מקור ל-DNA |
| `cs2_newestcombinedmatches.csv` | 🔜 שלב 9 (v2) |
| `newest_ts_ds.csv` | 🔜 שלב 9 (v2) |
| `players.csv` / `teams.csv` / `tournaments.csv` | 🗂️ lookup, קטנות |

**נשארו מחוץ ל-`data/` בכוונה** (עדיין ב-`Raw Data bases/` בלבד): `combined_round_by_round_all.csv` ו-`_with_map_names.csv` (סופרסדים ע"י `_cleaned`), `cs2_all_tiers_games.csv` (=tier1+2+3, לא בשימוש ישיר), `cs2_tier2_games.csv`/`cs2_tier3_games.csv` (לא בתוכנית — Tier-1 בלבד), `cs2_results.csv` (הוחלט להשמיט).

**⚠️ עדיין פתוח:** `cs2_newestcombinedmatches_team1_reference_reduced.csv` ו-`_reduced2.csv` **לא** נכנסו ל-`data/` — עדיין לא הוכרע ביניהן (הערכים שונים בפועל, לא רק סדר עמודות). להכריע **לפני** שלב 9.

**עדכון יישום:** לוגיקת הסינון, רשימת הפיצ'רים, הסימטריזציה והפיצ'רים היחסיים נמצאת ב־`src/features/tabular.py`; מנוע V1 Point-in-Time Elo נמצא ב־`src/features/elo.py`, ומנוע V2 מודע־השחקנים ב־`src/features/player_elo_state.py`. הטבלאות הנגזרות וה־SHA של ארטיפקטי המידול מתועדים תחת `data/` ו־`artifacts/`.

### קבצי המקור (15 קבצים, `Raw Data bases/`)
מקור: Kaggle (griffindesroches ואחרים) + הורדות נוספות. ביקורת מלאה נשמרה בפרויקט Claude (`raw-data-audit-and-merge-plan.md`). תקציר:

| קובץ | תפקיד |
|---|---|
| `combined_round_by_round_with_map_names_cleaned.csv` | round-by-round גולמי, אוג'2024–אוק'2025, 10,784 מפות — המקור ל-Team DNA |
| `cs2_tier1_games.csv` / `_all_tiers` / `_tier2` / `_tier3` | דאטה ברמת מפה עם box score מלא לכל שחקן, ינו'2023–יונ'2026 — **היחיד שמגיע ל-2026** |
| `cs2_newestcombinedmatches*.csv`, `newest_ts_ds.csv` | פיצ'רים הנדסיים ברמת מאצ' (rating diffs, head2head, opening kills) — עד אוק'2025 בלבד |
| `players.csv` / `teams.csv` / `tournaments.csv` | טבלאות lookup |

### קבצים גזורים (Feature Factory)
| קובץ | נבנה מ- | תוכן |
|---|---|---|
| `team_dna_features.csv` | round-by-round cleaned | `pistol_round_win_rate`, `ct_win_rate`, `t_win_rate` לכל (קבוצה, מפה), עם `n_` לכל rate |
| `final_tournament_features.csv` | `cs2_tier1_games.csv` + `team_dna_features.csv` | הטבלה הסופית ל-**v1** של Modeling: 11,151 שורות × 112 עמודות |

### ⚠️ `final_tournament_features.csv` הוא **לא** "כל המידע שיש לנו" — טבלת v1 בלבד

חשוב שכל סוכן שנוגע במודל ידע בדיוק מה כן ומה לא נכנס לקובץ הזה:

| קובץ | בפנים ב-v1? | הערה |
|---|---|---|
| `cs2_tier1_games.csv` | ✅ כן | המקור הראשי |
| `team_dna_features.csv` | ✅ כן | ממוזג פנימה |
| `cs2_tier2_games.csv` / `cs2_tier3_games.csv` | ❌ לא | במפורש הוצא — המטרה היא Tier-1 בלבד |
| `cs2_newestcombinedmatches*.csv` | ❌ לא עדיין | rating/ADR/KAST diffs, head2head — **מתוכנן ל-v2**, ראו סעיף 4 |
| `newest_ts_ds.csv` | ❌ לא עדיין | **opening kills/deaths** (הפרוקסי היחיד שיש לנו ל-first-kill), rolling-5 form — **מתוכנן ל-v2** |
| `cs2_results.csv` | ❌ לא | הוחלט להשמיט — דל, ללא תאריך אמין |
| `players.csv` / `teams.csv` / `tournaments.csv` | ❌ לא (lookup בלבד) | לא פיצ'רים |

**הערה מבנית קריטית ל-v2:** `cs2_newestcombinedmatches*` ו-`newest_ts_ds` הם **ברמת מאצ' בודד** (לא טבלת פרופיל-קבוצה כמו ה-DNA). כדי להשתמש בהם על מאצ'ים עתידיים חייבים Feature Factory שני שמחלץ מהם פרופיל-קבוצה סטטי — אותה מתודולוגיה כמו ל-DNA, כולל שלב inspection דומה (ראו לקח מ-#1 למטה) לפני שסומכים על העמודות.

### ⚠️ ממצאי איכות דאטה קריטיים (חובה לדעת לפני שנוגעים בקוד)
1. **באג off-by-one בעמודות הפיסטול** — `map1_round13_winner` בפועל **לא** הפיסטול השני האמיתי. תוקן: `PISTOL_ROUNDS = {1, 14}` (מבוסס על 0 sweeps נקיים מתוך 10,232 מפות — הוכחה סטטיסטית, לא אימות ידני מול HLTV).
2. **Padding אחרי סוף משחק אמיתי** — עמודות הסיבובים תמיד מלאות ל-24 גם כשהמשחק הסתיים קודם. תוקן ב-`compute_true_end_round()` — חותך את הסיבובים העודפים לפני אגרגציה.
3. **חלון ה-DNA (אוג'2024–אוק'2025) חופף ל-46% מהדאטה** — יש in-sample leakage מתון לשורות Train בתוך החלון הזה (הפיצ'ר של קבוצה כולל חלקית את המאצ' עצמו). **לא פוגע ב-Test (2026 כולו אחרי החלון)**, אבל מדדי ביניים על 2023-2025 עלולים להיות אופטימיים מדי.
4. **"Inner Circle"** ב-`teams.csv` הוא placeholder ליריב לא ידוע (40+ id שונים) — לא קבוצה אמיתית, סונן.
5. **כפילויות capitalization** ב-DNA table התגלו ורק ב-Grand Join (`PARTIZAN` מול `Partizan` וכו') — קיים תיקון (`collapse_dna_duplicates`, weighted merge) + assertion שמונע הישנות שקטה.
6. **אין נתוני כלכלה (economy/buy) בשום קובץ** — נבדק בחיפוש נרחב בתחילת הפרויקט, לא נמצא מקור זמין תוך timebox סביר.
7. **כיסוי DNA: 65.6%** מהשורות ב-`final_tournament_features.csv` עם DNA לשני הצדדים. 177 שמות קבוצה נשארו ללא DNA כלל (הופיעו רק מחוץ לחלון אוג'2024-אוק'2025).
8. **`TEAM_ALIASES`** — מילון ידני ל-20 ארגוני Tier-1, **ללא** קבוצות Academy/Junior (הוסרו במפורש כדי לא לזהם DNA של הקבוצה הראשית) + fallback ל-`rapidfuzz` (סף 90) למקרים שנשארו.

---

## 4. אסטרטגיית המידול המוסכמת

**Split כרונולוגי תלת-שכבתי** (לא random split, לא true walk-forward — נימוק מלא בהיסטוריית השיחה):

| Split | תקופה | מאצ'ים | שימוש |
|---|---|---|---|
| Train | 2023-01 → 2025-12 | ~3,779 | אימון |
| Validation | 2026 Q1 | 321 | Hyperparameter tuning בלבד |
| Test (Holdout) | 2026 Q2 (כולל IEM Cologne Major) | 318 | נמדד **פעם אחת**, בסוף |

**נקודות טכניות שנקבעו:**
- **Symmetrization** — כל שורה מוכפלת עם team1↔team2 מוחלפים + תווית הפוכה, כדי למנוע הטיית "team1 נוטה לנצח" מסדר הרישום.
- **NaN handling** — XGBoost native sparse-aware splits, לא imputation ידני (34.4% מהשורות חסרות DNA לפחות בצד אחד).
- **מדדים:** לא רק accuracy — **log-loss / Brier score** קריטיים כי סימולציית ברקט צריכה הסתברויות מכוילות. Reliability diagram כחלק מהבדיקה הסופית.
- **n=318 מאצ'ים ב-Test הוא לא גדול** — לדווח עם confidence interval (bootstrap), לא מספר יבש.

---

## 5. מגבלות ידועות / סיכונים פתוחים ל-v2

- DNA הוא **סטטי** (חלון קבוע אחד), לא point-in-time rolling — שינוי רוסטרים לא נלכד.
- אין נתוני כלכלה/וטו מפות (map veto order).
- פורמט הטורניר (Swiss/playoffs) לא מקודד — סימולטור הברקט צריך את המבנה הזה כקלט חיצוני לכל אירוע.
- 177 שמות קבוצה עדיין ללא DNA — פוטנציאל לשיפור עם alias dictionary מורחב.
- **v2 מתוכנן ומחויב** (לא "אולי אחרי error analysis") — `cs2_newestcombinedmatches*` + `newest_ts_ds` יעובדו ל-Feature Factory שני מיד אחרי מדידת v1 על Validation, לפני שנוגעים ב-Test. ראו `WORK_PLAN.md` שלבים 9-11.
