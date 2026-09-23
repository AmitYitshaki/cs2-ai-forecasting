# CS2 AI Model — Work Plan

> **Historical execution log.** This checklist documents how V1 was built and should not be read as the current project status. Version 2.0 is complete; see [Architecture](ARCHITECTURE.md), [Results](RESULTS.md), and [Reproducibility](REPRODUCIBILITY.md). Checked items below are retained as an engineering audit trail.

18 שלבים, כל אחד תוצר קונקרטי שמזין ישירות את הבא. תואם למבנה 6 השלבים של `Kaggle_AI_Agents_Playbook.md` (מצוין ליד כל שלב). הסימונים משקפים את מצב תוכנית V1 בזמן הביצוע; מקור האמת הנוכחי הוא `RESULTS.md`.

> **סטטוס עדכני — 16.09.2026:** שלבים 1–17 הושלמו. המודל הייצורי שנבחר הוא Canonical Elo בלבד עם XGBoost וכיול איזוטוני. Test נפתח פעם אחת עבור הארטיפקט שנפרס, בוצעו backtest לקלן ותחזית חיה ל־StarLadder, כולל שתי ריצות של מיליון טורנירים. שלב 18 יושלם רק לאחר סיום האירוע והשוואת התחזית לתוצאות האמת. המספרים המאומתים נמצאים ב־`docs/RESULTS.md`.

> **עדכון (אחרי דיון עם Team Lead):** v2 (הוספת `cs2_newestcombinedmatches*` + `newest_ts_ds`) **לא** מחכה ל-error analysis — הוא מתוכנן מיד אחרי v1, לפני שנוגעים ב-Test בכלל (שלבים 9-11). הסיבה: אנחנו כבר יודעים שהמקורות האלה קיימים ובעלי ערך (opening kills = הפרוקסי היחיד ל-first-kill), אז אין טעם לחכות ל"גילוי". אבל v1 ו-v2 עדיין נמדדים ומושווים **על Validation בלבד** — לא Test — כדי ש-Test יישאר "מבט אחד" אמיתי על הגרסה שסופית נבחרה, לא ייבדק פעמיים.

> **תזכורת מה-Playbook:** אחרי הסבר על קונספט מורכב — עצירה, שאלת וידוא הבנה, המתנה לאישור לפני שממשיכים ("Stop & Verify Gate"). לא מדלגים על זה כי "יש לוח זמנים".

---

### Playbook Stage 1: Setup & Git

- [x] **1. הקמת סביבת הפיתוח**
  פתיחת הפרויקט ב-VS Code, `git init` (אם עוד לא), יצירת virtual environment, התקנת `pandas` `numpy` `scikit-learn` `xgboost` `jupyter` `matplotlib` `seaborn` `rapidfuzz`. יצירת מבנה תיקיות: `src/`, `notebooks/`, `data/` (או קישור ל-`Raw Data bases/`). פתיחת מחברת Jupyter ראשונה: `notebooks/1.0_setup_and_data_sanity.ipynb`.
  **מייצר:** סביבה שכל שלב הבא רץ בתוכה.

- [x] **2. אימות טעינת דאטה (Sanity Check)**
  בתוך `1.0_setup_and_data_sanity.ipynb`: טעינת טבלת המקור והטבלאות הנגזרות, ווידוא shapes/dtypes תואמים למה שמתועד ב-`PROJECT_BRIEF.md`. אם ה-shape לא תואם — לעצור ולברר לפני שממשיכים.
  **מייצר:** ודאות שעובדים על הגרסה הנכונה של הדאטה לפני שבונים עליה.

---

### Playbook Stage 2: EDA & Cleaning

- [x] **3. פורטינג ה-Feature Factory למודולים**
  העברת הלוגיקה מ-`build_team_dna_features.py` ו-`build_final_tournament_features.py` למודול tabular. המיקום הסופי לאחר ניקוי המבנה הוא `src/features/tabular.py`. המחברת קוראת למחלקות, ולא כוללת את הלוגיקה inline.
  **מייצר:** pipeline משוחזר (reproducible) מדאטה גולמי, לא רק קובץ CSV סטטי שקיבלנו.

- [x] **4. EDA מלא**
  מחברת `notebooks/01_eda.ipynb`: התפלגות ניצחונות team1 מול team2 (בדיקת סימטריה), כיסוי DNA לאורך זמן, התפלגות לפי מפה, missingness map, בדיקת שפיות ה-DNA rates מול קבוצות מוכרות (למשל NAVI אמור להיות מעל 50% על Dust2 היסטורית).
  **מייצר:** רשימת בעיות/הפתעות בדאטה לפני שממשיכים לפיצ'רים — ומאשר או סותר את ההנחות מ-`PROJECT_BRIEF.md`.

---

### Playbook Stage 4: Feature Engineering & OOP Refactoring
*(שלב 3 של ה-Playbook — Baseline — יבוא רק אחרי שיש Train/Val/Test נעולים, ראו שלב 6-7 למטה)*

- [x] **5. Symmetrization + Diff Features**
  ב-`src/features/tabular.py`: הכפלת כל שורה עם team1↔team2 מוחלפים ותווית הפוכה. בניית גרסת diff-features (`team1_stat - team2_stat`) כאלטרנטיבה לבדיקה מול הפיצ'רים הדו-צדדיים הגולמיים.
  **מייצר:** שתי גרסאות פיצ'רים מוכנות להשוואה במודל.

- [x] **6. נעילת ה-Chronological Split**
  קוד שמייצר את 3 קבצי ה-split לפי הגבולות שנקבעו ב-`PROJECT_BRIEF.md` (Train ≤2025-12 / Val=2026Q1 / Test=2026Q2), שומר אותם כקבצים נפרדים (`data/train.parquet` וכו') כדי שהם **נעולים** ולא ייווצרו מחדש בכל הרצה בצורה לא עקבית.
  **מייצר:** 3 קבצים קבועים — מכאן והלאה כל מודל משתמש בדיוק באותם splits, השוואה הוגנת מובטחת.

---

### Playbook Stage 3: Baseline

- [x] **7. Baseline מהיר**
  מודל טריוויאלי (לוגיסטית פשוטה, או "הקבוצה עם `pistol_round_win_rate` גבוה יותר מנצחת") על ה-Train/Val. לא צריך להיות טוב — צריך לתת רצפה למדידה.
  **מייצר:** מספר ייחוס (baseline log-loss/accuracy) שכל מודל מורכב יותר חייב לנצח כדי להצדיק את המורכבות.

---

### Playbook Stage 5: Modeling & Tuning

- [x] **8. אימון ואבחון XGBoost v1 (Baseline Features)**
  המודל אומן על Train ונבדק על Val בלבד. אבחון Bootstrap ברמת משחק הראה שרווחי הסמך של השיפור בלוג-לוס ובדיוק כוללים אפס; לכן v1 נסגרה ללא tuning, בהתאם להחלטת הצוות. **לא** נגעו ב-Test.
  **מייצר:** מספר v1 מתועד על Validation וקו בסיס להשוואת v2 מולו.

- [x] **9. Feature Factory שני — Match-Level Enrichment**
  חילוץ פרופיל-קבוצה סטטי מ-`cs2_newestcombinedmatches*.csv` ו-`newest_ts_ds.csv` (rating/ADR/KAST diffs, head2head, **opening kills**, rolling-5 form) — **אותה מתודולוגיה כמו ל-DNA**, כולל שלב inspection לפני שסומכים על העמודות (למדנו את הלקח עם באג ה-off-by-one). ממוזג ל-`final_tournament_features_v2.csv` באותה תשתית join-key/alias שכבר בנינו.
  **מייצר:** טבלת פיצ'רים מורחבת, מוכנה להשוואה ישירה מול v1.

- [x] **10. אימון XGBoost v2 והשוואה ישירה (על Validation בלבד)**
  אימון עם הפיצ'רים המורחבים, tuning על אותו Val. השוואת log-loss/Brier מול v1 **על אותו split בדיוק**. עדיין **לא** נוגעים ב-Test — עוד לא בחרנו גרסה סופית.
  **מייצר:** החלטה מבוססת-נתונים: v1 או v2 (או שילוב) — ומכאן זו הגרסה היחידה שתיגע ב-Test.

- [x] **11. הערכה סופית על Test (פעם אחת, לגרסה שנבחרה בלבד)**
  הרצה יחידה על 2026 Q2 (כולל IEM Cologne Major) — **רק** על המודל שניצח בשלב 10. log-loss, Brier, accuracy עם bootstrap CI, reliability diagram (calibration), confusion matrix.
  **מייצר:** התוצאה הרשמית — ומצביע בדיוק איפה המודל טועה (input לשלב הבא).

- [x] **12. ניתוח שגיאות ממוקד**
  התמקדות ב-rows עם n_DNA נמוך, קבוצות ללא DNA/enrichment כלל, ומפות עם ביצועים חלשים. עכשיו זה ניתוח על מודל שכבר מנצל את כל המקורות הידועים — אז מה שנשאר הוא באמת קשה, לא "שכחנו דאטה".
  **מייצר:** רשימת שיפורים ממוקדת ל-v3, לא ניחוש כללי.

---

### בניית שכבת הסימולציה (ייחודי לפרויקט הזה, לא ב-Playbook הכללי)

- [x] **13. Series Simulator (Map → Match)**
  שכבה נפרדת: לוקחת הסתברויות מפה מהמודל, מדמה (Monte Carlo, N≥1000) סדרת Bo1/Bo3/Bo5. אימות מול כמה תוצאות אמיתיות ידועות.
  **מייצר:** יכולת לחזות מאצ' שלם, לא רק מפה בודדת.

- [x] **14. Bracket Simulator (Match → Tournament)**
  קלט: רשימת משתתפים + מבנה ברקט של טורניר ספציפי. מריץ את ה-Series Simulator שוב ושוב כדי להפיק הסתברות לכל קבוצה להגיע לכל שלב / לנצח את הטורניר.
  **מייצר:** התוצר הסופי שהפרויקט הובטח — Tournament Simulator.

- [x] **15. Backtest על IEM Cologne Major 2026**
  הרצת הסימולטור המלא רטרואקטיבית על Cologne (בתוך ה-Test set שלנו) — בדיקה איכותית: האם המודל נתן ל-Falcons (המנצחת בפועל) הסתברות סבירה?
  **מייצר:** אימות end-to-end של כל הצינור, לא רק של המודל הבודד.

---

### Playbook Stage 6: Ensembling & Final Submission

- [x] **16. תיעוד תוצאות**
  עדכון `PROJECT_BRIEF.md` עם התוצאות בפועל (לא רק התוכנית), יצירת `RESULTS.md` עם מדדים, גרפים, מסקנות.
  **מייצר:** מסמך שאפשר להראות/להגיש, לא רק קוד.

- [x] **17. הרצת Live Prediction**
  הרצת הסימולטור המלא על טורניר אמיתי **שעוד לא קרה** (PGL Bucharest או IEM Beijing, בהתאם לתזמון) — שמירת התחזיות **לפני** שהתוצאות ידועות.
  **מייצר:** הבדיקה האמיתית היחידה שאי אפשר לזייף בדיעבד.

- [ ] **18. Retrospective ותכנון v3**
  אחרי שהתוצאות האמיתיות ידועות: השוואת תחזית מול מציאות, תיעוד מה עבד/לא עבד, רשימת v3 (rolling DNA, נתוני כלכלה, roster tracking).
  **מייצר:** סגירת מעגל + נקודת המשך לפרויקט הבא.
