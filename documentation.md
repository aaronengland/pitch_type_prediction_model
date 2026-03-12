# Pitch Type Prediction Pipeline

## Overview

This project builds a machine learning pipeline to predict the next pitch type in Major League Baseball (MLB) using pitch-by-pitch data from the 2011 season. The task is a multi-class classification problem: given the game situation and pitcher/batter context known **prior to the pitch being thrown**, predict which of 8 pitch types will be thrown.

**Target variable**: `pitch_type` — one of FF (Four-seam Fastball), SL (Slider), SI (Sinker), FT (Two-seam Fastball), CH (Changeup), CU (Curveball), FC (Cutter), FS (Splitter).

**Data**: 698,318 pitches from 2,466 games across the 2011 MLB season (March 31 - October 28), involving 662 pitchers and 936 batters.

---

## 1. Exploratory Data Analysis (`01_eda/`)

### 1.1 Dataset Description

The raw dataset contains 718,961 rows and 125 columns. After filtering to non-null pitch types and the 8 main pitch types, 698,318 pitches remain. The overall missing rate across all columns is ~45%, primarily driven by runner-related columns (runners 4-7 are virtually never populated) and PITCHf/x measurement columns that are only available after the pitch is thrown.

### 1.2 Data Types

![Column Frequency by Data Type](01_eda/output/dtype_freq.png)

The dataset is predominantly numeric (float64 and int64 columns), with a smaller number of object (string) columns for identifiers, timestamps, and categorical fields.

### 1.3 Missing Data

![Proportion Missing by Column](01_eda/output/propna.png)

![Proportion Missing (Missing Columns Only)](01_eda/output/propna_missing_only.png)

Most missing values are concentrated in runner event columns (runners 4-7 are always null — rare baserunning scenarios) and PITCHf/x measurement columns. The features available prior to the pitch (the ones we use for modeling) have minimal missingness.

### 1.4 Target Distribution

![Pitch Type Distribution](01_eda/output/target_distribution.png)

The target is imbalanced: Four-seam Fastball (FF) dominates at ~34%, followed by Slider (SL) at ~16% and Sinker (SI) at ~13%. Splitter (FS) is the rarest at ~1.5%.

### 1.5 Pitch Type by Count

![Pitch Type by Count](01_eda/output/pitch_type_by_count.png)

Pitch selection changes dramatically by count:
- **Pitcher-ahead counts** (0-2, 1-2): More breaking balls (SL, CU) and off-speed pitches (CH).
- **Hitter-ahead counts** (3-0, 3-1): Predominantly fastballs (FF, FT, SI) to throw strikes.
- **Full count** (3-2): Mix shifts toward fastballs but breaking balls remain present.

### 1.6 Pitch Type by Handedness Matchup

![Pitch Type by Handedness](01_eda/output/pitch_type_by_handedness.png)

The platoon advantage significantly impacts pitch mix. Pitchers throw different combinations against same-hand vs. opposite-hand batters — e.g., more sliders to same-hand batters and more changeups to opposite-hand batters.

### 1.7 Pitch Type by Outs

![Pitch Type by Outs](01_eda/output/pitch_type_by_outs.png)

Pitch mix remains relatively stable across 0, 1, and 2 outs, with only minor shifts.

### 1.8 Pitcher Repertoire Diversity

![Pitch Types per Pitcher](01_eda/output/pitch_types_per_pitcher.png)

Most pitchers throw 3-5 distinct pitch types. A few specialists throw only 2, while some have a 6+ pitch repertoire.

### 1.9 Top Pitcher Pitch Mix

![Top Pitchers Mix](01_eda/output/top_pitchers_mix.png)

The top 20 pitchers by volume show highly individual pitch mixes — some are fastball-dominant (~60%+ FF), while others rely on sliders or sinkers as their primary pitch. This reinforces that **pitcher identity is the strongest predictor** of pitch type.

### 1.10 Pitch Type by Inning

![Pitch Type by Inning](01_eda/output/pitch_type_by_inning.png)

Pitch mix is relatively stable across innings 1-9, with subtle shifts as games progress (slightly more fastballs early).

### 1.11 Pitch Type by Runners on Base

![Pitch Type by Runners](01_eda/output/pitch_type_by_runners.png)

With runners on base, pitchers tend to slightly adjust their mix — e.g., more sinkers/two-seamers to induce ground balls for double plays.

### 1.12 Pitch Type by Pitcher Pitch Count

![Pitch Type by Pitch Count](01_eda/output/pitch_type_by_pitch_count.png)

As pitcher pitch count climbs through the game, the pitch mix shifts modestly. Pitchers may lean more on their primary pitch as fatigue sets in.

### 1.13 First Pitch vs. Later Pitches

![First Pitch vs Later](01_eda/output/first_pitch_vs_later.png)

Fastballs are overrepresented on the first pitch of an at-bat, while breaking balls and off-speed pitches become more common on subsequent pitches.

### 1.14 Temporal Coverage

![Pitches per Month](01_eda/output/pitches_per_month.png)

The season spans March through October, with April-September being the core months. October contains fewer pitches (postseason).

### 1.15 Baseline Accuracies

Three baselines were computed to set expectations:

| Baseline | Accuracy |
|---|---|
| Always predict FF (most common) | 34.2% |
| Predict each pitcher's most common pitch | 47.4% |
| Predict pitcher's most common pitch per count | 50.1% |

**Takeaway**: Pitcher identity + count alone gets to ~50% accuracy. The ML model needs to beat this to demonstrate value.

---

## 2. Data Split (`02_data_split/`)

### 2.1 Approach

An **out-of-time split** is used to simulate real-world deployment where the model trains on past data and predicts future pitches. Data is sorted chronologically and split by date:

| Split | Date Range | Rows | Percentage |
|---|---|---|---|
| Train | 2011-03-31 to 2011-08-22 | 538,294 | 77.1% |
| Validation | 2011-08-23 to 2011-09-21 | 120,479 | 17.3% |
| Test | 2011-09-22 to 2011-10-28 | 39,545 | 5.7% |

### 2.2 Distribution Stability

Pitch type distributions are consistent across splits (within 1-2 percentage points), confirming no major distributional drift within the season:

| Pitch Type | Train | Valid | Test |
|---|---|---|---|
| FF | 33.9% | 35.2% | 34.8% |
| SL | 15.8% | 15.7% | 15.2% |
| SI | 12.9% | 11.6% | 11.4% |
| FT | 11.5% | 11.8% | 11.9% |
| CH | 10.5% | 10.0% | 10.2% |
| CU | 8.0% | 8.1% | 8.5% |
| FC | 5.9% | 6.1% | 6.3% |
| FS | 1.5% | 1.5% | 1.6% |

---

## 3. Preprocessing (`03_preprocessing/`)

### 3.1 Architecture

Preprocessing follows a **class-based fit/transform pattern**:
1. Each preprocessing step is a class with `fit(X)` and `transform(X)` methods.
2. `fit` learns parameters from training data only (preventing data leakage).
3. `transform` applies the learned transformation to any split.
4. All fitted transformers are collected into a `PreprocessingModel` wrapper and serialized with `joblib`.

### 3.2 Transformer Pipeline

The transformers are applied in this order:

#### 3.2.1 SituationFeatures (stateless)
Creates game-situation features from pre-pitch columns:
- `score_diff`: Score differential from pitching team's perspective.
- `on_1b_flag`, `on_2b_flag`, `on_3b_flag`: Binary runner indicators.
- `runners_on`: Total runners on base.
- `same_hand`: Whether pitcher and batter share handedness (platoon indicator).
- `is_first_pitch`: Binary indicator for first pitch of at-bat.
- `b_height_inches`: Batter height converted from string (e.g., "6-2") to inches (e.g., 74).

#### 3.2.2 LagFeatures (stateless)
Creates within-at-bat pitch sequencing features:
- `prev_pitch_type`: The previous pitch thrown in this at-bat (`'NONE'` for first pitch).
- `prev_pitch_type_2`: The pitch thrown two pitches ago (`'NONE'` if unavailable).

#### 3.2.3 PitcherStats (stateful)
Learns pitcher-level aggregate statistics from training data:
- `pitcher_pct_{type}`: Each pitcher's overall pitch type distribution (8 columns).
- `pitcher_count_pct_{type}`: Pitcher's pitch distribution per ball-strike count (8 columns).
- `pitcher_total_pitches`: Total training pitches thrown by the pitcher.
- `pitcher_n_types`: Number of distinct pitch types in the pitcher's repertoire.

#### 3.2.4 PitcherHandednessStats (stateful)
Learns pitcher pitch mix conditioned on batter handedness:
- `pitcher_hand_pct_{type}`: Pitch distribution per (pitcher, same_hand) combination (8 columns).

#### 3.2.5 BatterStats (stateful)
Learns batter-level aggregate statistics from training data:
- `batter_pct_{type}`: Distribution of pitch types each batter faces (8 columns).
- `batter_total_pitches`: Total training pitches seen by the batter.

#### 3.2.6 CountCategory (stateless)
Creates a categorical count state feature:
- `count_category`: One of `first_pitch`, `ahead`, `behind`, `even`, `full_count`.

#### 3.2.7 PrepareFeatures (stateful)
Selects, encodes, and aligns the final feature matrix:
- **19 numeric features**: inning, top, outs, balls, strikes, fouls, pcount_at_bat, pcount_pitcher, score_diff, runner flags, same_hand, is_first_pitch, b_height_inches, pitcher_total_pitches, pitcher_n_types, batter_total_pitches.
- **32 mix features**: All pitcher_pct, pitcher_count_pct, pitcher_hand_pct, and batter_pct columns (filled with 0 for unseen entities).
- **5 categorical features** (one-hot encoded): p_throws, stand, prev_pitch_type, prev_pitch_type_2, count_category.

**Total features**: 78

---

## 4. Model (`04_model/`)

### 4.1 Algorithm

**XGBoost** with the native API (`xgb.train` + `DMatrix`), using `multi:softprob` objective for multi-class classification with calibrated probability outputs.

A `LabelEncoder` converts string pitch types (CH, CU, FC, FF, FS, FT, SI, SL) to integer labels 0-7 as required by XGBoost's `DMatrix`.

### 4.2 Hyperparameters

| Parameter | Value |
|---|---|
| objective | multi:softprob |
| num_class | 8 |
| eval_metric | mlogloss |
| max_depth | 4 |
| learning_rate | 0.05 |
| subsample | 0.7 |
| colsample_bytree | 0.5 |
| min_child_weight | 5 |
| reg_lambda | 1.0 |
| gamma | 0.1 |
| seed | 42 |
| num_boost_round | 3000 |
| early_stopping_rounds | 100 |

### 4.3 Training

The model trains on the training set with early stopping monitored on the validation set. The best iteration is selected by minimum validation `mlogloss`.

**Important implementation note**: In xgboost 2.x, `model.predict()` does **not** automatically use `best_iteration` after early stopping. All prediction calls must explicitly pass `iteration_range=(0, model.best_iteration + 1)` to avoid using the fully overfitted model.

### 4.4 Saved Artifacts

All artifacts are saved locally to `./output/`:
- `xgb_model.json`: The trained XGBoost model.
- `label_encoder.joblib`: The fitted LabelEncoder.
- `best_iteration.json`: Best iteration number for downstream inference.
- `test_predictions.csv`: Test set predictions (probabilities + predicted/actual labels).
- `params.json`: Training parameters and best iteration/score.

---

## 5. Model Evaluation (`05_model_eval/`)

### 5.1 Overall Metrics

| Metric | Value |
|---|---|
| Test Accuracy | 39.9% |
| Test Log-Loss | 1.7325 |
| Best Iteration | 13 |

### 5.2 Baseline Comparison

| Model | Accuracy | Lift vs. Best Baseline |
|---|---|---|
| Always predict FF | 34.8% | — |
| Pitcher mode | 45.6% | — |
| Pitcher mode per count | 47.2% | — |
| **XGBoost** | **39.9%** | **-7.2 pp** |

The model currently underperforms the pitcher-mode-per-count baseline. This is likely caused by a bug in the `model.predict()` call (missing `iteration_range`) that was identified and fixed but **not yet validated** by re-running the pipeline. The model stopped at iteration 13 with validation mlogloss of 1.9939, but predictions may have been made using all 113 trees (the fully overfitted model).

### 5.3 Confusion Matrix

![Confusion Matrix](05_model_eval/output/confusion_matrix.png)

Key observations:
- FF and SI have the highest recall (65% and 56%).
- Rare types (FS, CH, CU) have very low recall (<15%).
- The model tends to over-predict FF and SI at the expense of off-speed pitches.

### 5.4 Feature Importance

![Feature Importance](05_model_eval/output/feature_importance.png)

Top features by gain are the pitcher-level mix statistics (pitcher_pct_*, pitcher_count_pct_*), confirming that **pitcher identity and tendencies are the dominant predictors** of pitch type — consistent with EDA findings.

### 5.5 Calibration

![Calibration Plots](05_model_eval/output/calibration_plots.png)

Calibration plots show predicted probability vs. actual frequency by pitch type. Well-calibrated models should follow the diagonal.

### 5.6 Prediction Confidence

![Confidence Analysis](05_model_eval/output/confidence_analysis.png)

The confidence distribution and accuracy-by-confidence analysis show how reliable the model's predictions are at different confidence levels.

---

## 6. Next Steps

1. **Re-run the pipeline** with the `iteration_range` fix to validate corrected predictions.
2. **Hyperparameter tuning**: Systematic search (Bayesian optimization or grid search) over learning rate, max_depth, subsample, and colsample_bytree.
3. **Additional features**: Game-level momentum (recent AB outcomes), platoon-specific count tendencies, time-of-game features.
4. **Alternative models**: LightGBM, CatBoost, or neural network approaches that may capture sequential pitch patterns better.
5. **Ensemble**: Combine model predictions with the pitcher-mode-per-count baseline (e.g., weighted average or stacking).

---

## Project Structure

```
assessment_swish_analytics/
├── 00_data_collection/          # Raw data and metadata
│   ├── pitches (S3)
│   └── pitch_by_pitch_metadata.csv
├── 01_eda/                      # Exploratory data analysis
│   ├── notebook.ipynb
│   └── output/                  # Saved EDA figures
├── 02_data_split/               # Train/valid/test split
│   └── notebook.ipynb
├── 03_preprocessing/            # Feature engineering pipeline
│   ├── notebook.ipynb
│   └── output/                  # Pickled PreprocessingModel
├── 04_model/                    # XGBoost training
│   ├── notebook.ipynb
│   └── output/                  # Model, label encoder, predictions
├── 05_model_eval/               # Evaluation and diagnostics
│   ├── notebook.ipynb
│   └── output/                  # Evaluation figures
└── documentation.md             # This file
```
