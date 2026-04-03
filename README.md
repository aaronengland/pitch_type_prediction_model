# Pitch Type Prediction Model

## About This Project

In this project, I built a model to predict **what type of pitch a baseball pitcher will throw next** — and how likely each option is. Using pitch-by-pitch data from the entire 2011 MLB season, my model looks at the current game situation (the count, who's batting, what inning it is, etc.) and outputs a probability for each of 8 pitch types. For example, it might say: "There's a 42% chance the next pitch is a fastball, a 22% chance it's a slider, a 15% chance it's a changeup..." and so on.

Rather than simply guessing one pitch type, my model provides a **full set of probabilities** — giving a much richer picture of what to expect.

I organized each step of the analysis into its own folder with a Python notebook and an accompanying HTML export for easy review. The text within each notebook explains my analysis, graphs, and decisions at each stage.

### Pitch Type Abbreviations

Throughout this project, pitch types are referred to by their standard abbreviations:

| Abbreviation | Pitch Type |
|:---:|:---|
| **FF** | Four-seam Fastball |
| **SL** | Slider |
| **SI** | Sinker |
| **FT** | Two-seam Fastball |
| **CH** | Changeup |
| **CU** | Curveball |
| **FC** | Cutter |
| **FS** | Splitter |

### Dataset at a Glance

| Metric | Value |
|:---|:---|
| Total pitches | 698,318 |
| Games | 2,466 |
| Season window | March 31 – October 28, 2011 |
| Pitchers | 662 |
| Batters | 936 |

---

## 1. Exploring the Data

> *All exploratory work lives in the `01_eda/` directory.*

Before building any model, my first step was to understand the data — what it contains, what's missing, and what patterns are already visible.

### 1.1 What's in the Dataset?

The raw dataset contains 718,961 pitches and 125 columns of information about each one. After I removed pitches with missing or uncommon pitch types, 698,318 pitches across the 8 main types remained.

About 45% of data points across all columns are missing, but this is not as alarming as it sounds. Most of the missing values come from columns that are rarely relevant (e.g., data about the 5th, 6th, or 7th baserunner in an inning — an extremely rare scenario) and from sensor measurements that are only recorded *after* the pitch is thrown. The information available *before* the pitch — which is what I use for modeling — is largely complete.

### 1.2 Data Types

![Column Frequency by Data Type](01_eda/output/dtype_freq.png)

Most columns contain numbers (scores, counts, measurements), with a smaller number containing text (player names, dates, categories).

### 1.3 Missing Data

![Proportion Missing by Column](01_eda/output/propna.png)

![Proportion Missing — Missing Columns Only](01_eda/output/propna_missing_only.png)

These charts show which columns have missing values and how much is missing. The columns I used for prediction have very little missing data.

### 1.4 How Often Is Each Pitch Thrown?

![Pitch Type Distribution](01_eda/output/target_distribution.png)

The pitch types are not evenly distributed. Four-seam fastballs (FF) are by far the most common at ~34% of all pitches, while splitters (FS) are the rarest at ~1.5%. This imbalance is important — a model could appear to be 34% accurate just by guessing "fastball" every time, so I needed to keep this in mind when evaluating performance.

### 1.5 Pitch Type by Count

> *The "count" is the current number of balls and strikes in an at-bat. For example, a 3-2 count means 3 balls and 2 strikes — a "full count."*

![Pitch Type by Count](01_eda/output/pitch_type_by_count.png)

Pitch selection changes dramatically depending on the count:

- **When the pitcher is ahead** (e.g., 0-2, 1-2) — Pitchers can afford to throw trickier pitches like sliders and curveballs, since they have strikes to spare.
- **When the hitter is ahead** (e.g., 3-0, 3-1) — Pitchers mostly throw fastballs because they need to throw a strike and avoid walking the batter.
- **Full count** (3-2) — The mix shifts toward fastballs, but breaking pitches are still in play.

### 1.6 Pitch Type by Handedness Matchup

> *In baseball, batters tend to perform better against pitchers who throw from the opposite side (e.g., a left-handed batter vs. a right-handed pitcher). This is called the "platoon advantage," and it influences which pitches a pitcher chooses to throw.*

![Pitch Type by Handedness](01_eda/output/pitch_type_by_handedness.png)

Pitchers adjust their pitch selection based on whether the batter hits from the same side or the opposite side. For example, pitchers tend to throw more sliders to same-side batters and more changeups to opposite-side batters.

### 1.7 Pitch Type by Outs

![Pitch Type by Outs](01_eda/output/pitch_type_by_outs.png)

The number of outs (0, 1, or 2) has only a minor effect on pitch selection.

### 1.8 How Many Pitch Types Does Each Pitcher Throw?

![Pitch Types per Pitcher](01_eda/output/pitch_types_per_pitcher.png)

Most pitchers throw 3–5 distinct pitch types. A few specialists rely on just 2, while some have a 6+ pitch arsenal.

### 1.9 Pitch Mix of Top Pitchers

![Top Pitchers Mix](01_eda/output/top_pitchers_mix.png)

The top 20 pitchers by volume show highly individual pitch mixes. Some are fastball-dominant (60%+ fastballs), while others rely heavily on sliders or sinkers. This highlights a key insight: **knowing who the pitcher is tells you a lot about what pitch is coming next**.

### 1.10 Pitch Type by Inning

![Pitch Type by Inning](01_eda/output/pitch_type_by_inning.png)

Pitch mix is relatively stable across innings 1–9, with only subtle shifts as games progress (slightly more fastballs early on).

### 1.11 Pitch Type by Runners on Base

![Pitch Type by Runners](01_eda/output/pitch_type_by_runners.png)

With runners on base, pitchers tend to slightly adjust their mix — for example, throwing more sinkers and two-seam fastballs to try to induce ground balls for double plays.

### 1.12 Pitch Type by Pitcher Fatigue

![Pitch Type by Pitch Count](01_eda/output/pitch_type_by_pitch_count.png)

As a pitcher throws more pitches during a game, their pitch mix shifts modestly. Pitchers may lean more on their go-to pitch as fatigue sets in.

### 1.13 First Pitch vs. Later Pitches

![First Pitch vs Later](01_eda/output/first_pitch_vs_later.png)

Fastballs are disproportionately common on the first pitch of an at-bat, while breaking balls and off-speed pitches appear more frequently on later pitches.

### 1.14 Temporal Coverage

![Pitches per Month](01_eda/output/pitches_per_month.png)

The season spans March through October, with April–September as the core months. October contains fewer pitches (postseason only).

### 1.15 Simple Baselines

Before building my machine learning model, I established simple "baselines" — straightforward rules of thumb to set expectations for what counts as good performance:

| Baseline Strategy | Accuracy |
|:---|:---:|
| Always guess fastball (the most common pitch) | 34.2% |
| Always guess each pitcher's most common pitch | 47.4% |
| Guess each pitcher's most common pitch for the current count | 50.1% |

**Key takeaway:** Just knowing who the pitcher is and what the count is gets you to ~50% accuracy. This set a high bar for my model — I needed to either beat these simple rules or add value in other ways (like providing well-calibrated probabilities rather than a single guess).

---

## 2. Splitting the Data

> *All data split work lives in the `02_data_split/` directory.*

### 2.1 Why the Split Matters

To honestly evaluate a prediction model, you can't test it on the same data it learned from — that would be like letting a student see the exam answers before the test. The model needs to be tested on data it has never seen.

I also needed to avoid a subtler problem: **look-ahead bias**. If the model trains on September games and is then tested on August games, it could unknowingly benefit from patterns that hadn't yet emerged in August. To prevent this, I split the data **chronologically** — the model trains on earlier games and is tested on later ones, just like a real prediction system would work:

| Split | Date Range | Pitches | Percentage |
|:---|:---|:---:|:---:|
| Training | March 31 – August 22, 2011 | 538,294 | 77.1% |
| Validation | August 23 – September 21, 2011 | 120,479 | 17.3% |
| Test | September 22 – October 28, 2011 | 39,545 | 5.7% |

- **Training set** — The model learns from this data.
- **Validation set** — Used during training to monitor whether the model is improving or starting to memorize. Helps decide when to stop training.
- **Test set** — Held out entirely until the very end. Provides the final, unbiased measure of model performance.

### 2.2 Distribution Stability

The pitch type distributions are consistent across all three splits (within 1–2 percentage points), confirming that pitch behavior didn't change drastically over the season:

| Pitch Type | Training | Validation | Test |
|:---:|:---:|:---:|:---:|
| FF | 33.9% | 35.2% | 34.8% |
| SL | 15.8% | 15.7% | 15.2% |
| SI | 12.9% | 11.6% | 11.4% |
| FT | 11.5% | 11.8% | 11.9% |
| CH | 10.5% | 10.0% | 10.2% |
| CU | 8.0% | 8.1% | 8.5% |
| FC | 5.9% | 6.1% | 6.3% |
| FS | 1.5% | 1.5% | 1.6% |

---

## 3. Preparing the Data for the Model

> *All preprocessing work lives in the `03_preprocessing/` directory.*

Raw data rarely comes in a form that a machine learning model can use directly. In this section, I describe how I transformed the raw pitch data into a clean set of **features** — the specific pieces of information my model uses to make predictions.

### 3.1 Avoiding Data Leakage

**Data leakage** is one of the most important pitfalls in machine learning. It happens when information that wouldn't be available at prediction time accidentally sneaks into the model during training, making the model appear better than it really is.

To prevent this, I ensured that every transformation that learns something from the data (like computing a pitcher's average pitch mix) is learned **only from the training set**. I then applied those learned values identically to the validation and test sets — ensuring the model never gets an unfair advantage.

### 3.2 Smoothing Pitcher and Batter Statistics

When computing pitcher-level statistics (like "this pitcher throws 40% fastballs"), pitchers with very few pitches in the training data can produce unreliable statistics. A pitcher who has thrown just 5 pitches might appear to throw 100% fastballs simply by chance.

To handle this, I use **Bayesian smoothing** — a technique that blends each pitcher's individual statistics with the league-wide average. Pitchers with many pitches keep their true tendencies, while pitchers with few pitches are pulled toward the league average. This also provides sensible defaults for pitchers who appear in the validation or test sets but weren't in the training data (e.g., September call-ups).

### 3.3 Features Used by the Model

My model uses approximately 90 features, organized into several categories:

#### Game Situation Features

Information about the current state of the game that any fan watching could observe:

- **Score difference** — Is the pitching team ahead or behind, and by how much?
- **Runners on base** — Which bases are occupied? Are the bases loaded?
- **Handedness matchup** — Are the pitcher and batter on the same side (e.g., both right-handed) or opposite sides?
- **First pitch of the at-bat** — Pitchers often approach the first pitch differently.
- **Batter height** — Taller and shorter batters present different strike zones.

#### Pitch Sequencing Features

What happened earlier in this at-bat:

- **Previous pitch type** — What was the last pitch thrown in this at-bat?
- **Two pitches ago** — What was thrown two pitches back? Pitchers often avoid repeating the same pitch multiple times in a row.

#### Game Context Features

Broader context about the game state:

- **Total runs scored** — How many runs have been scored in the game so far?
- **Close game indicator** — Is the score within 2 runs? Pitchers tend to be more careful in tight games.
- **Late inning indicator** — Is it the 7th inning or later? Late-game situations often shift pitch selection.

#### Pitcher Tendency Features

Historical statistics about how each pitcher tends to pitch, which I computed from the training data:

- **Overall pitch mix** — What percentage of each pitch type does this pitcher throw?
- **Pitch mix by count** — How does this pitcher's mix change in different counts?
- **Pitch mix by handedness** — How does this pitcher adjust against same-side vs. opposite-side batters?
- **Pitch mix by handedness and count** — The most detailed breakdown: how does this pitcher pitch against a specific handedness matchup in a specific count?
- **Number of pitch types** — How many different pitches does this pitcher throw?

#### Batter Tendency Features

- **Pitch mix faced** — What types of pitches does this batter typically see? Some batters attract more breaking balls than others.

#### Count Category

A simplified grouping of the ball-strike count into categories: first pitch, pitcher ahead, hitter ahead, even count, or full count.

---

## 4. Building the Model

> *All model training work lives in the `04_model/` directory.*

### 4.1 Algorithm

I used **XGBoost** (eXtreme Gradient Boosting), a widely-used machine learning algorithm that builds predictions by combining many simple decision trees. Think of each decision tree as a flowchart of yes/no questions about the game situation (e.g., "Is the count 0-2?" → "Is the pitcher right-handed?" → "Was the last pitch a slider?"). XGBoost combines hundreds of these small flowcharts together to make a final prediction.

Crucially, my model doesn't just predict a single pitch type — it outputs **probabilities for all 8 pitch types** on every prediction. This gives a much more useful and nuanced answer than simply saying "fastball."

### 4.2 Model Settings

Machine learning models have settings (called **hyperparameters**) that control how the model learns. I set these conservatively to prioritize a model that generalizes well to new data over one that memorizes the training data:

| Setting | Value | What It Controls |
|:---|:---:|:---|
| Learning rate | 0.03 | How quickly the model adapts — slower is more careful |
| Max tree depth | 3 | How many questions each decision tree can ask — shallower trees are simpler |
| Min samples per leaf | 50 | How much data a decision tree needs before making a conclusion |
| Row sampling | 70% | Each tree only sees 70% of the data, reducing overfitting |
| Feature sampling | 50% | Each tree only considers half the features, adding diversity |
| Split threshold | 0.5 | How much improvement is needed before adding a new question |
| Regularization | 5 | Penalizes overly complex trees to keep predictions stable |

With more time, these settings could be systematically optimized to improve performance.

### 4.3 Training Process

A common problem in machine learning is **overfitting** — when the model memorizes the training data so well that it performs poorly on new, unseen data. It's like a student who memorizes practice exam answers word-for-word but can't answer rephrased questions.

To prevent this, I used **early stopping**: during training, my model continuously checks its performance on the validation set (data it hasn't trained on). If the validation performance stops improving for 100 consecutive rounds, training automatically stops. This ensures the model learns useful patterns without memorizing noise.

### 4.4 Saved Outputs

I saved the trained model and supporting files to the `04_model/output/` directory for use in evaluation.

---

## 5. Evaluating the Model

> *All evaluation work lives in the `05_model_eval/` directory.*

### 5.1 Does the Model Generalize?

The most important question I needed to answer is: does my model perform similarly on data it has seen (training set) versus data it hasn't (validation and test sets)? A large gap would suggest the model has memorized rather than learned.

| Data Split | Accuracy | Log-Loss (lower is better) |
|:---|:---:|:---:|
| Training | 41.0% | 1.694 |
| Validation | 30.5% | 1.908 |
| Test | 37.1% | 1.766 |

> *Log-loss measures how well the predicted probabilities match reality. A lower score means the model's probability estimates are more accurate.*

There is a ~10 percentage point gap between training and validation accuracy. Part of this is explained by timing: the validation period (late August to mid-September) coincides with MLB's September roster expansion, which introduces many new pitchers the model hasn't seen before. The test set (late September through October postseason) contains more established pitchers, which is why test accuracy (37.1%) is higher than validation (30.5%).

### 5.2 How Does It Compare to Simple Strategies?

| Strategy | Accuracy |
|:---|:---:|
| Always guess fastball | 34.8% |
| Guess each pitcher's most common pitch | 45.6% |
| Guess each pitcher's most common pitch for the current count | 47.2% |
| **Machine learning model** | **37.1%** |

My model outperforms the simplest strategy (always guess fastball) by +2.3 percentage points, but it falls short of the pitcher-specific lookup strategies. Here's why:

The pitcher-specific strategies work by simply memorizing each pitcher's tendencies from historical data — and knowing *who* the pitcher is turns out to be the single most powerful predictor of what pitch is coming. I did compute pitcher tendency features during preprocessing, but including them in the model caused severe overfitting (the training-validation gap ballooned to ~25%). After I removed those features, my model relies on game-situation information (count, handedness, inning, runners, pitch sequencing) that applies broadly across all pitchers — useful information, but not enough to compensate for losing pitcher identity.

**What I would do differently in practice:** The ideal approach would be to **combine** both methods — use the pitcher-specific lookup to capture *what each pitcher tends to throw*, and layer the machine learning model on top to capture *how the game situation shifts that mix*. I believe this combined approach would outperform either method alone.

### 5.3 Confusion Matrix

> *A confusion matrix is a grid that shows how often the model correctly and incorrectly predicted each pitch type. The rows represent what was actually thrown, and the columns represent what the model predicted. A perfect model would have all its counts along the diagonal (top-left to bottom-right).*

![Confusion Matrix](05_model_eval/output/confusion_matrix.png)

Key takeaways:

- My model is best at identifying **four-seam fastballs (FF)** and **sinkers (SI)** — the two most common pitch types.
- For **rarer pitch types** (splitters, changeups, curveballs), my model struggles and tends to default to predicting fastball. Without pitcher-specific information, it doesn't have enough context to distinguish these less common pitches.
- Overall, the model **over-predicts fastballs** — a natural consequence of fastballs being the most frequent pitch in the dataset.

### 5.4 Feature Importance

![Feature Importance](05_model_eval/output/feature_importance.png)

This chart shows which pieces of information my model relies on most heavily when making predictions. The most influential features relate to the count and game situation.

### 5.5 Calibration

> *Calibration answers the question: "When the model says there's a 40% chance of a fastball, does a fastball actually happen about 40% of the time?" A well-calibrated model means you can trust its probabilities at face value.*

![Calibration Plots](05_model_eval/output/calibration_plots.png)

These plots compare my model's predicted probabilities against actual outcomes for each pitch type. Points that fall along the diagonal line indicate well-calibrated predictions.

### 5.6 Prediction Confidence

![Confidence Analysis](05_model_eval/output/confidence_analysis.png)

This chart explores the relationship between **how confident my model is** in a prediction and **how often it's correct**. Ideally, high-confidence predictions should be right more often than low-confidence ones.

---

## 6. Next Steps

With additional time, these are the improvements I would prioritize, roughly in order of expected impact:

1. **Better pitcher identity features** — This is the single highest-impact improvement. My current model struggles because it can't reliably use pitcher-specific tendencies without overfitting. I would explore more sophisticated techniques (like target encoding with safeguards or learned pitcher representations) to safely incorporate this critical information.

2. **Combine my model with simple lookups** — I would merge the pitcher-specific lookup strategy (which knows *what* each pitcher tends to throw) with my machine learning model (which knows *when* the game situation changes that tendency). I believe this combined approach would outperform either method alone.

3. **Optimize model settings** — I would systematically search for better hyperparameter values rather than setting them manually. This is a standard step that often yields meaningful improvements.

4. **Try alternative algorithms** — I would experiment with other machine learning algorithms (LightGBM, CatBoost, or neural networks) that may be better suited to capturing sequential pitch patterns.

5. **Engineer additional features** — I would add information about pitcher fatigue relative to their typical workload, game momentum from recent score changes, day vs. night games, and ballpark effects.

6. **More robust evaluation** — I would use rolling-window cross-validation (testing on multiple time periods rather than just one) for more reliable performance estimates.

---

## How to Review

Each numbered directory contains:

- A **Jupyter notebook** (`.ipynb`) with the full analysis, code, and explanations
- An **HTML export** (`.html`) for reviewing the analysis in a web browser without needing to run any code

I recommend opening the HTML files in a browser for the easiest reading experience.

---

## Project Structure

```
pitch_type_prediction_model/
├── 00_data_collection/          # Raw data and metadata
│   ├── pitches (S3)
│   └── pitch_by_pitch_metadata.csv
├── 01_eda/                      # Exploring the data
│   ├── notebook.ipynb
│   ├── notebook.html            # HTML export for easy review
│   └── output/                  # Saved charts and figures
├── 02_data_split/               # Splitting data chronologically
│   ├── notebook.ipynb
│   └── notebook.html
├── 03_preprocessing/            # Preparing data for the model
│   ├── notebook.ipynb
│   ├── notebook.html
│   └── output/                  # Saved preprocessing pipeline
├── 04_model/                    # Building and training the model
│   ├── notebook.ipynb
│   ├── notebook.html
│   └── output/                  # Trained model and predictions
├── 05_model_eval/               # Evaluating model performance
│   ├── notebook.ipynb
│   ├── notebook.html
│   └── output/                  # Evaluation charts and figures
└── README.md                    # This file
```
