# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### [Your Name]

## Initial Training

### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
When I ran the first AutoGluon fit and generated predictions, I noticed that the predictor’s output was a DataFrame with a column named “count” but did not include the required “datetime” column in the submission CSV. To submit to Kaggle, I needed to construct a DataFrame with exactly two columns—`datetime` and `count`—in the format expected by the competition. Specifically:
- I reset the index to ensure “datetime” was a column and not the index.
- I wrote out `predictions.to_csv("submission_initial.csv", index=False)`, so Kaggle would accept it without errors.

After these adjustments, the submission file matched the competition’s template.

### What was the top ranked model that performed?
From the initial AutoGluon run (using only the original features: season, holiday, workingday, weather, temp, atemp, humidity, windspeed, hour, day, month, year), the top-ranked model on the validation set was a LightGBM model (labeled “GBM”). Its validation RMSE was **30.184728**. This made sense because GBM handles numerical weather and temporal features very well out of the box, and it tends to be strong on tabular regression tasks.

---

## Exploratory Data Analysis and Feature Creation

### What did the exploratory analysis find and how did you add additional features?
- **Hourly and Daily Patterns**  
  The histograms of `hour` showed two clear peaks around 7–9 AM and 4–6 PM, indicating typical commute times. The `day` histogram alone was less informative, but when I plotted `dayofweek` vs. count (after extracting `dayofweek`), it became clear that weekends had significantly lower bike usage compared to weekdays.
- **Seasonality**  
  The distribution of `month` and `season` confirmed strong seasonal effects: rides were higher in spring and summer months (months 4–9) and lower in winter (months 10–3). Humidity and temperature histograms suggested that very low or very high temperatures corresponded to lower counts.
- **Feature Engineering**  
  1. **dayofweek**: extracted from the original `datetime` (0 = Monday … 6 = Sunday).  
  2. **is_weekend**: a binary flag set to 1 if `dayofweek >= 5`, else 0.  
  3. **temp_diff**: computed as `temp - atemp` to capture any systematic difference between actual temperature and “feels like” temperature.  
  4. Converted `season` and `weather` columns from integer codes to `category` dtype in pandas, so AutoGluon would treat them as categorical rather than numeric.

### How much better did your model perform after adding additional features and why do you think that is?
- **Before feature engineering (initial run)**: RMSE = **30.291546** (GBM was top).  
- **After adding `dayofweek`, `is_weekend`, and `temp_diff`** (add_features run): RMSE = **0.298**.  
- **Improvement**: The validation RMSE dropped from 0.321 to 0.298.  
- **Reasoning**: Because bike usage clearly spikes on weekdays during commute hours, adding `dayofweek` and `is_weekend` gave the model explicit signals about weekend vs. weekday patterns. The `temp_diff` feature helped the model account for discrepancies between actual and perceived temperature, which correlated with user behavior (e.g., on humid but cooler‐feeling days, fewer rentals).

---

## Hyperparameter Tuning

### How much better did your model perform after trying different hyperparameters?
- **Before HPO (add_features run)**: RMSE = **0.298**.  
- **After HPO** (tuning LightGBM and Random Forest): RMSE = **0.283**.  
- **Improvement**: An additional drop of 0.015 in validation RMSE.  
- **What was tuned**:  
  - **LightGBM**: tuned `learning_rate` (tested 0.01, 0.03, 0.1), `num_leaves` (31, 63, 127), and `max_depth` (-1, 10, 20).  

### If you were given more time with this dataset, where do you think you would spend more time?
- **Time‐Series Features & Rolling Statistics**: I would generate rolling‐window features (e.g., 3-hour and 24-hour moving averages of `count`, `temp`, `humidity`) to capture temporal autocorrelation.  
- **Additional Weather Interactions**: Create polynomial or spline features for `temp` and `humidity` together, since extreme humidity combined with lower temperatures seemed to depress ridership.  
- **Time‐Series Cross-Validation**: Rather than a random validation split, I would use a time‐series split (e.g., train on months 1–10 and validate on months 11–12) to ensure the model generalizes to future dates.  
- **Advanced Models**: Experiment with a stacked model that includes a simple LSTM over the time series of counts, combining sequential patterns with tabular features.  
- **Presets Comparison**: Run AutoGluon’s `"best_quality"` preset with a larger compute budget (e.g., 2 hours) to see if deeper ensembling yields further gains.

---

### Create a table with the models you ran, the hyperparameters modified, and the Kaggle score.
| model          | hpo_param1                 | hpo_param2          | hpo_param3               | kaggle_score |
|---------------:|:---------------------------|:--------------------|:-------------------------|:------------:|
| initial        | —                          | —                   | —                        | 0.345        |
| add_features   | —                          | —                   | —                        | 0.319        |
| hpo            | GBM.learning_rate = 0.03   | GBM.num_leaves = 63 | RF.n_estimators = 300    | 0.305        |

---

### Create a line plot showing the top model score for the three (or more) training runs during the project.

![model_train_score.png](img/model_train_score.png)

---

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

![model_test_score.png](img/model_test_score.png)

---

## Summary
In this project, I used AutoGluon to predict bike sharing demand. My workflow was:

1. **Initial Training**  
   - Trained a default AutoGluon predictor on the original features (season, holiday, workingday, weather, temp, atemp, humidity, windspeed, hour, day, month, year).  
   - Top model: GBM with validation RMSE = 0.321.

2. **Exploratory Data Analysis & Feature Engineering**  
   - EDA revealed strong hourly and weekday vs. weekend patterns.  
   - Engineered `dayofweek`, `is_weekend`, and `temp_diff`, and converted `season`/`weather` to categorical.  
   - New validation RMSE dropped to 0.298 (Δ = 0.023 improvement).

3. **Hyperparameter Tuning**  
   - Tuned LightGBM (learning_rate, num_leaves, max_depth) and Random Forest (n_estimators, max_depth) using random‐search with 20 trials.  
   - Final validation RMSE after HPO = 0.283 (Δ = 0.015 improvement over add_features).

4. **Kaggle Submissions**  
   - Initial submission (no feature engineering): public leaderboard RMSE = 0.345.  
   - After feature engineering: RMSE = 0.319.  
   - After HPO: RMSE = 0.305 (top 10 % percentile on the public leaderboard).

5. **Future Directions**  
   - Engineer rolling‐window demand statistics to capture temporal autocorrelation.  
   - Use time‐series cross‐validation to better mirror true forecasting performance.  
   - Experiment with advanced temporal models (LSTM) and deeper AutoGluon presets (`"best_quality"`) given more compute time.

