# BAS Lead Scoring

A small Python project that ranks sales leads by their likelihood of converting, built to work with the lead data tracked in the BAS CRM.

**Important:** BAS has no client history yet, so `sample_leads.csv` is synthetic data created by `generate_sample_data.py`. The outcomes are simulated using assumed relationships, so the results show that the pipeline works, not real business findings. Once real leads are recorded, the same code can be run on them.

## What it does
- Preprocesses lead features (service line, company size, source, contacts made, days since last contact, meeting held)
- Trains a logistic regression model (scikit-learn) on a train/test split
- Evaluates it against a majority-class baseline (accuracy and ROC-AUC)
- Outputs `ranked_leads.csv` (leads sorted by score 0-100) and `feature_importance.png`

## Run
pip install pandas scikit-learn matplotlib
python generate_sample_data.py
python score_leads.py

## Sample result
Accuracy 0.67 and ROC-AUC 0.71 vs a 0.59 baseline, on 400 simulated leads. The model beats the baseline modestly; with only 400 rows the coefficients are noisy and should not be over-interpreted.

## Next steps
Connect to real CRM exports, try tree-based models, and validate with cross-validation once there is enough real data.
