# BAS Lead Scoring

A Python project that gives each sales lead a score from 0 to 100 for how likely it is to become a client. I made it for BAS Innovations, to get hands-on with machine learning on a business problem.

## Important: the data is fake
BAS doesn't have any real clients yet, so I generated 400 made-up leads with `generate_sample_data.py`. I decided how the outcomes would depend on things like lead source and meetings held, so the results only show that my code works. They don't tell us anything real about BAS. When real leads exist, the same code can run on them.

## What it does
1. Takes lead details: service line, company size, source, number of contacts, days since last contact, and whether a meeting happened
2. Trains a logistic regression model with scikit-learn
3. Tests it on leads it hasn't seen, and compares it to simply guessing the most common outcome
4. Saves a ranked list (`ranked_leads.csv`) and a chart of which factors matter (`feature_importance.png`)

## Run it
```
pip install pandas scikit-learn matplotlib
python generate_sample_data.py
python score_leads.py
```

## Results
Accuracy was 0.67 against 0.59 for the guessing baseline, and ROC-AUC was 0.71. So it's better than guessing, but not by a lot. With only 400 rows I wouldn't trust the individual factors in the chart much. For example, "Event" scored higher than "Referral", which is mostly random noise from the small sample.

## What's next
Use real CRM data when there is some, try a random forest, and use cross-validation instead of a single split.
