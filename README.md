# Python-Analysis_AI_Learning_Buddy_Teaching-
GenAI Preservice Teacher Study — Python Reproducibility Files

This package contains the Python analysis code and generated statistical-output tables for the manuscript “Generative AI as a Learning Buddy and Teaching Assistant among Preservice Teachers.”

Software

The original analyses were conducted in IBM SPSS Statistics Version 23. For the manuscript revision, analyses were independently reproduced and extended in Python using NumPy, SciPy, statsmodels, and supporting numerical routines. G*Power 3.1.9.7 was used separately for the sample-size sensitivity analysis.

Data

The script expects a de-identified analysis file named:

GenAI_Reconciled_Cleaned_Data.csv

in the same working directory as the script. The participant-level dataset is not included in this public-code package. Availability should follow the study's ethics approval and institutional requirements.

Run

python genai_reproducible_analysis.py

The script writes output tables to the analysis_outputs/ directory.

Analyses reproduced in Python

Task frequencies and participant-based percentages

Learning versus instructional-preparation frequency descriptives

Wilcoxon signed-rank test and effect size

Spearman correlation

Level 400 sensitivity analysis

Attitude descriptives and Cronbach alpha

KMO and Bartlett test

Parallel analysis

Principal-axis factoring with oblimin rotation

EFA pattern matrix, communalities, retained-factor common variance, and factor-correlation matrix

HC3 multiple linear regression

VIF, Breusch–Pagan, Jarque–Bera, Ramsey RESET, and Cook's-distance diagnostics

Proportional-odds ordinal logistic regression

Brant proportional-odds tests implemented from nested cumulative binary logits with cross-threshold covariance

Threshold-specific binary logistic sensitivity models

Institution-adjusted sensitivity models and institution block tests

Key Python verification results added for the revision

EFA retained-factor common variance: 47.177% (reported as 47.2%)

Factor 1–Factor 2 correlation: 0.397 (reported as .40)

Learning-use competence Brant test: chi-square(3) = 11.046, p = .011

Instructional-preparation-use competence Brant test: chi-square(3) = 5.620, p = .132

Learning-use omnibus Brant test: chi-square(21) = 28.981, p = .114

Instructional-preparation-use omnibus Brant test: chi-square(21) = 92.218, p < .001

Competence remained positive and statistically significant at every cumulative threshold:

Learning-use ORs: 2.10–10.12

Instructional-preparation-use ORs: 2.77–5.84

Maximum VIF across linear models: 1.34

Breusch–Pagan p-values: .519–.878

Ramsey RESET p-values: .114–.870

Maximum Cook's distance: < .09

Competence-adjusted teaching-support Jarque–Bera p = .003

Institution block tests were nonsignificant in all four sensitivity models (all p > .20)

Notes

Factor signs are arbitrary in exploratory factor analysis. The script orients factor signs to match the manuscript's positive presentation and applies the same orientation to the oblique factor-correlation matrix.
