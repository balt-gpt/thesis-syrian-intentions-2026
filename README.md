# Predicting Return Intentions of Syrian Refugees in Europe Using Machine Learning

This repository contains the code for my Master's thesis in **Data Science and Society** at Tilburg University (2026).

## Project summary

The thesis investigates whether the short-term residence intentions of Syrian refugees and asylum-seekers in Europe can be predicted using supervised machine learning, in the period after the fall of the Assad regime in December 2024.

The outcome is a four-class problem: **Stay**, **Return**, **Move**, or **Unsure**. Three classification models are compared: multinomial logistic regression (baseline), Random Forest, and Gradient Boosting. The analysis explicitly addresses the severe class imbalance in the data (only 2.3% of respondents intend to return).

## Data

The data come from the **UNHCR Intentions Survey with Syrian Refugees and Asylum-Seekers in Europe — 2025** (3,738 respondents across Austria, Cyprus, Germany, the Netherlands, the United Kingdom, and other European countries).

**The dataset is not included in this repository.** It is owned by UNHCR and was shared under a restricted academic-use licence (License ID: 4980) that does not allow redistribution. Researchers wishing to replicate this work must request access through the [UNHCR Microdata Library](https://microdata.unhcr.org/).

The script expects a file named `hh_anonym.xlsx` placed in a `data/` folder at the project root.

## Requirements

- Python 3.13
- pandas 3.0.1
- NumPy 2.4.3
- scikit-learn 1.8.0
- Matplotlib 3.10.8
- SciPy

## How to run

After placing the dataset in `data/hh_anonym.xlsx`:

```bash
python thesis_code.py
```

The script performs preprocessing, hyperparameter tuning by five-fold cross-validation, model training, evaluation on a held-out test set, and subgroup error analysis. Output figures are saved to the `figures/` folder.

## Author

Bayram Altıntaş — Master's student in Data Science and Society, Tilburg University.



