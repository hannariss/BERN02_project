# Multinomial Logistic Regression for Stress Classification

**Authors:** Jule Grimm, Johanna Rissbacher  
**Date:** 21/10  
**Project:** BERN02

## Overview

This project implements multinomial logistic regression models to classify stress levels into three categories:
- **Eustress (Positive Stress)** - Stress that motivates and enhances performance
- **No Stress** - Currently experiencing minimal to no stress  
- **Distress (Negative Stress)** - Stress that causes anxiety and impairs well-being

The analysis is performed separately for male and female participants to account for potential gender differences in stress patterns.

## Dataset

The analysis uses a stress dataset (`Stress_wrangled.csv`) containing various psychological and physiological indicators. The target variable is stress type, which is mapped to ordered categories for modeling purposes.

## Methodology

### Data Preparation
- Renamed target column from `stresstype` to `target`
- Mapped stress categories to ordered numerical values (0, 1, 2)
- Split data by gender (male/female) for separate modeling
- Applied 80/20 train-test split for model evaluation

### Feature Selection
The project employs sequential feature selection to identify the most predictive features:

1. **Forward Selection** - Adds features one by one based on improvement
2. **Backward Selection** - Removes features one by one based on performance
3. **Floating Selection** - Combines forward and backward steps for optimal selection

### Hyperparameter Tuning
- Tested upper boundaries for feature selection (1-10 features)
- Evaluated model performance using accuracy scores
- Selected optimal number of features based on test set performance

### Model Selection
- **Female Model**: 6 features selected via forward selection
- **Male Model**: 5 features selected via forward selection
- Used AIC (Akaike Information Criterion) to compare competing models

## Key Features Selected

### Female Model (6 features)
- `recent_stress`
- `anxiety`
- `sadness`
- `headaches`
- `weight`
- `work_env`

### Male Model (5 features)
- `recent_stress`
- `heartbeat`
- `sadness`
- `workload`
- `lack_conf_subjects`

## Statistical Analysis

### Confidence Intervals
The project includes comprehensive statistical analysis:
- Computed 95% confidence intervals for all model coefficients
- Used Hessian-based standard error estimation
- Implemented custom softmax and standard error calculation functions

### Visualization
- Accuracy vs. number of features plots for both genders
- Coefficient plots with confidence intervals
- Color-coded visualization for different stress categories

## Dependencies

```python
pandas
numpy
scikit-learn
seaborn
matplotlib
mlxtend
```

## File Structure

```
├── multinomial_regression.ipynb    # Main analysis notebook
├── data/
│   ├── Stress_wrangled.csv         # Input dataset
│   └── data_process/
│       ├── female_accuracies.csv   # Female model accuracy results
│       └── male_accuracies.csv     # Male model accuracy results
└── README.md                       # This file
```

## Usage

1. Ensure all dependencies are installed
2. Place the `Stress_wrangled.csv` file in the `data/` directory
3. Run the notebook cells sequentially
4. The analysis will generate accuracy plots and coefficient visualizations

## Results

The analysis demonstrates that:
- Gender-specific models perform better than a single combined model
- Different features are predictive for males vs. females
- Forward feature selection provides optimal model performance
- The models achieve good classification accuracy on the test set

## Technical Notes

- Uses Newton-CG solver for logistic regression optimization
- Implements custom statistical functions for confidence interval calculation
- Employs proper train-test splitting to avoid data leakage
- Uses cross-validation principles for model selection

## Future Work

Potential extensions could include:
- Cross-validation for more robust model evaluation
- Additional feature engineering
- Ensemble methods for improved performance
- Real-time stress prediction applications
