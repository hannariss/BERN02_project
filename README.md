# Multinomial Logistic Regression for Stress Classification

**Authors:** Jule Grimm, Johanna Rissbacher  

## Problem description
The goal of the analysis is to find potential gender differences in stress patterns.
This was done by implementing multinomial logistic regression models (separatly for male and female participants) to classify stress levels into three categories:
- **Eustress (Positive Stress)** - Stress that motivates and enhances performance
- **No Stress** - Currently experiencing minimal to no stress  
- **Distress (Negative Stress)** - Stress that causes anxiety and impairs well-being

## Dataset

The analysis uses a dataset, which is based on a survey on stress and well-being factors among college students (ages 18–21) Available [here](https://www.kaggle.com/datasets/mdsultanulislamovi/student-stress-monitoring-datasets).
This dataset contains various psychological and physiological indicators. The target variable is stress type (Eustress, No stress, Distress), which is mapped to ordered categories for modeling purposes.

## Dependencies
See `requirements.txt` for exact package versions

## Methodology

### Data Wrangling
The data was wrangled in the file `data_wrangling.py`

### Feature Selection
To find the best predictors a model selection was performed:
1. **Forward Selection** 
2. **Backward Selection**
3. **Floating Selection**

To find a balance between high accuracy and low complexity, an additional hyperparameter tuning was performed to find the optimal number of features.

### Selected Models
- **Female Model**: 6 features selected via forward selection
- **Male Model**: 5 features selected via forward selection

### Parameter Estimation with Confidence Intervals
The project includes maximum likelihood estimation of the model parameters and their 95% confidence intervals

### Visualization
- Accuracy vs. number of features plots for both genders
- Coefficient plots with confidence intervals


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

## Results

The analysis demonstrates that:
- Gender-specific models perform better than a single combined model
- Different features are predictive for males vs. females
- Forward feature selection provides optimal model performance
- The models achieve good classification accuracy on the test set

## Technical Notes

- Uses Newton-CG solver for multinomial logistic regression optimization
- Implements custom statistical functions for confidence interval calculation
- Employs proper train-test splitting to avoid data leakage
- Uses cross-validation principles for model selection

