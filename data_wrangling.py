import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load data

df = pd.read_csv(os.path.join('data', 'Stress_Dataset.csv'))
df = df.drop('Have you been dealing with anxiety or tension recently?.1', axis=1)

# keywords for column titles 
keys = [
    "recent_stress",
    "heartbeat",
    "anxiety",
    "sleep",
    "headaches",
    "irritation",
    "concentration",
    "sadness",
    "illness",
    "isolation",
    "workload",
    "competition",
    "relationships",
    "teacher_diff",
    "work_env",
    "relaxation",
    "home_env",
    "confidence_performance",
    "lack_conf_subjects",
    "activities",
    "attendance",
    "weight",
    "stresstype"
]

df = df.rename(columns=dict(zip(df.columns[2:25], keys)))

# Same order as in kaggle
cols_ordered = [
    "Gender",
    "Age",
    "recent_stress",
    "heartbeat",
    "anxiety",
    "sleep",
    "concentration",
    "sadness",
    "irritation",
    "isolation",
    "headaches",
    "illness",
    "weight",
    "workload",
    "competition",
    "confidence_performance",
    "lack_conf_subjects",
    "activities",
    "attendance",
    "teacher_diff",
    "work_env",
    "home_env",
    "relationships",
    "relaxation",
    "stresstype"
]

df = df[cols_ordered]
df.to_csv(os.path.join('data', 'Stress_wrangled.csv'), index=False)

