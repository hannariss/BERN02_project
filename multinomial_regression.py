import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

stress_data = pd.read_csv('data/Stress_wrangled.csv')

stress_data.rename(columns={'stresstype': 'target'}, inplace=True)
label_encoder = sk.preprocessing.LabelEncoder()

# Fit and transform
stress_data['target_encoded'] = label_encoder.fit_transform(stress_data['target'])

# define model
multi_reg = LogisticRegression(solver='newton-cg', max_iter=1000)

X = stress_data.drop(columns=['target', 'target_encoded'])
y = stress_data['target_encoded']

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

multi_reg.fit(X_train, y_train)
pred_y = multi_reg.predict(X_test)

# inspect coefficients
classes = multi_reg.classes_
# Coefficients (one row per class)
coef_df = pd.DataFrame(multi_reg.coef_, columns=X.columns, index=classes) 
#column shows how strongly that predictor affects the odds of belonging to that class.
