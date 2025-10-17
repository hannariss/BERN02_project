import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


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

multi_reg.fit(X, y)
pred_y = multi_reg.predict(X)

# inspect coefficients
classes = multi_reg.classes_
# Coefficients (one row per class)
coef_df_1 = pd.DataFrame(multi_reg.coef_, columns=X.columns, index=classes) 
#column shows how strongly that predictor affects the odds of belonging to that class.


# Different target codes
target_order = {"Eustress (Positive Stress) - Stress that motivates and enhances performance.": 0, "No Stress - Currently experiencing minimal to no stress.": 1, "Distress (Negative Stress) - Stress that causes anxiety and impairs well-being.": 2}
stress_data["target_ordered"] = stress_data["target"].map(target_order)

# Split in male and female
female = stress_data[stress_data['Gender'] == 1]
male = stress_data[stress_data['Gender'] == 0]

X_f = female.drop(columns=['Gender', 'target', 'target_encoded', 'target_ordered'])
y_f = female['target_ordered']

X_m = male.drop(columns=['Gender', 'target', 'target_encoded', 'target_ordered'])
y_m = male['target_ordered']

# split data
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_f, y_f, test_size=0.2, random_state=42)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_m, y_m, test_size=0.2, random_state=42)

# define model
multi_reg_f = LogisticRegression(solver='newton-cg', max_iter=1000)
multi_reg_m = LogisticRegression(solver='newton-cg', max_iter=1000)

# multi_reg_f.fit(X_train_f, y_train_f)
# pred_y_f = multi_reg_f.predict(X_test_f)

# multi_reg_m.fit(X_train_m, y_train_m)
# pred_y_m = multi_reg_m.predict(X_test_m)


features_f = SequentialFeatureSelector(multi_reg_f)
features_m = SequentialFeatureSelector(multi_reg_m)

features_f.fit(X_train_f, y_train_f)
all_features_f = features_f.feature_names_in_
selected_f = all_features_f[features_f.support_]
print(selected_f)

features_m.fit(X_train_m, y_train_m)
all_features_m = features_m.feature_names_in_
selected_m = all_features_m[features_m.support_]
print(selected_m)

sfs_f = SFS(multi_reg_f, k_features='best', forward=True)
sfs_f = sfs_f.fit(X_train_f, y_train_f)
results_f = pd.DataFrame.from_dict(sfs_f.get_metric_dict()).T


sfs_m = SFS(multi_reg_m, k_features='best', forward=True)
sfs_m = sfs_m.fit(X_train_m, y_train_m)
results_m = pd.DataFrame.from_dict(sfs_m.get_metric_dict()).T