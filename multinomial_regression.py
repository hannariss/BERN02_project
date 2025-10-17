import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt


# load data
stress_data = pd.read_csv('data/Stress_wrangled.csv')

#------------------------
# Data preparation
#------------------------
stress_data.rename(columns={'stresstype': 'target'}, inplace=True)

# Different target codes
target_order = {"Eustress (Positive Stress) - Stress that motivates and enhances performance.": 0, "No Stress - Currently experiencing minimal to no stress.": 1, "Distress (Negative Stress) - Stress that causes anxiety and impairs well-being.": 2}
stress_data["target_ordered"] = stress_data["target"].map(target_order)

# Split in male and female
female = stress_data[stress_data['Gender'] == 1]
male = stress_data[stress_data['Gender'] == 0]

X_f = female.drop(columns=['Gender', 'target', 'target_ordered'])
y_f = female['target_ordered']

X_m = male.drop(columns=['Gender', 'target', 'target_ordered'])
y_m = male['target_ordered']

# split data
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_f, y_f, test_size=0.2, random_state=42)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_m, y_m, test_size=0.2, random_state=42)

# define model
multi_reg_f = LogisticRegression(solver='newton-cg', max_iter=1000)
multi_reg_m = LogisticRegression(solver='newton-cg', max_iter=1000)

#---------------------------------------------
# Hyperparameter tuning for upper boundary
#---------------------------------------------

# Female
if 0:
    accs = [[],[],[]]
    for upper in range(1, 11):
        #-------------------------------
        # Feature selection (mlxtend)
        #-------------------------------
        # forward
        sfs_f_f = SFS(multi_reg_f, k_features=(1, upper), forward=True)
        sfs_f_f = sfs_f_f.fit(X_train_f, y_train_f)
        results_f_f = pd.DataFrame.from_dict(sfs_f_f.get_metric_dict()).T
        features_ff = list(sfs_f_f.k_feature_names_)

        # backward
        sfs_f_b = SFS(multi_reg_f, k_features=(1, upper), forward=False)
        sfs_f_b = sfs_f_b.fit(X_train_f, y_train_f)
        results_f_b = pd.DataFrame.from_dict(sfs_f_b.get_metric_dict()).T
        features_fb = list(sfs_f_b.k_feature_names_)

        # stepwise selection (floating)
        sfs_f_s = SFS(multi_reg_f, k_features=(1, upper), forward=True, floating=True)
        sfs_f_s = sfs_f_s.fit(X_train_f, y_train_f)
        results_f_s = pd.DataFrame.from_dict(sfs_f_s.get_metric_dict()).T
        features_fs = list(sfs_f_s.k_feature_names_)
        
        #-------------------------------
        # Model evaluation
        #-------------------------------
        model_ff = LogisticRegression(solver='newton-cg', max_iter=1000)
        model_ff.fit(X_train_f[features_ff], y_train_f)
        pred_ff = model_ff.predict(X_test_f[features_ff])
        acc_ff = accuracy_score(y_test_f, pred_ff)

        model_fb = LogisticRegression(solver='newton-cg', max_iter=1000)
        model_fb.fit(X_train_f[features_fb], y_train_f)
        pred_fb = model_fb.predict(X_test_f[features_fb])
        acc_fb = accuracy_score(y_test_f, pred_fb)

        model_fs = LogisticRegression(solver='newton-cg', max_iter=1000)
        model_fs.fit(X_train_f[features_fs], y_train_f)
        pred_fs = model_fs.predict(X_test_f[features_fs])
        acc_fs = accuracy_score(y_test_f, pred_fs)

        accs[0].append(np.round(acc_ff, 4))
        accs[1].append(np.round(acc_fb, 4))
        accs[2].append(np.round(acc_fs, 4))
        # check accuarcy of female model
        # print(f'female forward: {np.round(acc_ff, 4)}, backward: {np.round(acc_fb, 4)}, stepwise: {np.round(acc_fs, 4)}')

    accs = np.array(accs)
    df_accs = pd.DataFrame({'forward' : accs[0], 'backward': accs[1], 'floating': accs[2]})
    df_accs.to_csv('female_accuracies.csv')

# Male
if 0: 
    accs = [[],[],[]]
    for upper in range(1, 11):
        #-------------------------------
        # Feature selection (mlxtend)
        #-------------------------------
        # forward
        sfs_m_f = SFS(multi_reg_m, k_features=(1, upper), forward=True)
        sfs_m_f = sfs_m_f.fit(X_train_m, y_train_m)
        results_m_f = pd.DataFrame.from_dict(sfs_m_f.get_metric_dict()).T
        features_mf = list(sfs_m_f.k_feature_names_)

        # backward
        sfs_m_b = SFS(multi_reg_m, k_features=(1, upper), forward=False)
        sfs_m_b = sfs_m_b.fit(X_train_m, y_train_m)
        results_m_b = pd.DataFrame.from_dict(sfs_m_b.get_metric_dict()).T
        features_mb = list(sfs_m_b.k_feature_names_)

        # stepwise selection (floating)
        sfs_m_s = SFS(multi_reg_m, k_features=(1, upper), forward=True, floating=True)
        sfs_m_s = sfs_m_s.fit(X_train_m, y_train_m)
        results_m_s = pd.DataFrame.from_dict(sfs_m_s.get_metric_dict()).T
        features_ms = list(sfs_m_s.k_feature_names_)
        
        #-------------------------------
        # Model evaluation
        #-------------------------------
        model_mf = LogisticRegression(solver='newton-cg', max_iter=1000)
        model_mf.fit(X_train_m[features_mf], y_train_m)
        pred_mf = model_mf.predict(X_test_m[features_mf])
        acc_mf = accuracy_score(y_test_m, pred_mf)

        model_mb = LogisticRegression(solver='newton-cg', max_iter=1000)
        model_mb.fit(X_train_m[features_mb], y_train_m)
        pred_mb = model_mb.predict(X_test_m[features_mb])
        acc_mb = accuracy_score(y_test_m, pred_mb)

        model_ms = LogisticRegression(solver='newton-cg', max_iter=1000)
        model_ms.fit(X_train_m[features_ms], y_train_m)
        pred_ms = model_ms.predict(X_test_m[features_ms])
        acc_ms = accuracy_score(y_test_m, pred_ms)

        accs[0].append(np.round(acc_mf, 4))
        accs[1].append(np.round(acc_mb, 4))
        accs[2].append(np.round(acc_ms, 4))
        # # check accuarcy of male model
        # print(f'male forward: {np.round(acc_mf, 4)}, backward: {np.round(acc_mb, 4)}, stepwise: {np.round(acc_ms, 4)}')
    
    accs = np.array(accs)
    df_accs = pd.DataFrame({'forward' : accs[0], 'backward': accs[1], 'floating': accs[2]})
    df_accs.to_csv('male_accuracies.csv')

# with k_features='best' we get scores:
# female forward: 0.9830508474576272, backward: 1.0, stepwise: 0.9830508474576272
# male forward: 0.9363636363636364, backward: 0.9181818181818182, stepwise: 0.9363636363636364

# k_features=(1, 10) we get scores:
# female forward: 0.9491525423728814, backward: 0.9491525423728814, stepwise: 0.9491525423728814
# male forward: 0.9090909090909091, backward: 0.9272727272727272, stepwise: 0.9454545454545454

df_accs_f = pd.read_csv('female_accuracies.csv', index_col=0)
df_accs_m = pd.read_csv('male_accuracies.csv', index_col=0)


for col in df_accs_f.columns:
    plt.scatter(df_accs_f.index + 1, df_accs_f[col], label=col)
plt.xlabel("Number of selected features")
plt.ylabel("Accuracy")
plt.title("Accuracy after stepwise selection vs. number of features - Female model")
plt.grid()
plt.legend()
plt.xticks(np.arange(1, len(df_accs_f.index) + 1, 1))
plt.show()

for col in df_accs_m.columns:
    plt.scatter(df_accs_m.index + 1, df_accs_m[col], label=col)
plt.xlabel("Number of selected features")
plt.ylabel("Accuracy")
plt.title("Accuracy after stepwise selection vs. number of features - Male model")
plt.grid()
plt.legend()
plt.xticks(np.arange(1, len(df_accs_m.index) + 1, 1))
plt.show()


#---------------------------------------------
# Fit models with optimum number of features
#---------------------------------------------

# Female: optimum of 6 features
# Male: optimum of 5 features

# check if the same features were selected in forward and floating

## female
# forward
sfs_f_f = SFS(multi_reg_f, k_features=(1, 6), forward=True)
sfs_f_f = sfs_f_f.fit(X_train_f, y_train_f)
results_f_f = pd.DataFrame.from_dict(sfs_f_f.get_metric_dict()).T
features_ff = list(sfs_f_f.k_feature_names_)

# stepwise selection (floating)
# sfs_f_s = SFS(multi_reg_f, k_features=(1, 6), forward=True, floating=True)
# sfs_f_s = sfs_f_s.fit(X_train_f, y_train_f)
# results_f_s = pd.DataFrame.from_dict(sfs_f_s.get_metric_dict()).T
# features_fs = list(sfs_f_s.k_feature_names_)

# features are the same for forward and floating selection, we take the forward selection model

## male
# forward
sfs_m_f = SFS(multi_reg_m, k_features=(1, 5), forward=True)
sfs_m_f = sfs_m_f.fit(X_train_m, y_train_m)
results_m_f = pd.DataFrame.from_dict(sfs_m_f.get_metric_dict()).T
features_mf = list(sfs_m_f.k_feature_names_)

# stepwise selection (floating)
# sfs_m_s = SFS(multi_reg_m, k_features=(1, 5), forward=True, floating=True)
# sfs_m_s = sfs_m_s.fit(X_train_m, y_train_m)
# results_m_s = pd.DataFrame.from_dict(sfs_m_s.get_metric_dict()).T
# features_ms = list(sfs_m_s.k_feature_names_)


# not the same features, check some scores
model_mf = LogisticRegression(solver='newton-cg', max_iter=1000)
model_mf.fit(X_train_m[features_mf], y_train_m)
pred_mf = model_mf.predict(X_test_m[features_mf])
acc_mf = accuracy_score(y_test_m, pred_mf)

model_ms = LogisticRegression(solver='newton-cg', max_iter=1000)
model_ms.fit(X_train_m[features_ms], y_train_m)
pred_ms = model_ms.predict(X_test_m[features_ms])
acc_ms = accuracy_score(y_test_m, pred_ms)

r2_mf = r2_score(y_test_m, pred_mf)
r2_ms = r2_score(y_test_m, pred_ms)

# calculate AIC for male model to
# find the better model between forward and floating

#forward
probs_mf = model_mf.predict_proba(X_test_m[features_mf])
log_like = np.sum(np.log(probs_mf[np.arange(len(y_test_m)), y_test_m]))

n_classes = 3
n_features = len(features_mf)
k = n_features * (n_classes -1) + (n_classes -1)

aic_mf = 2*k - 2*log_like

#floating
probs_ms = model_ms.predict_proba(X_test_m[features_ms])
log_like = np.sum(np.log(probs_ms[np.arange(len(y_test_m)), y_test_m]))

n_classes = 3
n_features = len(features_ms)
k = n_features * (n_classes -1) + (n_classes -1)

aic_ms = 2*k - 2*log_like
print(f'forward: {aic_mf}, floating: {aic_ms}')
# Conclusion: forward selection model is better

#--------------------------------
# Final model:
# female: 6 features (forward)
# male: 5 features (forward)
#--------------------------------

# fit model for female
model_ff = LogisticRegression(solver='newton-cg', max_iter=1000)
model_ff.fit(X_train_f[features_ff], y_train_f)
pred_ff = model_ff.predict(X_test_f[features_ff])

# inspect coefficients
# Coefficients (one row per class)
classes = model_ff.classes_
coef_ff = pd.DataFrame(model_ff.coef_, columns=X_test_f[features_ff].columns, index=classes) 
coef_mf = pd.DataFrame(model_mf.coef_, columns=X_test_m[features_mf].columns, index=classes) 

# save training and test data with selected features
np.savetxt('features_ff.csv', features_ff, delimiter=",", fmt='%s')
np.savetxt('features_mf.csv', features_mf, delimiter=",", fmt='%s')
