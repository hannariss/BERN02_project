import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import accuracy_score, r2_score
import matplotlib.pyplot as plt
import statsmodels.api as sm


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


# # not the same features, check some scores
model_mf = LogisticRegression(solver='newton-cg', max_iter=1000)
model_mf.fit(X_train_m[features_mf], y_train_m)
# pred_mf = model_mf.predict(X_test_m[features_mf])
# acc_mf = accuracy_score(y_test_m, pred_mf)

# model_ms = LogisticRegression(solver='newton-cg', max_iter=1000)
# model_ms.fit(X_train_m[features_ms], y_train_m)
# pred_ms = model_ms.predict(X_test_m[features_ms])
# acc_ms = accuracy_score(y_test_m, pred_ms)

# r2_mf = r2_score(y_test_m, pred_mf)
# r2_ms = r2_score(y_test_m, pred_ms)

# calculate AIC for male model to
# find the better model between forward and floating

#forward
probs_mf = model_mf.predict_proba(X_test_m[features_mf])
log_like = np.sum(np.log(probs_mf[np.arange(len(y_test_m)), y_test_m]))

n_classes = 3
n_features = len(features_mf)
k = n_features * (n_classes -1) + (n_classes -1)

aic_mf = 2*k - 2*log_like

# #floating
# probs_ms = model_ms.predict_proba(X_test_m[features_ms])
# log_like = np.sum(np.log(probs_ms[np.arange(len(y_test_m)), y_test_m]))

# n_classes = 3
# n_features = len(features_ms)
# k = n_features * (n_classes -1) + (n_classes -1)

# aic_ms = 2*k - 2*log_like
# print(f'forward: {aic_mf}, floating: {aic_ms}')
# # Conclusion: forward selection model is better

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

#------------------------------------------
# Confidence Intervals
#------------------------------------------
from scipy.linalg import inv
from numpy.linalg import inv, svd

# %%
def softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)

def hessian_full_softmax(X, y, B):
    """
    Compute the Hessian and covariance for the full softmax model
    (no reference class dropped).

    Parameters
    ----------
    X : (n_samples, p)
        Feature matrix (include intercept if needed)
    y : (n_samples,)
        Labels 0..K-1
    B : (p, K)
        Coefficient matrix (all K classes, not baseline-relative)

    Returns
    -------
    H : (p*K, p*K)
        Observed information matrix (positive semi-definite)
    cov : (p*K, p*K)
        Generalized inverse (pseudoinverse) of H for covariance
    se : (p*K,)
        Standard errors for flattened coefficients
    """
    n, p = X.shape
    K = B.shape[1]
    logits = X @ B
    P = softmax(logits)

    H = np.zeros((p * K, p * K))
    for i in range(n):
        p_i = P[i]
        W_i = np.diag(p_i) - np.outer(p_i, p_i)  # (K, K)
        XiXiT = np.outer(X[i], X[i])             # (p, p)
        H += np.kron(W_i, XiXiT)

    # H is positive semidefinite but singular because of invariance
    # Use Moore-Penrose pseudoinverse instead of regular inverse
    U, s, Vt = svd(H)
    tol = 1e-10
    s_inv = np.where(s > tol, 1.0 / s, 0.0)
    cov = (Vt.T * s_inv) @ U.T
    se = np.sqrt(np.diag(cov))
    return H, cov, se
# %%
# Female 
# assemble coef including intercept as first column
coef_f = model_ff.coef_           # (K, p_no_intercept)
intercept_f = model_ff.intercept_ # (K,)

# build full coef matrix shape (K, p) where p = p_no_intercept + 1
betas_f = np.hstack([intercept_f[:, None], coef_f])  # (K, p)

X = X_train_f[features_ff]
y = y_train_f

# X must include intercept column first
X_aug = np.hstack([np.ones((X.shape[0], 1)), X])  # (n, p)
H, cov, se = hessian_full_softmax(X_aug, y, betas_f.T)

z = 1.96
ci_lower = betas_f.flatten() - z * se
ci_upper = betas_f.flatten() + z * se

ci_lower_f = ci_lower.reshape(betas_f.shape)
ci_upper_f = ci_upper.reshape(betas_f.shape)

# %%
# Male 
# assemble coef including intercept as first column
coef_m = model_mf.coef_           # (K, p_no_intercept)
intercept_m = model_mf.intercept_ # (K,)

# build full coef matrix shape (K, p) where p = p_no_intercept + 1
betas_m = np.hstack([intercept_m[:, None], coef_m])  # (K, p)

X = X_train_m[features_mf]
y = y_train_m

# X must include intercept column first
X_aug = np.hstack([np.ones((X.shape[0], 1)), X])  # (n, p)
H, cov, se = hessian_full_softmax(X_aug, y, betas_m.T)

z = 1.96
ci_lower = betas_m.flatten() - z * se
ci_upper = betas_m.flatten() + z * se

ci_lower_m = ci_lower.reshape(betas_m.shape)
ci_upper_m = ci_upper.reshape(betas_m.shape)

# %%
# Plot with CIs 
import seaborn as sns

# Define custom colors
custom_palette = {
    0: sns.color_palette("muted")[2],  # greenish
    1: sns.color_palette("muted")[0],  # bluish
    2: sns.color_palette("muted")[3],  # reddish
}

ci_lower_f_df = pd.DataFrame(ci_lower_f[:, 1:], columns=coef_ff.columns)
ci_upper_f_df = pd.DataFrame(ci_upper_f[:, 1:], columns=coef_ff.columns)

ci_lower_m_df = pd.DataFrame(ci_lower_m[:, 1:], columns=coef_mf.columns)
ci_upper_m_df = pd.DataFrame(ci_upper_m[:, 1:], columns=coef_mf.columns)

df_f = coef_ff.reset_index().melt(id_vars='index', var_name='Column', value_name='Value')
df_m = coef_mf.reset_index().melt(id_vars='index', var_name='Column', value_name='Value')

# Map categorical 'Column' to numeric positions
col_mapping_f = {col: i  for i, col in enumerate(df_f['Column'].unique())}
df_f['y_numeric'] = df_f['Column'].map(col_mapping_f)

n_points = df_f['index'].nunique() 
offset = 0.25
offset_dict = {0: 0, 1: offset, 2: -offset} 

# Map index to offset
index_mapping = {idx: offset_dict[i] for i, idx in enumerate(df_f['index'].unique())}
df_f['y_numeric_shifted'] = df_f['y_numeric'] + df_f['index'].map(index_mapping)

# Female model CI lines
ci_lower_long_f = ci_lower_f_df.melt(var_name='Column', value_name='ci_lower')
ci_upper_long_f = ci_upper_f_df.melt(var_name='Column', value_name='ci_upper')

# Merge to align by Column and index (class)
df_ci = pd.concat([df_f, ci_lower_long_f.iloc[:,1], ci_upper_long_f.iloc[:, 1]],axis=1)


# Map categorical 'Column' to numeric positions
col_mapping_m = {col: i  for i, col in enumerate(df_m['Column'].unique())}
df_m['y_numeric'] = df_m['Column'].map(col_mapping_m)

n_points = df_m['index'].nunique() 
offset = 0.25
offset_dict = {0: 0, 1: offset, 2: -offset} 

# Map index to offset
index_mapping = {idx: offset_dict[i] for i, idx in enumerate(df_m['index'].unique())}
df_m['y_numeric_shifted'] = df_m['y_numeric'] + df_m['index'].map(index_mapping)

# Female model CI lines
ci_lower_long_m = ci_lower_m_df.melt(var_name='Column', value_name='ci_lower')
ci_upper_long_m = ci_upper_m_df.melt(var_name='Column', value_name='ci_upper')

# Merge to align by Column and index (class)
df_ci_m = pd.concat([df_m, ci_lower_long_m.iloc[:,1], ci_upper_long_m.iloc[:, 1]],axis=1)


# Plot
fig, axes = plt.subplots(1, 2, figsize=(10,7), sharex=True)

sns.scatterplot(
    data=df_f,
    x='Value',
    y='y_numeric_shifted',
    hue='index',
    palette=custom_palette,
    s=150,
    ax=axes[0]
)

axes[0].axvline(0, color='gray', linestyle='--', linewidth=1)
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].set_xlabel(r"Coefficient value ($\beta_i$)")  # Remove x-label on top plot
axes[0].set_ylabel("")
axes[0].set_title("Female model", fontsize=14)
axes[0].set_yticks(list(col_mapping_f.values()), list(col_mapping_f.keys())[::-1])

# Plot CIs as horizontal lines
for _, row in df_ci.iterrows():
    axes[0].hlines(
        y=row['y_numeric_shifted'],
        xmin=row['ci_lower'],
        xmax=row['ci_upper'],
        color=custom_palette[row['index']],
        alpha=0.9,
        linewidth=3
    )

# Male model
sns.scatterplot(
    data=df_m,
    x='Value',
    y='y_numeric_shifted',
    hue='index',
    palette=custom_palette,
    s=150,
    ax=axes[1]
)
axes[1].axvline(0, color='gray', linestyle='--', linewidth=1)
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].set_xlabel(r"Coefficient value ($\beta_i$)")
axes[1].set_yticks(list(col_mapping_m.values()), list(col_mapping_m.keys())[::-1])
axes[1].set_ylabel("")
axes[1].set_title("Male model", fontsize=14)

# Plot CIs as horizontal lines
for _, row in df_ci_m.iterrows():
    axes[1].hlines(
        y=row['y_numeric_shifted'],
        xmin=row['ci_lower'],
        xmax=row['ci_upper'],
        color=custom_palette[row['index']],
        alpha=0.9,
        linewidth=3
    )

# Shared legend outside the plot
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    ['Eustress', 'No Stress', 'Distress'],
    title='Categories',
    loc='lower center',
    ncol=3,              
    frameon=True,
    facecolor='white',
    bbox_to_anchor=(0.5, -0.1) 
)

# Remove duplicate legends inside subplots
axes[0].get_legend().remove()
axes[1].get_legend().remove()

fig.suptitle("Feature coefficients for different stress levels", fontsize=15, ha='center')
plt.tight_layout()
plt.show()

# %%
