import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn as sk

# Load data 
stress_data = pd.read_csv('data/Stress_wrangled.csv')

# Different target codes
stress_data.rename(columns={'stresstype': 'target'}, inplace=True)
target_order = {"Eustress (Positive Stress) - Stress that motivates and enhances performance.": 0, "No Stress - Currently experiencing minimal to no stress.": 1, "Distress (Negative Stress) - Stress that causes anxiety and impairs well-being.": 2}
stress_data["target_ordered"] = stress_data["target"].map(target_order)

# Split in male and female
female = stress_data[stress_data['Gender'] == 1]
male = stress_data[stress_data['Gender'] == 0]

X_f = female.drop(columns=['Gender', 'target', 'target_ordered'])
y_f = female['target_ordered']
X_m = male.drop(columns=['Gender', 'target', 'target_ordered'])
y_m = male['target_ordered']

# Split in train and test data
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(X_f, y_f, test_size=0.2, random_state=42)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_m, y_m, test_size=0.2, random_state=42)

# Load features
features_f = pd.read_csv('features_ff.csv', header=None).squeeze().tolist()
features_m = pd.read_csv('features_mf.csv', header=None).squeeze().tolist()


# Fit models
model_f = LogisticRegression(solver='newton-cg', max_iter=1000)
model_f.fit(X_train_f[features_f], y_train_f)

model_m = LogisticRegression(solver='newton-cg', max_iter=1000)
model_m.fit(X_train_m[features_m], y_train_m)

coef_df_f = pd.DataFrame(model_f.coef_, columns=X_test_f[features_f].columns, index=model_f.classes_) 
coef_df_m = pd.DataFrame(model_m.coef_, columns=X_test_m[features_m].columns, index=model_f.classes_) 


# Plot 

custom_palette = {
    0: sns.color_palette("muted")[2],  # greenish
    1: sns.color_palette("muted")[0],  # bluish
    2: sns.color_palette("muted")[3],  # reddish
}

# Create figure and subplots
fig, axes = plt.subplots(1, 2, figsize=(10,5), sharex=True)

# --- Female Model ---
sns.scatterplot(
    data=coef_df_f.reset_index().melt(id_vars='index', var_name='Column', value_name='Value'),
    x='Value',
    y='Column',
    hue='index',
    palette=custom_palette,
    s=120,
    ax=axes[0]
)
axes[0].axvline(0, color='gray', linestyle='--', linewidth=1)
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].set_xlabel(r"Coefficient value ($\beta_i$)")  # Remove x-label on top plot
axes[0].set_ylabel("")
axes[0].set_title("Female model", fontsize=14)

# --- Male Model ---
sns.scatterplot(
    data=coef_df_m.reset_index().melt(id_vars='index', var_name='Column', value_name='Value'),
    x='Value',
    y='Column',
    hue='index',
    palette=custom_palette,
    s=120,
    ax=axes[1]
)
axes[1].axvline(0, color='gray', linestyle='--', linewidth=1)
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].set_xlabel(r"Coefficient value ($\beta_i$)")
axes[1].set_ylabel("")
axes[1].set_title("Male model", fontsize=14)

# --- Shared legend outside the plot ---
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    ['Eustress', 'No Stress', 'Distress'],
    title='Categories',
    loc='lower center',   # position at bottom center
    ncol=3,               # number of columns = number of legend entries
    frameon=True,
    facecolor='white',
    bbox_to_anchor=(0.5, -0.1)  # optional: slightly below the plot
)

# Remove duplicate legends inside subplots
axes[0].get_legend().remove()
axes[1].get_legend().remove()
fig.suptitle("Feature coefficients for different stress levels", fontsize=15, ha='center')

plt.tight_layout()
plt.show()