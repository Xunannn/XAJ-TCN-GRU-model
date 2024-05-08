import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from statsmodels.stats.outliers_influence import variance_inflation_factor
import shap
from sklearn.preprocessing import MinMaxScaler
import xgboost

df = pd.read_csv(r'D:\MASTER\my_paper\paper_2\Processed data\DL\MIC-Qingyi.csv')
values = df.values

X = values[:, :-1]
y= values[:, -1]

# Train an XGBoost model
model = xgboost.XGBRegressor()
model.fit(X, y)

# 计算 SHAP 值y
explainer = shap.Explainer(model, X)
shap_values = explainer.shap_values(X)

# 计算平均绝对 SHAP 值
mean_abs_shap_values = np.abs(shap_values).mean(axis=0)

# 计算特征重要性
fi = model.feature_importances_

# 计算 VIF
vif = np.array([variance_inflation_factor(X, i) for i in range(X.shape[1])])

# 计算 PFI
pfi = permutation_importance(model, X, y, n_repeats=30, random_state=42)
pfi_df = pfi.importances_mean

# 打印结果
print("平均绝对SHAP值：\n", mean_abs_shap_values)
print("\n特征重要性：\n", fi)
print("\nVIF：\n", vif)
print("\nPFI：\n", pfi_df)
