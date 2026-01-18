---
title: "PICU患者死亡率预测模型"
collection: portfolio
type: "Machine Learning"
permalink: /portfolio/picu-mortality-prediction
date: 2024-06-01
excerpt: "基于13,258例儿科重症监护数据构建可解释的死亡率预测模型，AUC达0.85，识别关键风险因素辅助临床决策"
header:
  teaser: /images/portfolio/picu-mortality-prediction/roc_curve_comparison.png
tags:
  - 机器学习
  - 医疗数据分析
  - 死亡率预测
  - 可解释AI
  - 随机森林
tech_stack:
  - name: Python
  - name: Pandas
  - name: Scikit-learn
  - name: SHAP
---
# 特征选择与缺失值处理
missing_rate = df.isnull().sum() / len(df)
selected_features = missing_rate[missing_rate < 0.9].index.tolist()
df_processed = df[selected_features].copy()

# 使用中位数填充缺失值并标准化
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)

# 分层采样划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 随机森林超参数优化
rf_param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': [10, 20, 30, None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2']
}

rf_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(
        random_state=42, 
        class_weight='balanced', 
        n_jobs=-1
    ),
    param_distributions=rf_param_dist,
    n_iter=20,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
rf_search.fit(X_train, y_train)

# SHAP解释器初始化与全局解释
explainer = shap.Explainer(lr_model, X_train)
shap_values = explainer(X_train)

# 绘制蜂群图展示特征影响分布
shap.summary_plot(shap_values, X_train, show=False)

# 单个特征依赖分析
shap.dependence_plot('lab_5235_max', shap_values.values, X_train)
