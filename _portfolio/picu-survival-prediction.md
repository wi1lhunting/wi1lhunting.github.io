---
title: "基于机器学习的 PICU 患者院内死亡风险预测"
collection: portfolio
type: "Healthcare AI Project"
permalink: /portfolio/picu-mortality-prediction
date: 2026-01-18
excerpt: "利用随机森林与逻辑回归模型分析 PICU 临床数据，预测患者院内死亡风险（AUC 0.85），并结合 SHAP 值实现模型的可解释性分析。"
header:
  teaser: <br/><img src='/images/portfolio/picu-survival-prediction/pic_model_evaluation.png'>
tags:
  - Machine Learning
  - Healthcare
  - Scikit-learn
  - SHAP
  - Python
tech_stack:
  - name: Python
  - name: Scikit-learn
  - name: Pandas
  - name: SHAP
  - name: Seaborn
---

## 项目背景 (Background)

儿科重症监护室（PICU）患者的病情变化迅速，早期识别高危患者对于优化医疗资源分配和改善预后至关重要。本项目旨在利用机器学习算法挖掘 PICU 患者的临床数据（包括人口学特征、生命体征、实验室检查指标等），构建预测**院内死亡风险**的分类模型。

项目不仅仅关注预测精度，还特别引入了 **SHAP (SHapley Additive exPlanations)** 值分析，致力于打开机器学习的“黑箱”，为临床医生提供可解释的决策支持。

## 数据探索与分析 (Exploratory Data Analysis)

项目首先对 13,258 名患者的数据进行了深入的统计分析。通过可视化手段，我们探索了年龄、性别与死亡率之间的关系，并筛选了关键的实验室指标。

![基线分析](/images/portfolio/picu-survival-prediction/pic_baseline.png)
*图1:基线分析。*

![人口统计学分析](/images/portfolio/picu-survival-prediction/pic_demographic_analysis.png)
*图2：患者年龄与性别分布及其对死亡率的影响。数据显示婴儿组（0-1岁）和青少年组（10岁+）的死亡率相对较高。*

此外，我们分析了各临床特征与目标变量（是否死亡）的相关性，识别出与死亡风险最相关的前 20 个特征。

![特征相关性分析](/images/portfolio/picu-survival-prediction/pic_correlation_analysis.png)
*图3：与死亡风险相关性最强的正相关（红色）与负相关（蓝色）特征。*

## 核心实现 (Implementation)

### 1. 数据预处理
针对医疗数据的特点，我们执行了严格的数据清洗流程，包括剔除高缺失率特征、中位数填补缺失值以及标准化处理。

```python
# 数据预处理核心逻辑
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. 特征选择：移除ID类非特征列
id_cols = ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME']
feature_cols = [col for col in df_processed.columns 
                if col not in id_cols + ['HOSPITAL_EXPIRE_FLAG', 'is_early_death', 'age_group']]

X = df_processed[feature_cols].copy()
y = df_processed['HOSPITAL_EXPIRE_FLAG'].copy()

# 2. 缺失值处理：使用中位数填充
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 3. 数据标准化
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)

# 4. 数据集划分 (80% 训练, 20% 测试)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
```

### 2. 模型训练与超参数优化
本项目对比了 **逻辑回归 (Logistic Regression)** 和 **随机森林 (Random Forest)** 两种模型。为了提升性能，我们使用 `RandomizedSearchCV` 进行了超参数调优，特别是针对随机森林的树数量、深度和分裂标准进行了搜索。

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# 定义随机森林参数搜索空间
rf_param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': [10, 20, 30, None],
    'min_samples_split': randint(2, 20),
    'class_weight': ['balanced'] # 处理类别不平衡
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)

# 执行随机搜索
rf_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rf_param_dist,
    n_iter=20,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42
)

rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_
```

### 3. 模型评估 (Evaluation)
经过优化，随机森林模型在测试集上表现优异，AUC 达到 0.85 左右，优于逻辑回归模型。
![AUC](/images/portfolio/picu-survival-prediction/pic_model_evaluation.png)
![Recall and F1](/images/portfolio/picu-survival-prediction/pic_detailed_evaluation.png)

可解释性分析 (SHAP Analysis)
为了不仅知道“谁风险高”，还能知道“为什么风险高”，我们引入了 SHAP 值进行解释。
![SHAP Analysis](/images/portfolio/picu-survival-prediction/shap_bar_detailed.png)

全局解释性
SHAP 蜂群图展示了哪些特征对模型输出影响最大。例如，lab_5235_max（假设为某生化指标）的高值（红色点）主要分布在 SHAP 值大于 0 的区域，说明该指标升高会增加死亡风险。
![SHAP swarm](/images/portfolio/picu-survival-prediction/shap_swarm_detailed.png)

局部解释性
我们针对单个患者生成了“力图 (Force Plot)”，直观展示各特征如何共同作用导致该患者被预测为高风险。
图
![Force plot](/images/portfolio/picu-survival-prediction/shap_static_force_plot.png)

```python
import shap

# 初始化解释器
explainer = shap.Explainer(best_rf, X_train)
shap_values = explainer(X_train)

# 绘制特定样本的力图 (示例代码)
shap.force_plot(explainer.expected_value, shap_values.values[6], X_train.iloc[6])
```

### 4. 结论 (Conclusion)
模型性能：随机森林模型在处理高维 PICU 数据时表现出较强的鲁棒性，AUC 指标优于传统的线性模型。

关键因子：通过特征重要性和 SHAP 分析，我们识别出 age_month 以及一系列特定的实验室指标（如 lab_5235, lab_5237）是预测死亡风险的关键因素。

临床价值：结合 SHAP 的可视化解释工具，可以帮助临床医生快速定位高危指标，辅助临床决策。
