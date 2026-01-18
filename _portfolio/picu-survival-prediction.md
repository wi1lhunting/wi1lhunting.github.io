---
title: "PICU患者生存预测模型"
collection: portfolio
type: "Machine Learning"
permalink: /portfolio/picu-survival-prediction
date: 2026-01-18
excerpt: "基于PICU临床数据，构建逻辑回归模型预测患者生存情况，提升重症监护决策支持能力。"
header:
  teaser: /images/portfolio/picu-survival-prediction/roc_curve.png
tags:
- 儿科重症监护
- 生存预测
- 逻辑回归
- 机器学习
- 数据分析
tech_stack:
- name: Python
- name: Scikit-learn
- name: Pandas
- name: Matplotlib
---
# 读取数据，处理缺失值
pic_data = pd.read_csv('picu_data.csv')
pic_data.fillna(pic_data.median(), inplace=True)

# 特征与标签分离，数据标准化
X = pic_data.drop('Outcome', axis=1)
y = pic_data['Outcome']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集，训练逻辑回归模型
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
---

# 预测与评估
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(report)
