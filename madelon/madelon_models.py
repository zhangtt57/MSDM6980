import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from dwave_qbsolv import QBSolv


def load_sparse_data(data_path):
    labels = []
    features_list = []

    with open(data_path, 'r') as file:
        for line in file:
            elements = line.strip().split()
            label = int(elements[0])  # 第一个元素是标签
            labels.append(label)
            
            # 创建字典，保存该每一行的特征
            feature_dict = {int(k): float(v) for element in elements[1:] for k, v in [element.split(':')]}
            features_list.append(feature_dict)

    num_features = len(features_list[0].keys())
    
    features = np.zeros((len(labels), num_features), dtype=float)
    for i, feature_dict in enumerate(features_list):
        for j, val in feature_dict.items():
            features[i, j-1] = val  # j-1是因为特征索引从1开始
    
    return features, np.array(labels)


data_path = 'C:/Users/19433/Desktop/madelon/madelon'  
features, labels = load_sparse_data(data_path)  # (2000, 500), (2000,)

# %%  QFS
def discretize_features(X, B):
    discrete_X = np.zeros_like(X)
    for j in range(X.shape[1]):
        quantiles = np.linspace(min(X[:, j]), max(X[:, j]), B+1)
        discrete_X[:, j] = np.digitize(X[:, j], quantiles) 
    return discrete_X

def solve_with_qbsolv(Q):
    Q_dict = {(i, j): Q[i][j] for i in range(Q.shape[0]) for j in range(Q.shape[1])}
    response = QBSolv().sample_qubo(Q_dict)
    sample = next(response.samples())
    return np.array([sample[i] for i in range(Q.shape[0])])

def QUBO_FS(features, k, labels):   
    alpha = 0.5  # 平衡 I 和 R（避免过拟合）
    a = 0  # min_alpha = 0, Q_ij = R_ij
    b = 1  # max_alpha = 1, Q_ij = - I_j if i = j else 0
    
    X_B = discretize_features(features, 20)  # 离散化特征值
    n = X_B.shape[1]  # 特征维度
    
    # 计算冗余性矩阵 R
    R = np.zeros((n, n))  
    for j in range(n):
        R[:, j] = mutual_info_classif(X_B, X_B[:, j])  
    np.fill_diagonal(R, 0)
    
    # 计算所有标签的重要性向量 I 
    I = mutual_info_classif(X_B, labels) 
    
    def Q_star(alpha, eps): 
        Q = R - alpha * (R + np.diag(I))
        for i in range(n):
            if alpha * I[i] < eps:
                Q[i, i] = np.max(Q)
        best_vector = solve_with_qbsolv(Q)
        return best_vector
    
    best_vector = Q_star(alpha, 1e-8)
    k_star = np.sum(best_vector)
    
    while k_star != k:
        if k_star > k:
            b = alpha
        else:
            a = alpha
        alpha = (a + b) / 2
        best_vector = Q_star(alpha, 1e-8)
        k_star = np.sum(best_vector)
            
    selected_features = np.where(best_vector == 1)[0]
    
    return selected_features

# %%  Logistic Reg. Ranking
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


def LR_ranking_FS(features, k, labels):
    model = OneVsRestClassifier(LogisticRegression(max_iter=1000))

    pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
    pipeline.fit(features, labels)

    coefficients = pipeline.named_steps['model'].estimators_  

    # 计算每个特征的范数：累加对应系数的平方
    feature_norms = np.zeros(features.shape[1])
    for coef in coefficients:
        feature_norms += np.linalg.norm(coef.coef_, axis=0) ** 2  
    # feature_norms = np.sqrt(feature_norms)  # 取平方根（不影响排序）得到欧几里得范数

    # 创建 DataFrame，方便根据范数值降序排序
    df = pd.DataFrame({'Feature': range(features.shape[1]), 'Norm': feature_norms})
    sorted_features = df.sort_values(by='Norm', ascending=False)
    
    selected_features = sorted_features.head(k)['Feature'].values

    return selected_features

# %%  Extra Tree Ranking
from sklearn.ensemble import ExtraTreesClassifier

def ET_ranking_FS(features, k, labels):
    model = ExtraTreesClassifier(n_estimators=100, random_state=42)

    pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
    pipeline.fit(features, labels)

    model = pipeline.named_steps['model']

    # # 基于每个特征对减少杂质的贡献，获取每个特征的重要性
    importances = model.feature_importances_

    # 创建 DataFrame，方便根据重要性值降序排序
    df = pd.DataFrame({'Feature': range(features.shape[1]), 'Importance': importances})
    sorted_features = df.sort_values(by='Importance', ascending=False)

    selected_features = sorted_features.head(k)['Feature'].values
    
    return selected_features

# %%  Decision Tree RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE

def RFE_ranking_FS(features, k, labels):
    model = DecisionTreeClassifier(max_depth=10, random_state=42)

    # 递归特征消除
    rfe = RFE(estimator=model, n_features_to_select=k)
    rfe.fit(features, labels)

    selected_features = np.where(rfe.support_)[0]

    return selected_features

# %%
from sklearn.neural_network import MLPClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

k = 20  
results = {
    'QFS': QUBO_FS(features, k, labels),
    'Logistic Reg. Ranking': LR_ranking_FS(features, k, labels),
    'Extra Tree Ranking': ET_ranking_FS(features, k, labels),
    'Decision Tree RFE': RFE_ranking_FS(features, k, labels),
}

models = {
    'Neural Network': MLPClassifier(hidden_layer_sizes=(int(np.sqrt(k) + 0.5),), 
                                    max_iter=1000, activation='relu', 
                                    learning_rate_init=0.01),
    'Logistic Regression': OneVsRestClassifier(LogisticRegression(max_iter=1000)),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'Naive Bayes': GaussianNB(),
}

comparison_data = {}

for FS_name, result in results.items():  
    comparison_data[FS_name] = {}
    for model_name, model in models.items():  
        X_selected = features[:, result] 
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
        # 进行交叉验证
        scores = cross_val_score(pipeline, X_selected, labels, cv=10)  # 10-fold
        comparison_data[FS_name][model_name] = (scores.mean(), scores.std())

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
bar_width = 0.1
index = np.arange(len(models))  # 模型的索引

# 遍历每个特征选择方法，绘制其对应的准确率
for i, (FS_name, fs_results) in enumerate(comparison_data.items()):
    accuracies = [fs_results[model_name][0] for model_name in models.keys()]  # 提取准确率
    stds = [fs_results[model_name][1] for model_name in models.keys()]        # 提取标准差
    
    plt.bar(index + i * bar_width, accuracies, bar_width, 
            label=FS_name, yerr=stds, capsize=5)

# 设置图表标签和标题
plt.ylabel('Accuracy')
plt.ylim(0.2, 1)  # 设置 y 轴范围
plt.xticks(index + bar_width, models.keys())  # 设置 x 轴标签为模型名称
plt.title('Models for madelon')
plt.legend()  # 显示图例
plt.tight_layout()
plt.show()


