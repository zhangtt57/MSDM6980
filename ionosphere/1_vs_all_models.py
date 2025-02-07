import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from dwave_qbsolv import QBSolv

data_path = "C:/Users/19433/Desktop/ionosphere/ionosphere.data"

df = pd.read_csv(data_path, header=None)

features = df.iloc[:, :-1].values  # 特征矩阵 (351, 34)
labels = df.iloc[:, -1].values  # 标签向量 (351,)

unique_labels = np.unique(labels)  # 提取唯一标签

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
    
    # 创建所有二元标签矩阵
    Y = np.array([labels == label for label in unique_labels]).astype(int)  # (10, 70000)
    
    # 计算所有标签的重要性向量 I 
    I = np.array([mutual_info_classif(X_B, Y[i, :]) for i in range(len(unique_labels))])  # (10, 784)  
    
    selected_features_dict = {}
    
    def Q_star(alpha, eps, I_label): 
        Q = R - alpha * (R + np.diag(I_label))
        for i in range(n):
            if alpha * I_label[i] < eps:
                Q[i, i] = np.max(Q)
        best_vector = solve_with_qbsolv(Q)
        return best_vector
    
    for i, label in enumerate(unique_labels):
        I_label = I[i, :]
        best_vector = Q_star(alpha, 1e-8, I_label)
        k_star = np.sum(best_vector)
        
        while k_star != k:
            if k_star > k:
                b = alpha
            else:
                a = alpha
            alpha = (a + b) / 2
            best_vector = Q_star(alpha, 1e-8, I_label)
            k_star = np.sum(best_vector)
            
        selected_features_dict[label] = np.where(best_vector == 1)[0]
    
    return selected_features_dict

# %%  Logistic Reg. Ranking
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


def LR_ranking_FS(features, k, labels):
    selected_features_dict = {}
    
    for label in unique_labels:
        # 构建二元标签
        binary_labels = (labels == label).astype(int)  # 1 表示当前类别，0 表示其他类别
        
        scaler = StandardScaler()
        model = OneVsRestClassifier(LogisticRegression(solver='lbfgs', max_iter=1000))

        # 使用 Pipeline 将标准化和模型训练结合起来
        pipeline = Pipeline([('scaler', scaler), ('model', model)])
        pipeline.fit(features, binary_labels)

        coefficients = pipeline.named_steps['model'].estimators_  # 获取每个类别的模型

        # 计算每个特征的范数：累加每个模型对应范数的平方
        feature_norms = np.zeros(features.shape[1])
        for coef in coefficients:
            feature_norms += np.linalg.norm(coef.coef_, axis=0) ** 2  
        # feature_norms = np.sqrt(feature_norms)  # 取平方根（不影响排序）得到欧几里得范数

        # 创建 DataFrame，方便根据范数值降序排序
        df = pd.DataFrame({'Feature': range(features.shape[1]), 'Norm': feature_norms})
        sorted_features = df.sort_values(by='Norm', ascending=False)
        
        selected_features = sorted_features.head(k)  
        selected_features_dict[label] = selected_features['Feature'].values

    return selected_features_dict

# %%  Extra Tree Ranking
from sklearn.ensemble import ExtraTreesClassifier

def ET_ranking_FS(features, k, labels):
    selected_features_dict = {}
    
    for label in unique_labels:
        # 构建二元标签
        binary_labels = (labels == label).astype(int)  # 1 表示当前类别，0 表示其他类别

        scaler = StandardScaler()
        model = ExtraTreesClassifier(n_estimators=100, random_state=42)

        # 使用 Pipeline 将标准化和模型训练结合起来
        pipeline = Pipeline([('scaler', scaler), ('model', model)])
        pipeline.fit(features, binary_labels)

        model = pipeline.named_steps['model']

        # # 基于每个特征对减少杂质的贡献，获取每个特征的重要性
        importances = model.feature_importances_

        # 创建 DataFrame，方便根据重要性值降序排序
        df = pd.DataFrame({'Feature': range(features.shape[1]), 'Importance': importances})
        sorted_features = df.sort_values(by='Importance', ascending=False)

        selected_features = sorted_features.head(k)  
        selected_features_dict[label] = selected_features['Feature'].values

    return selected_features_dict

# %%  Decision Tree RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE

def RFE_ranking_FS(features, k, labels):
    selected_features_dict = {}  

    for label in unique_labels:
        # 构建二元标签
        binary_labels = (labels == label).astype(int)  # 1 表示当前类别，0 表示其他类别

        model = DecisionTreeClassifier(max_depth=10, random_state=42)

        # 递归特征消除
        rfe = RFE(estimator=model, n_features_to_select=k)
        rfe.fit(features, binary_labels)

        selected_features_dict[label] = np.where(rfe.support_)[0]

    return selected_features_dict