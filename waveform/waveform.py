import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from dwave_qbsolv import QBSolv 

data_path = 'C:/Users/19433/Desktop/waveform/waveform.data' 

df = pd.read_csv(data_path, header=None)

features = df.iloc[:, :-1].values  # (5000, 21)
labels = df.iloc[:, -1].values  # (5000,)

# %%
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

def QUBO_feature_selection(features, k, labels):   
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
    unique_labels = np.unique(labels) 
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

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def classif_RF(features, labels):
    """使用交叉验证计算随机森林分类器的准确性"""
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        max_features=5
    )
    scores = cross_val_score(clf, features, labels, cv=10)  # 10-fold
    return scores.mean(), scores.std()


def classif_RF_1_vs_all(features, labels):
    unique_labels = np.unique(labels)  # 获取所有唯一的标签
    results = {}

    for label in unique_labels:
        y_binary = (labels == label).astype(int)
        
        accuracy_mean, accuracy_std = classif_RF(features, y_binary)
        results[label] = {'accuracy': accuracy_mean, 'std': accuracy_std}
    
    return results
    

def random_FS_classif_RF_1_vs_all(features, k, labels):
    unique_labels = np.unique(labels)  # 获取所有唯一的标签
    results = {}

    for label in unique_labels:
        scores_across_runs = []
        
        for _ in range(5):  # 对于每个标签，执行 5 次随机特征选择，每次选择不同的随机子集
            indices = np.random.choice(features.shape[1], k, replace=False)
            X_selected = features[:, indices]
            y_binary = (labels == label).astype(int)
            
            score_mean, _ = classif_RF(X_selected, y_binary)  # 忽略每次运行的std
            scores_across_runs.append(score_mean)
        
        # 计算所有运行的平均准确率和标准差
        accuracy_mean = np.mean(scores_across_runs)
        accuracy_std = np.std(scores_across_runs)
        
        results[label] = {'accuracy': accuracy_mean, 'std': accuracy_std}
    
    return results


def QUBO_FS_classif_RF_1_vs_all(features, k, labels):
    results = {}
    selected_features_dict = QUBO_feature_selection(features, k, labels)

    for label, feature_indices in selected_features_dict.items():
        X_selected = features[:, feature_indices]  
        y_binary = (labels == label).astype(int)
        
        accuracy_mean, accuracy_std = classif_RF(X_selected, y_binary)
        results[label] = {'accuracy': accuracy_mean, 'std': accuracy_std, 
                          'selected_features': feature_indices}
    
    return results

# %%
import matplotlib.pyplot as plt

results_all_features = classif_RF_1_vs_all(features, labels)
results_random_fs = random_FS_classif_RF_1_vs_all(features, k=5, labels)
results_qubo_fs = QUBO_FS_classif_RF_1_vs_all(features, k=5, labels)

comparison_data = {}
unique_labels = np.unique(labels)

for label in unique_labels:
    comparison_data[label] = {
        'All Features': {'accuracy': results_all_features[label]['accuracy'], 
                         'std': results_all_features[label]['std']},
        'Random FS': {'accuracy': results_random_fs[label]['accuracy'], 
                      'std': results_random_fs[label]['std']},
        'QUBO FS': {'accuracy': results_qubo_fs[label]['accuracy'], 
                    'std': results_qubo_fs[label]['std']}
    }

# 遍历每个标签，绘制其对应的三种方法的准确率
plt.figure(figsize=(10, 6))
bar_width = 0.1
index = np.arange(len(unique_labels))

for i, method in enumerate(['All Features', 'Random FS', 'QUBO FS']):
    accuracies = [comparison_data[label][method]['accuracy'] 
                  for label in unique_labels]
    std = [comparison_data[label][method]['std'] 
           for label in unique_labels]
    plt.bar(index + i * bar_width, accuracies, bar_width, 
            label=method, yerr=std, capsize=5)

plt.xlabel('Labels of waveform')
plt.ylabel('Accuracy')
plt.ylim(0.2, 1)
plt.xticks(index + bar_width, unique_labels)  # 每个标签的实际名称
plt.tight_layout()
plt.legend()
plt.show()
