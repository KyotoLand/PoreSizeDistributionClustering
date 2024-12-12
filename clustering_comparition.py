import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
import shutil
import os

# Загрузка пространства признаков из Excel файла
feature_space_df = pd.read_excel('resources/feature_space.xlsx', index_col=0)
feature_space = np.abs(feature_space_df.values)  # Использование модулей значений

# Загрузка сырых данных из файла
raw_data_df = pd.read_excel('./mnt/data/raw_data.xlsx', index_col=0)
raw_data = raw_data_df.values


# Функция для выполнения кластеризации K-means и оценки качества кластеризации
def cluster_and_evaluate_multiple_pca(feature_space, n_components, max_clusters=15, label=''):
    pca = PCA(n_components=n_components)
    reduced_feature_space = pca.fit_transform(feature_space)

    silhouette_scores = []
    davies_bouldin_scores = []
    calinski_harabasz_scores = []

    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(reduced_feature_space)

        silhouette = silhouette_score(reduced_feature_space, labels)
        davies_bouldin = davies_bouldin_score(reduced_feature_space, labels)
        calinski_harabasz = calinski_harabasz_score(reduced_feature_space, labels)

        silhouette_scores.append(silhouette)
        davies_bouldin_scores.append(davies_bouldin)
        calinski_harabasz_scores.append(calinski_harabasz)

    return silhouette_scores, davies_bouldin_scores, calinski_harabasz_scores

# Выполнение кластеризации и оценки качества на данных с 10 главными компонентами (предобработанные данные)
preprocessed_silhouette, preprocessed_davies_bouldin, preprocessed_calinski_harabasz = cluster_and_evaluate_multiple_pca(feature_space, n_components=20, label='Предобработанные данные')

# Выполнение кластеризации и оценки качества на сырых данных с 10 главными компонентами
raw_silhouette, raw_davies_bouldin, raw_calinski_harabasz = cluster_and_evaluate_multiple_pca(raw_data, n_components=20, label='Сырые данные')

# Наложение графиков качества кластеризации для предобработанных и сырых данных
plt.figure(figsize=(15, 18))

plt.subplot(3, 1, 1)
plt.plot(range(2, 15 + 1), preprocessed_silhouette, marker='o', label='Предобработанные данные')
plt.plot(range(2, 15 + 1), raw_silhouette, marker='o', label='Сырые данные')
plt.xlabel('Количество кластеров')
plt.ylabel('Коэффициент силуэта')
plt.title('Силуэт для предобработанных и сырых данных')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(range(2, 15 + 1), preprocessed_davies_bouldin, marker='o', label='Предобработанные данные')
plt.plot(range(2, 15 + 1), raw_davies_bouldin, marker='o', label='Сырые данные')
plt.xlabel('Количество кластеров')
plt.ylabel('Коэффициент Девиса-Болдуина')
plt.title('Девис-Болдуин для предобработанных и сырых данных')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(range(2, 15 + 1), preprocessed_calinski_harabasz, marker='o', label='Предобработанные данные')
plt.plot(range(2, 15 + 1), raw_calinski_harabasz, marker='o', label='Сырые данные')
plt.xlabel('Количество кластеров')
plt.ylabel('Коэффициент Кавински-Харабаса')
plt.title('Кавински-Харабас для предобработанных и сырых данных')
plt.legend()
plt.grid(True)

plt.tight_layout(pad=2.0)
plt.savefig('resources/cluster_quality_metrics_comparison.png')
plt.show()