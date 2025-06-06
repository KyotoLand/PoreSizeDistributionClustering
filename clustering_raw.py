import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
import shutil
import os

# Загрузка пространства признаков из Excel файла
feature_space_df = pd.read_excel('./mnt/data/raw_data.xlsx', index_col=0)
feature_space = feature_space_df.values  # Использование модулей значений

# Определение оптимального числа главных компонент с помощью метода локтя
explained_variance = []
pca = PCA()
pca.fit(feature_space)
explained_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
plt.xlabel('Количество главных компонент')
plt.ylabel('Накопленная доля объясненной дисперсии')
plt.title('Метод локтя для выбора оптимального числа главных компонент')
plt.grid(True)
plt.savefig('resources/elbow_method_pca_raw.png')
plt.show()

# Функция для выполнения кластеризации K-means и оценки качества кластеризации
def cluster_and_evaluate_multiple_pca(feature_space, n_components, max_clusters=15):
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

    plt.figure(figsize=(10, 18))
    plt.subplot(3, 1, 1)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.xlabel('Количество кластеров')
    plt.ylabel('Коэффициент силуэта')
    plt.title(f'Силуэт (PCA: {n_components} компонентов)')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(range(2, max_clusters + 1), davies_bouldin_scores, marker='o')
    plt.xlabel('Количество кластеров')
    plt.ylabel('Коэффициент Девиса-Болдуина')
    plt.title(f'Девис-Болдуин (PCA: {n_components} компонентов)')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(range(2, max_clusters + 1), calinski_harabasz_scores, marker='o')
    plt.xlabel('Количество кластеров')
    plt.ylabel('Коэффициент Кавински-Харабаса')
    plt.title(f'Кавински-Харабаса (PCA: {n_components} компонентов)')
    plt.grid(True)

    plt.tight_layout(pad=2.0)
    plt.savefig('resources/cluster_quality_metrics_pca_20_raw.png')
    plt.show()

# Функция для выполнения финальной кластеризации и визуализации кластеров
def final_clustering_and_visualization(feature_space, n_components, n_clusters=15):
    pca = PCA(n_components=n_components)
    reduced_feature_space = pca.fit_transform(feature_space)

    optimal_kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    optimal_labels = optimal_kmeans.fit_predict(reduced_feature_space)
    feature_space_with_clusters = feature_space_df.copy()
    feature_space_with_clusters['Кластер'] = optimal_labels
    feature_space_with_clusters.to_excel('resources/feature_space_with_clusters.xlsx', index=True)

    # Визуализация кластеров на графике
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_feature_space[:, 0], reduced_feature_space[:, 1], c=optimal_labels, cmap='tab10', s=50, alpha=0.7, edgecolors='k')
    plt.xlabel('Главная компонента 1')
    plt.ylabel('Главная компонента 2')
    plt.title(f'Визуализация кластеров после PCA ({n_clusters} кластеров)')
    plt.grid(True)
    plt.savefig('resources/cluster_visualization_pca_12_raw.png')
    plt.show()

    # Сохранение графиков по группам кластеров
    cluster_groups = feature_space_with_clusters.groupby('Кластер')
    for cluster, group in cluster_groups:
        cluster_indices = group.index.astype(int)

        os.makedirs(f'resources/clusters/cluster_raw{cluster}', exist_ok=True)
        for idx in cluster_indices:
            # Копирование соответствующих скалограмм и графиков распределения пор по размерам в соответствующую папку
            pore_distribution_path = f'resources/grafs/pore_size_distribution_{int(idx)}.png'
            wavelet_path = f'resources/wavelets/wavelet_scalogram_{int(idx)}.png'
            if os.path.exists(pore_distribution_path):
                shutil.copy(pore_distribution_path,
                            f'resources/clusters/cluster_raw{cluster}/pore_size_distribution_{int(idx)}.png')
            if os.path.exists(wavelet_path):
                shutil.copy(wavelet_path, f'resources/clusters/cluster_raw{cluster}/wavelet_scalogram_{int(idx)}.png')

# Выполнение кластеризации и оценки качества на данных с 10 главными компонентами
cluster_and_evaluate_multiple_pca(feature_space, n_components=20)

# Выполнение финальной кластеризации и визуализации
final_clustering_and_visualization(feature_space, n_components=20, n_clusters=12)
