import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt

# Загрузка данных из файла
file_path = './mnt/data/test_data.xlsx'
data = pd.read_excel(file_path)

# Извлечение значений X и Y из файла
x_values = data.columns[1:].astype(float)  # Значения X (исключаем первый столбец с названиями пластов)
y_values_all = data.iloc[:, 1:].values  # Значения Y для всех образцов
sample_names = data.iloc[:, 0]  # Названия пластов

# Создание папок для сохранения графиков
os.makedirs('resources/wavelets', exist_ok=True)
os.makedirs('resources/grafs', exist_ok=True)


# Функция для построения графика распределения пор по размерам с логарифмической шкалой по оси X и сохранения в файл
def plot_pore_size_distribution_logx(x_values, y_values, sample_name, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, marker='o', linestyle='-')
    plt.title(f'Распределение пор по размерам для {sample_name}')
    plt.xlabel('Радиус пор, мкм')
    plt.ylabel('Объем пор, %')
    plt.xscale('log')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


# Функция для построения вейвлет-скалограммы с логарифмической шкалой по оси X и сохранения в файл
def plot_wavelet_scalogram_logx(signal, x_values, sample_name, save_path, wavelet='mexh'):
    plt.figure(figsize=(10, 6))
    coef, freqs = pywt.cwt(signal, scales=np.logspace(0.05, 2, num=50), wavelet=wavelet)
    plt.contourf(x_values, np.log2(freqs), np.abs(coef), levels=100, extend='both', cmap='jet')
    plt.title(f'Вейвлет-скалограмма для {sample_name}')
    plt.xlabel('Радиус пор, мкм')
    plt.ylabel('Логарифм частот')
    plt.colorbar(label='Величина')
    plt.xscale('log')
    plt.savefig(save_path)
    plt.close()


# Функция для формирования единого пространства признаков после вейвлет-преобразования
def create_feature_space(y_values_all, scales=np.logspace(0.1, 2, num=50), wavelet='mexh'):
    feature_space = []
    for y_values in y_values_all:
        coef, _ = pywt.cwt(y_values, scales=scales, wavelet=wavelet)
        feature_vector = coef.flatten()  # Преобразование матрицы коэффициентов в вектор признаков
        feature_space.append(feature_vector)
    return np.array(feature_space)


# Построение графиков для всех образцов и формирование пространства признаков
def process_data(file_path):
    data = pd.read_excel(file_path)
    x_values = data.columns[1:].astype(float)  # Значения X (исключаем первый столбец с названиями пластов)
    y_values_all = data.iloc[:, 1:].values  # Значения Y для всех образцов
    sample_names = data.iloc[:, 0]  # Названия пластов

    # Создание папок для сохранения графиков
    os.makedirs('resources/wavelets', exist_ok=True)
    os.makedirs('resources/grafs', exist_ok=True)

    # Построение графиков для всех образцов
    num_samples = len(sample_names)

    for sample_index in range(num_samples):
        y_values = y_values_all[sample_index]  # Значения Y для текущего образца
        sample_name = sample_names[sample_index]

        # Построение и сохранение графика распределения пор по размерам
        graf_path = f'resources/grafs/pore_size_distribution_{sample_index + 1}.png'
        plot_pore_size_distribution_logx(x_values, y_values, sample_name, graf_path)

        # Построение и сохранение вейвлет-скалограммы
        wavelet_path = f'resources/wavelets/wavelet_scalogram_{sample_index + 1}.png'
        plot_wavelet_scalogram_logx(y_values, x_values, sample_name, wavelet_path)

    # Формирование единого пространства признаков
    feature_space = create_feature_space(y_values_all)

    # Сохранение пространства признаков в Excel файл
    feature_space_df = pd.DataFrame(feature_space, index=sample_names)
    feature_space_df.to_excel('resources/feature_space.xlsx', index=True, header=False)


# Выполнение обработки данных
process_data('./mnt/data/test_data.xlsx')