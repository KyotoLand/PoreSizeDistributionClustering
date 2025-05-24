import os
import matplotlib.pyplot as plt
from PIL import Image
import random

# Функция для создания комбинированного изображения графиков распределения пор по размерам

def create_pore_distribution_image(groups, group_path_template, output_path):
    fig, axes = plt.subplots(3, len(groups), figsize=(len(groups) * 6, 12))
    #fig.suptitle('Графики распределения пор по размерам для каждой группы', fontsize=22)

    for i, group in enumerate(groups):
        group_folder = group_path_template.format(group)
        if not os.path.exists(group_folder):
            for j in range(3):
                axes[j, i].axis('off')
            continue

        # Получаем список всех файлов в папке
        all_files = os.listdir(group_folder)

        # Фильтруем файлы для распределения пор
        pore_files = [f for f in all_files if 'pore_size_distribution' in f]

        # Выбираем любые 3 файла из категории
        selected_pore_files = random.sample(pore_files, min(3, len(pore_files)))

        # Добавляем изображения распределения пор по размерам
        for j, pore_file in enumerate(selected_pore_files):
            pore_distribution_path = os.path.join(group_folder, pore_file)
            if os.path.exists(pore_distribution_path):
                image = Image.open(pore_distribution_path)
                axes[j, i].imshow(image)
                axes[j, i].axis('off')
                if j == 0:
                    axes[j, i].set_title(f'Группа {group}', fontsize=22)
            else:
                axes[j, i].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(output_path)
    plt.show()

# Функция для создания комбинированного изображения графиков вейвлет скалограмм

def create_wavelet_scalogram_image(groups, group_path_template, output_path):
    fig, axes = plt.subplots(3, len(groups), figsize=(len(groups) * 6, 12))
    #fig.suptitle('Графики вейвлет скалограмм для каждой группы', fontsize=22)

    for i, group in enumerate(groups):
        group_folder = group_path_template.format(group)
        if not os.path.exists(group_folder):
            for j in range(3):
                axes[j, i].axis('off')
            continue

        # Получаем список всех файлов в папке
        all_files = os.listdir(group_folder)

        # Фильтруем файлы для вейвлет скалограмм
        wavelet_files = [f for f in all_files if 'wavelet_scalogram' in f]

        # Выбираем любые 3 файла из категории
        selected_wavelet_files = random.sample(wavelet_files, min(3, len(wavelet_files)))

        # Добавляем изображения вейвлет скалограмм
        for j, wavelet_file in enumerate(selected_wavelet_files):
            wavelet_scalogram_path = os.path.join(group_folder, wavelet_file)
            if os.path.exists(wavelet_scalogram_path):
                image = Image.open(wavelet_scalogram_path)
                axes[j, i].imshow(image)
                axes[j, i].axis('off')
                if j == 0:
                    axes[j, i].set_title(f'Группа {group}', fontsize=22)
            else:
                axes[j, i].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(output_path)
    plt.show()

# Параметры для групп и шаблонов путей

GROUPS = [2, 3, 1]
GROUP_PATH_TEMPLATE = 'resources/clusters/cluster_{}'

# Создание комбинированного изображения для распределения пор
PORE_OUTPUT_PATH = 'resources/combined_pore_distribution_graphs.png'
create_pore_distribution_image(GROUPS, GROUP_PATH_TEMPLATE, PORE_OUTPUT_PATH)

# Создание комбинированного изображения для вейвлет скалограмм
WAVELET_OUTPUT_PATH = 'resources/combined_wavelet_scalogram_graphs.png'
create_wavelet_scalogram_image(GROUPS, GROUP_PATH_TEMPLATE, WAVELET_OUTPUT_PATH)
