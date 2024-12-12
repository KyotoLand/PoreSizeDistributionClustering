import os
import pandas as pd

# Функция для создания индекса объектов и их групп

def create_index_file(groups, group_path_template, output_path):
    index_data = []

    for group in groups:
        group_folder = group_path_template.format(group)
        if not os.path.exists(group_folder):
            continue

        # Получаем список всех файлов в папке
        all_files = os.listdir(group_folder)

        # Фильтруем файлы для распределения пор и скалограмм
        pore_files = [f for f in all_files if 'pore_size_distribution' in f]
        wavelet_files = [f for f in all_files if 'wavelet_scalogram' in f]

        # Добавляем информацию об индексах и группах для каждого файла
        for file_name in pore_files + wavelet_files:
            index = int(file_name.split('_')[-1].split('.')[0])
            index_data.append({'Индекс объекта': index, 'Группа': group})

    # Создаем DataFrame и сохраняем в Excel
    index_df = pd.DataFrame(index_data)
    index_df.to_excel(output_path, index=False)

# Параметры для групп и шаблонов путей

GROUPS = list(range(len(os.listdir('resources/clusters'))))
GROUP_PATH_TEMPLATE = 'resources/clusters/cluster_{}'

# Создание индекса объектов и их групп и сохранение в Excel
INDEX_OUTPUT_PATH = 'resources/index_file.xlsx'
create_index_file(GROUPS, GROUP_PATH_TEMPLATE, INDEX_OUTPUT_PATH)