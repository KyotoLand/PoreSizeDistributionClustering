import pandas as pd

# Пути к файлам
INDEX_FILE_PATH = './resources/index_file.xlsx'
POROSITY_PERMEABILITY_DATA_PATH = './mnt/data/data_with_por_perm.xlsx'

# Загрузка индекса и данных
index_df = pd.read_excel(INDEX_FILE_PATH)
por_perm_df = pd.read_excel(POROSITY_PERMEABILITY_DATA_PATH, header=None)

# Последние две строки данных соответствуют проницаемости и пористости
permeability = por_perm_df.iloc[-2, :]
porosity = por_perm_df.iloc[-1, :]

# Добавление значений проницаемости и пористости в DataFrame индекса
index_df['Проницаемость'] = index_df['Индекс объекта'].apply(lambda x: permeability[x - 1] if x - 1 < len(permeability) else None)
index_df['Пористость'] = index_df['Индекс объекта'].apply(lambda x: porosity[x - 1] if x - 1 < len(porosity) else None)

# Группировка по группам и расчет средних значений и дисперсий
group_stats = index_df.groupby('Группа').agg(
    Средняя_проницаемость=('Проницаемость', 'mean'),
    Дисперсия_проницаемости=('Проницаемость', 'var'),
    Средняя_пористость=('Пористость', 'mean'),
    Дисперсия_пористости=('Пористость', 'var')
)

# Сохранение результатов в Excel
GROUP_STATS_OUTPUT_PATH = './mnt/data/group_stats_updated.xlsx'
group_stats.to_excel(GROUP_STATS_OUTPUT_PATH, index=True)
