import pandas as pd
import matplotlib.pyplot as plt

# Load the uploaded Excel file
file_path = 'resources/index_file.xlsx'
excel_data = pd.ExcelFile(file_path)

# Load the data from the first sheet
df = excel_data.parse('Sheet1')

# Filter the data to include only the specified layers
layers_of_interest = ['Як2', 'Дл1']
filtered_df = df[df['Пласт'].isin(layers_of_interest)]

# Group by 'Пласт' and 'Группа', and count the number of samples in each group for the selected layers
filtered_distribution = filtered_df.groupby(['Пласт', 'Группа']).size().reset_index(name='Количество образцов')

# Create pie charts for each layer to show the distribution by clusters
unique_layers = filtered_distribution['Пласт'].unique()

plt.figure(figsize=(20, 15))

for i, layer in enumerate(unique_layers, start=1):
    layer_data = filtered_distribution[filtered_distribution['Пласт'] == layer]
    plt.subplot(2, 3, i)
    plt.pie(layer_data['Количество образцов'], labels=['Группа ' + str(x) for x in layer_data['Группа']], autopct='%1.1f%%', startangle=140, wedgeprops={'edgecolor': 'black'}, textprops={'fontsize': 14})
    plt.title(f'Типы керна пласта {layer}', fontsize=18)

plt.tight_layout()
plt.savefig('./resources/cluster_distribution_pie_charts.png')
plt.show()
