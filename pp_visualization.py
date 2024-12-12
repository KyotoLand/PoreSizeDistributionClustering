import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file and sheet
def load_data(file_path):
    xls = pd.ExcelFile(file_path)
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    # Convert 'Группа' to numeric to ensure compatibility for plotting
    df['Группа'] = pd.to_numeric(df['Группа'], errors='coerce')
    return df

# Plot percentage error of permeability by group and overall average
def plot_permeability_error(df, overall_avg_error):
    plt.figure(figsize=(12, 7))

    # Plotting individual group errors
    plt.bar(df['Группа'], df['Процент ошибки проницаемости'], label='Ошибка по группам')

    # Plotting the overall average as a horizontal line
    plt.axhline(y=overall_avg_error, color='r', linestyle='--', label='Процент ошибки по всей выборке')

    plt.xlabel('Группа')
    plt.ylabel('Процент ошибки проницаемости')
    plt.title('Сравнение процентной ошибки проницаемости в группах и по всей выборке')
    plt.legend()
    plt.grid(axis='y')
    plt.show()

# Plot percentage error of porosity by group and overall average
def plot_porosity_error(df):
    plt.figure(figsize=(12, 7))

    # Plotting individual group errors
    plt.bar(df['Группа'], df['Процент ошибки пористости'], label='Ошибка по группам')

    # Plotting the overall average as a horizontal line
    overall_avg_error_porosity = df['Процент ошибки пористости'].mean()
    plt.axhline(y=overall_avg_error_porosity, color='r', linestyle='--', label='Процент ошибки по всей выборке')

    plt.xlabel('Группа')
    plt.ylabel('Процент ошибки пористости')
    plt.title('Сравнение процентной ошибки пористости в группах и по всей выборке')
    plt.legend()
    plt.grid(axis='y')
    plt.show()


if __name__ == "__main__":
    # File path to the Excel file
    file_path = './mnt/data/group_stats_updated.xlsx'

    # Load the data
    df = load_data(file_path)

    # Load the overall average percentage error from cell G18
    overall_avg_error_permeability = pd.read_excel(file_path, sheet_name='Sheet1', usecols="G", nrows=18).iloc[-1, 0]

    # Plot the data
    plot_permeability_error(df, overall_avg_error_permeability)

    # Plot the porosity error
    plot_porosity_error(df)
