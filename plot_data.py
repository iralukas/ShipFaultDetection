import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
# Настройки отображения
# plt.style.use('seaborn')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [12, 6]


def print_preview(filename='simulations_data1.parquet'):
    # Чтение файла
    df = pd.read_parquet(filename)

    # Основная информация
    print(f"Количество строк: {len(df)}")
    print(f"Колонки: {df.columns.tolist()}")

    # Первые 5 строк
    print("\nПервые 5 записей:")
    print(df.head())

    # Статистика по числовым колонкам
    print("\nСтатистика:")
    print(df.describe())

    # Проверка уникальных симуляций
    print(f"\nУникальные sim_id: {df['sim_id'].unique()}")
    print(f"\nУникальные fault_active: {df['fault_active'].unique()}")
    print(print(len(df[df['sim_id']==0])))




def load_and_prepare_data(filepath):
    """Загрузка и предварительная обработка данных"""
    df = pd.read_parquet(filepath)

    # Конвертация времени в минуты для удобства
    if 'time' in df.columns:
        df['time_min'] = df['time'] / 60

    # Добавление меток для ошибок
    if 'fault_state' in df.columns:
        fault_labels = {
            0: 'Нормальное поведение',
            1: 'Ошибка датчика скорости',
            2: 'Ошибка датчика курса',
            3: 'Ошибка датчика ускорения',
            4: 'Ошибка датчика угловой скорости',
            5: 'Ошибка актуатора скорости',
            6: 'Ошибка рулевого актуатора'
        }
        df['fault_label'] = df['fault_state'].map(fault_labels)

    return df


def plot_simulation_variables(df, sim_id=None, variables=None):
    """Построение графиков для выбранных переменных"""
    if variables is None:
        variables = ['real_V', 'sensor_V', 'V_set', 'u_V_real']

    if sim_id is not None:
        df = df[df['sim_id'] == sim_id]
        title_suffix = f" (Симуляция {sim_id})"
    else:
        title_suffix = " (Все симуляции)"

    fig, axes = plt.subplots(len(variables), 1, figsize=(14, 3 * len(variables)))

    for ax, var in zip(axes.flatten(), variables):
        if sim_id is not None:
            # График для конкретной симуляции
            sns.lineplot(data=df, x='time_min', y=var, ax=ax, label=var)
        else:
            # Усреднение по всем симуляциям
            sns.lineplot(data=df, x='time_min', y=var, ax=ax,
                         errorbar=('ci', 95), label=f"Среднее {var}")

        ax.set_title(f"{var}{title_suffix}")
        ax.set_xlabel("Время (мин)")
        ax.set_ylabel(var)
        ax.grid(True)

        # Добавление меток ошибок если есть
        if 'fault_label' in df.columns and sim_id is not None:
            faults = df[df['fault_label'].notna()]
            for _, row in faults.drop_duplicates('fault_label').iterrows():
                ax.axvline(x=row['time_min'], color='r', linestyle='--',
                           alpha=0.3, label=row['fault_label'])

        ax.legend()

    plt.tight_layout()
    return fig


def save_plots(fig, filename):
    """Сохранение графиков в файл"""
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    print(f"График сохранен как {output_dir / filename}")

def plot_course(data_file ="simulations_1_fault_0.parquet"):
    # 1. Загрузка данных
      # Укажите ваш файл
    df = load_and_prepare_data(data_file)

    # 2. Примеры визуализации
    print("Доступные колонки:", df.columns.tolist())

    # График для конкретной симуляции
    example_sim_id = df['sim_id'].sample(1).iloc[0]

    # 3. Дополнительные графики (по желанию)
    if 'real_K' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        data = df[df['sim_id'] == example_sim_id]
        sns.lineplot(data=data,
                     x='time', y='real_K', ax=ax, label='Реальный курс')
        sns.lineplot(data=data,
                     x='time', y='set_K', ax=ax, label='Целевой курс')
        sns.lineplot(data=data,
                     x='time', y='sensor_K', ax=ax, label='Измеренный курс')
        ax.set_title(f"Динамика курса (Симуляция {example_sim_id})")
        ax.set_ylabel("Курс (рад)")
        save_plots(fig, f"course_comparison_{example_sim_id}.png")

    plt.show()


def plot_velocity(data_file = "simulations_1_fault_0.parquet"):
    # 1. Загрузка данных
      # Укажите ваш файл
    df = load_and_prepare_data(data_file)

    # 2. Примеры визуализации
    print("Доступные колонки:", df.columns.tolist())

    # График для конкретной симуляции
    example_sim_id = df['sim_id'].sample(1).iloc[0]
    data = df[df['sim_id'] == example_sim_id]

    # 3. Дополнительные графики (по желанию)
    if 'real_V' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=data,
                     x='time', y='real_V', ax=ax, label='Реальная скорость')
        sns.lineplot(data=data,
                     x='time', y='set_V', ax=ax, label='Целевая скорость')

        ax.set_title(f"Динамика скорости (Файл {data_file} Симуляция {example_sim_id})")
        ax.set_ylabel("Скорость")

        save_plots(fig, f"velocity_comparison_{example_sim_id}.png")

    plt.show()


# Основной блок выполнения
if __name__ == "__main__":
    # plot_course(
    #     "data/14.04 (15 минут, 5 точек в секунду, поворот в первой половине сценария)/simulation_fault2_length1000.parquet")
    print_preview("data/14.04 (15 минут, 5 точек в секунду, поворот в первой половине сценария)/simulation_fault2_length1000.parquet")