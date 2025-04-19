import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from params import *
import pandas as pd
from pid import PID
import random
import enum
from typing import Optional


class FaultType(enum.Enum):
    NO_FAULT = 0
    SENSOR_V_FAULT = 1  # Поломка сенсора скорости
    SENSOR_K_FAULT = 2  # Поломка сенсора курса
    SENSOR_VY_FAULT = 3  # Поломка сенсора поперечной скорости
    SENSOR_W_FAULT = 4  # Поломка сенсора угловой скорости
    ACTUATOR_V_FAULT = 5  # Поломка актуатора скорости
    ACTUATOR_K_FAULT = 6  # Поломка рулевого актуатора


class ShipModel:
    '''
    Модель надводного судна с управлением и поломками
    '''
    def __init__(self, fault_type: Optional[int] = None):
        self.fault_type = fault_type

        self.dataset = []  # Список для хранения данных
        self.fault_params = {}  # Параметры поломки (время возникновения, степень и т.д.)

        self.pid_V = PID(6, 0.001, 3, 0.1, -2.2, 2.2)
        self.pid_K = PID(3, 0.01, 0.1, 0.1, -25 * np.pi / 180, 25 * np.pi / 180)

        # Инициализация параметров поломки
        self._init_fault_params()

    def _select_random_fault(self) -> int:
        """Выбирает случайный тип поломки из возможных вариантов"""
        possible_faults = [
            FaultType.SENSOR_V_FAULT.value,
            FaultType.SENSOR_K_FAULT.value,
            FaultType.SENSOR_VY_FAULT.value,
            FaultType.SENSOR_W_FAULT.value,
            FaultType.ACTUATOR_V_FAULT.value,
            FaultType.ACTUATOR_K_FAULT.value
        ]
        return random.choice(possible_faults)

    def _init_fault_params(self):
        """Инициализирует параметры поломки в зависимости от типа"""
        # Время возникновения поломки (случайное в диапазоне 20-80% от времени моделирования)
        self.fault_params['fault_time'] = random.uniform(0.2, 0.8)

        # Параметры для разных типов поломок
        if self.fault_type == FaultType.SENSOR_V_FAULT.value:
            self.fault_params['bias'] = random.uniform(-0.5, 0.5)  # Смещение показаний
            self.fault_params['noise_std'] = random.uniform(0.1, 0.3)  # Уровень шума

        elif self.fault_type == FaultType.SENSOR_K_FAULT.value:
            self.fault_params['scale'] = random.uniform(0.7, 1.3)  # Масштабирование показаний
            self.fault_params['deadzone'] = random.uniform(0, 0.2)  # Мертвая зона

        elif self.fault_type == FaultType.SENSOR_VY_FAULT.value:
            self.fault_params['stuck_value'] = random.uniform(-0.5, 0.5)  # Застывшее значение

        elif self.fault_type == FaultType.SENSOR_W_FAULT.value:
            self.fault_params['delay'] = random.randint(1, 5)  # Задержка в измерениях
            self.fault_params['prev_values'] = []

        elif self.fault_type == FaultType.ACTUATOR_V_FAULT.value:
            self.fault_params['reduction'] = random.uniform(0.3, 0.7)  # Снижение эффективности
            self.fault_params['offset'] = random.uniform(-0.2, 0.2)  # Постоянное смещение

        elif self.fault_type == FaultType.ACTUATOR_K_FAULT.value:
            self.fault_params['saturation'] = random.uniform(0.2, 0.5)  # Ограничение диапазона
            self.fault_params['deadband'] = random.uniform(0.05, 0.15)  # Мертвая зона

    def _apply_sensor_faults(self, y: np.ndarray, t: float, time_span: float) -> np.ndarray:
        """Применяет сенсорные поломки к показаниям датчиков"""
        if t < self.fault_params['fault_time'] * time_span:
            # Поломка еще не произошла
            return y

        if self.fault_type == FaultType.SENSOR_V_FAULT.value:
            # Добавляем смещение и шум к показаниям скорости
            y[0][0] += self.fault_params['bias']
            y[0][0] += np.random.normal(0, self.fault_params['noise_std'])

        elif self.fault_type == FaultType.SENSOR_K_FAULT.value:
            # Масштабирование и мертвая зона для курса
            if abs(y[-1][0]) > self.fault_params['deadzone']:
                y[-1][0] *= self.fault_params['scale']
            else:
                y[-1][0] = 0

        elif self.fault_type == FaultType.SENSOR_VY_FAULT.value:
            # Датчик застывает на определенном значении
            y[1][0] = self.fault_params['stuck_value']

        elif self.fault_type == FaultType.SENSOR_W_FAULT.value:
            # Задержка в показаниях угловой скорости
            self.fault_params['prev_values'].append(y[2][0])
            if len(self.fault_params['prev_values']) > self.fault_params['delay']:
                y[2][0] = self.fault_params['prev_values'].pop(0)

        return y

    def _generate_V_set(self, total_points=1000, min_speed=1.0, max_speed=5.0,
                        min_interval=0.2, max_interval=0.4):
        """
        Генерирует ступенчатый профиль скорости с случайными интервалами

        Параметры:
            total_points - общее количество точек
            min_speed/max_speed - диапазон скоростей (м/с)
            min_interval/max_interval - минимальная/максимальная длительность интервала (в точках)
        """
        speed_profile = []
        current_point = 0

        while current_point < total_points:
            # Генерируем новую скорость и длительность интервала
            new_speed = np.random.uniform(min_speed, max_speed)
            interval = random.randint(int(total_points * min_interval), int(total_points * max_interval))

            # Добавляем интервал с постоянной скоростью
            speed_profile.extend([new_speed] * min(interval, total_points - current_point))
            current_point += interval

        return np.array(speed_profile[:total_points])  # Обрезаем до точного размера

    def _generate_K_set(self, num_points):
        # Генерация ступенчатого изменения курса
        step_time = random.randint(int(0.1 * num_points), int(0.3 * num_points))
        course = np.random.uniform(-np.pi / 4, np.pi / 4)

        set_K = np.concatenate([
            np.full(step_time, 0.0),
            np.full(num_points - step_time, course),
        ])
        return set_K

    def dxdt(self, t, x, time_points, set_points, params):
        # Проверка входящих значений
        if np.any(np.isnan(x)):
            print(f"Обнаружены NaN в t={t}")
            return np.zeros_like(x)  # Или обработка ошибки

        # 1. Интерполяция целевых значений
        rows = set_points.shape[0]
        set_point = []
        for i in range(rows):
            temp = set_points[i, :].ravel()
            set_point_i = np.interp(t, time_points, temp)
            set_point.append(set_point_i)
        set_point = np.array(set_point).reshape(rows, 1).ravel()
        V_set, K_set = set_point[0], set_point[1]

        # 2. Получение матрицы состояния
        A = get_A(x, params)
        B = get_B(x, params)
        C = get_C()
        D = get_D()
        fd = get_fd(x, params, t)

        # 3. Моделирование сенсорных ошибок
        y = np.matmul(C, x.reshape(-1, 1))
        y = self._apply_sensor_faults(y, t, time_points[-1])  # Применяем поломки

        # 4. Расчёт управляющих сигналов и моделирование ошибок актуатора
        V_sensor, K_sensor = y.ravel()[0], y.ravel()[-1]
        u_V_cmd = self.pid_V.get_u(V_sensor, V_set)  # Командный сигнал
        u_K_cmd = -self.pid_K.get_u(K_sensor, K_set)
        u_K_real, u_V_real = self.get_u_real(t, u_K_cmd, u_V_cmd, time_points[-1]) # Реальное управление (с поломкой)

        # 5. Применение реальных управляющих сигналов к системе
        u_real = np.array([[u_V_real], [u_K_real]])
        phi = get_phi(x, u_real, params)
        dxdt = np.matmul(A, x.reshape(-1, 1)) + np.matmul(B, u_real) + phi + np.matmul(D, fd)

        # 6. Сбор данных
        data_point = {
            'time': t,
            'sensor_V': y[0][0],
            'sensor_K': y[3][0],
            'sensor_Vy': y[1][0],
            'sensor_w': y[2][0],
            'real_V': x[0],  # Реальные состояния
            'real_K': x[3],
            'real_Vy': x[1],
            'real_w': x[2],
            'set_K': K_set,  # Целевые состояния
            'set_V': V_set,
            'u_V_cmd': u_V_cmd,  # Что "хотел" регулятор
            'u_K_cmd': u_K_cmd,
            'u_V_real': u_V_real,  # Что фактически выполнил актуатор
            'u_K_real': u_K_real,
            'fault_type': self.fault_type, # Вид поломки (0 при нормальном состоянии)
            'fault_active': t >= self.fault_params['fault_time'] * time_points[-1],
            'fault_time': self.fault_params['fault_time']  # Момент появления поломки
        }
        self.dataset.append(data_point)

        return dxdt.ravel()

    def get_u_real(self, t, u_K_cmd, u_V_cmd, total_time):
        # Фактическое значение (изначально = командному)
        u_V_real = u_V_cmd
        u_K_real = u_K_cmd

        if t < self.fault_params['fault_time'] * total_time:
            return u_K_real, u_V_real # До момента поломки реальное совпадает с расчетным

        if self.fault_type == FaultType.ACTUATOR_V_FAULT.value:
            # Ошибка актуатора скорости
            u_V_real = u_V_cmd * self.fault_params['reduction'] + self.fault_params['offset']

        elif self.fault_type == FaultType.ACTUATOR_K_FAULT.value:
            # Ошибка рулевого актуатора
            if abs(u_K_real) < self.fault_params['deadband']:
                u_K_real = 0
            else:
                u_K_real = np.clip(u_K_real,
                                   -self.fault_params['saturation'],
                                   self.fault_params['saturation'])
        return u_K_real, u_V_real

    def sim(self, stop_time=600, sample_rate=5):
        num_points = int(sample_rate * stop_time)
        simulation_time = np.linspace(0, stop_time, num_points)

        set_V = self._generate_V_set(num_points)
        set_K = self._generate_K_set(num_points)

        x0 = np.array([
            set_V[0],  # Начальная скорость продольная
            0,  # Начальная скорость поперечная
            0,  # угловая скорость
            set_K[0],  # курс
        ])
        self.dataset = []  # Очищаем датасет перед новой симуляцией

        set_points = np.vstack([set_V, set_K])

        sol = solve_ivp(self.dxdt,
                        t_span=[0, stop_time],
                        t_eval=simulation_time,
                        y0=x0.ravel(),
                        method='RK45',
                        max_step=0.5,  # Максимальный шаг интегрирования
                        args=(simulation_time, set_points, identification_params),
                        dense_output=True)

        if not sol.success:
            print(f"Интегрирование остановлено: {sol.message}")
            print(f"Последнее время: {sol.t[-1]}")
            plt.plot(sol.t, sol.y[0])
            plt.axvline(sol.t[-1], color='r', linestyle='--')
            plt.title(f"Остановка на t={sol.t[-1]:.1f}")
            plt.show()

        assert len(sol.t) == len(simulation_time), "Несоответствие количества точек в решении системы"

        # Собираем данные только в нужных точках
        self.dataset = []
        for t in simulation_time:
            x = sol.sol(t)  # Получаем решение через интерполяцию
            self.dxdt(t, x, simulation_time, set_points, identification_params)  # Для сохранения

        assert len(self.dataset) == len(simulation_time), "Несоответствие количества точек в датасете"
        return sol, pd.DataFrame(self.dataset)

    def plot(self, sol, df):
        # plt.plot(sol['t'], sol['y'][0], label='Vx')
        # plt.plot(sol['t'], sol['y'][1], label='Vy')
        # plt.plot(sol['t'], sol['y'][2], label="w")
        plt.plot(sol['t'], sol['y'][3], label='K')
        plt.plot(df['time'], df['set_K'], label='K_set')
        plt.plot(df['time'], df['sensor_K'], label='sensor_K')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    model = ShipModel(fault_type=0)
    sol, df = model.sim(stop_time=900)
    model.plot(sol, df)
