from enum import Enum

import numpy as np

class SystemState(Enum):
    NO_FAULT = 0
    SENSOR_V_FAULT = 1     # Поломка сенсора скорости
    SENSOR_VY_FAULT = 2     # Поломка сенсора курса
    SENSOR_W_FAULT = 3    # Поломка сенсора поперечной скорости
    SENSOR_K_FAULT = 4     # Поломка сенсора угловой скорости
    ACTUATOR_V_FAULT = 5   # Поломка актуатора скорости
    ACTUATOR_K_FAULT = 6   # Поломка рулевого актуатора


# Матрицы динамики
k_tilde_a0 = 0.0021
w_min = 0
w_max = 0.15
tilde_a11 = -0.0078
tilde_a12 = -0.8828
tilde_a21 = -2.44 * 10**(-5)
tilde_a22 = -0.0162
k_tilde_T = 0.0112
tilde_Yr = -0.0032
tilde_Mr = -4.9 * 10**(-5)
tilde_n01 = 1.187 * 10**(-4)
tilde_n02 = 1.1351
tilde_Xr = 0.0025
tilde_n11 = -0.0202
tilde_n12 = -1.3224
n = 2.5
# Возмущения
k_M_az = 0.075
tilde_d0 = -8.87 * 10**(-7)
tilde_d11 = -3.74 * 10**(-6)
tilde_d21 = 2.24 * 10**(-6)
tilde_d12 = -7.77 * 10**(-10)
tilde_d22 = -1.6 * 10**(-14)
# Шумы
sigma_1 = 0.03
alpha_1 = 1/30
sigma_2 = np.pi/1800
alpha_2 = 1/1800
alpha_3 = 3 * 10**(-4)
beta = 2 * np.pi / (84.4*60)
k = 2 * 10**(-5)
g3 = 10**(-7)
F = np.array([[-alpha_1, 0, 0, 0, 0],
              [0, -alpha_1, 0, 0, 0],
              [0, 0, -alpha_2, 0, 0],
              [0, 0, k, -beta**2, -2*alpha_3]])
G = np.array([[sigma_1*np.sqrt(2*alpha_1), 0, 0, 0],
              [0, sigma_1*np.sqrt(2*alpha_1), 0, 0],
              [0, 0, sigma_2*np.sqrt(2*alpha_2), 0],
              [0, 0, 0, 0],
              [0, 0, 0, g3]])
H = np.array([[1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0]])
# Начальные условия
P_34 = k * sigma_2**2 / (beta**2 +alpha_2 * (alpha_2 + 2*alpha_3))
P_35 = alpha_2 * P_34
P_44 = (2*k * (alpha_2 + 2*alpha_3) * P_34 + g3**2) / (4 * alpha_3 * beta**2)
P_55 = (2*k*P_35 + g3**2) / (4*alpha_3)
P = np.array([[sigma_1**2, 0, 0, 0, 0],
              [0, sigma_1**2, 0, 0, 0],
              [0, 0, sigma_2**2, P_34, P_35],
              [0, 0, P_34, P_44, 0],
              [0, 0, P_35, 0, P_55]])
# Таблица ветра и волн от бальности (Va, h_w, omega_w)
B_aw_table = np.array([[0, 0, 3],
                       [0.5, 0.25, 2.57],
                       [1.7, 0.75, 2],
                       [3.3, 1.25, 1.31],
                       [5.2, 2, 1.08],
                       [7.4, 3.5, 0.819],
                       [9.8, 6, 0.636],
                       [12.4, 8.5, 0.555],
                       [15, 11, 0.508]])

B_aw = 0 # балльность
# Параметры, которые требуется идентифицировать
identification_params = {'n': 2.5,
                         'w_min': w_min,
                         'w_max': w_max,
                         'tilde_a11': tilde_a11,
                         'tilde_a12': tilde_a12,
                         'tilde_a21': tilde_a21,
                         'tilde_a22': tilde_a22,
                         'k_tilde_T': k_tilde_T,
                         'tilde_Yr': tilde_Yr,
                         'tilde_Mr': tilde_Mr,
                         'tilde_n01': tilde_n01,
                         'tilde_n02': tilde_n02,
                         'tilde_Xr': tilde_Xr,
                         'tilde_n11': tilde_n11,
                         'tilde_n12': tilde_n12,
                         'tilde_d0': tilde_d0,
                         'tilde_d11': tilde_d11,
                         'tilde_d21': tilde_d21,
                         'tilde_d12': tilde_d12,
                         'tilde_d22': tilde_d22,
                         'k_M_az': k_M_az,
                         'Va': B_aw_table[B_aw, 0],
                         'h_w': B_aw_table[B_aw, 1],
                         'omega_w': B_aw_table[B_aw, 2],
                         'Ka': np.pi/6, # направление ветра
                         'Kw': np.pi/3, # угол набегающей волны
                         }


# Вспомогательные функции для расчета матриц
def get_V(state):
    state = state.ravel()
    return np.sqrt(state[0] ** 2 + state[1] ** 2)


def get_tilde_a0(V, n, k_w_min, k_w_max):
    if V < 0.5:
        w = k_w_min
    else:
        w = k_w_max
    return k_tilde_a0 * n * n * (1 - w)


def get_tilde_T(n, k_tilde_T):
    return k_tilde_T * np.abs(n) ** 3


def f_alpha(alpha):
    return np.pi * np.tan(alpha)


def get_Vk(state, identification_params):
    state = state.ravel()
    Vx = state[0]
    K = state[3]
    Va = identification_params['Va']
    Ka = identification_params['Ka']
    if identification_params['Va'] < 0.2:
        Vx_safe = max(Vx, 0.001)  # Чтобы не было нуля или отрицательных значений
        Vk = np.sqrt(Vx_safe)
    else:
        Vk = np.sqrt(Vx ** 2 + Va ** 2 - 2 * Vx * Va * np.cos(Ka - K + beta))
    return Vk


def get_SN_CS(state, identification_params):
    Vk = get_Vk(state, identification_params)
    if Vk < 0.2:
        SN = 0
        CS = 1
    else:
        state = state.ravel()
        Vx = state[0]
        Vy = state[1]
        K = state[3]
        Va = identification_params['Va']
        Ka = identification_params['Ka']
        beta = -np.arctan2(Vy, Vx)
        SN = (Vx * np.sin(beta) + Va * np.sin(Ka - K)) / Vk
        CS = (-Vx * np.cos(beta) + Va * np.cos(Ka - K)) / Vk
    return SN, CS


def get_A(state, identification_params):
    V = get_V(state)
    tilde_a0 = get_tilde_a0(V, identification_params['n'], identification_params['w_min'],
                            identification_params['w_max'])
    A = np.array([[tilde_a0, 0, 0, 0],
                  [0, identification_params['tilde_a11'] * V, identification_params['tilde_a12'] * V, 0],
                  [0, identification_params['tilde_a21'] * V, identification_params['tilde_a22'] * V, 0],
                  [0, 0, 1, 0]])
    return A


def get_B(state, identification_params):
    V = get_V(state)
    tilde_T = get_tilde_T(identification_params['n'], identification_params['k_tilde_T'])
    B = np.array([[tilde_T, 0],
                  [0, identification_params['tilde_Yr'] * V ** 2],
                  [0, identification_params['tilde_Mr'] * V ** 2],
                  [0, 0]])
    return B


def get_phi(state, u, identification_params):
    V = get_V(state)
    state = state.ravel()
    Vx = state[0]
    Vy = state[1]
    omega_z = state[2]
    u = u.ravel()
    delta = u[1]
    phi = np.array([[-identification_params['tilde_n01'] * Vx ** 2 +
                     identification_params['tilde_n02'] * Vy * omega_z -
                     identification_params['tilde_Xr'] * V ** 2 * delta ** 2],
                    [identification_params['tilde_n11'] * Vy * np.abs(Vy) +
                     identification_params['tilde_n12'] * Vy * omega_z * np.sign(Vy * omega_z)],
                    [0],
                    [0]])
    return phi


def get_C():
    C = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])  # можно  np.diag([1,2,3,4])
    return C


def get_D():
    D = np.array([[1, 0, 0, 0, 0],
                  [0, 1, 0, 1, 0],
                  [0, 0, 1, 0, 1],
                  [0, 0, 0, 0, 0]])
    return D


def get_fd(state, identification_params, t):
    state = state.ravel()
    Vx = state[0]
    K = state[3]
    tilde_d0 = identification_params['tilde_d0']
    Vk = get_Vk(state, identification_params)
    SN, CS = get_SN_CS(state, identification_params)
    tilde_d11 = identification_params['tilde_d11']
    tilde_d21 = identification_params['tilde_d11']
    k_M_az = identification_params['k_M_az']
    gamma = np.arctan2(SN, CS)
    tilde_d12 = identification_params['tilde_d12']
    omega_w = identification_params['omega_w']
    Kw = identification_params['Kw']
    h_w = identification_params['h_w']
    omega_k = np.abs(1 + Vx * np.cos(Kw - K) * omega_w / 9.81) * omega_w
    alpha_w = -omega_w ** 2 * h_w * np.sin(omega_k * t) / (2 * 9.81)
    dot_alpha_w = omega_w ** 2 * h_w * np.cos(omega_k * t) / (2 * 9.81)

    fd = np.array([[tilde_d0 * Vk ** 2 * CS],
                   [tilde_d11 * Vk ** 2 * SN],
                   [tilde_d21 * (k_M_az - np.abs(gamma) / (2 * np.pi)) * Vk ** 2 * SN],
                   [tilde_d12 * alpha_w],
                   [tilde_d22 * omega_w ** 2 / omega_k ** 2 * dot_alpha_w]])
    return fd
