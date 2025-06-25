import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

# --- Параметры ---

# Параметры целевой функции f(x) = 1/2 * (L*x1^2 + mu*x2^2)
L = 100.0
MU = 0.0001
# Стандартное отклонение шума в стохастическом градиенте
NOISE_SIGMA = 0.5
# Начальная точка для всех алгоритмов
X_INIT = np.array([1.0, 1.0])

# --- Целевая функция и ее градиенты ---

def f(x):
    """Целевая функция (детерминированная)."""
    return 0.5 * (L * x[0]**2 + MU * x[1]**2)

def grad_f(x):
    """Градиент целевой функции (детерминированный)."""
    return np.array([L * x[0], MU * x[1]])

def stochastic_grad_f(x):
    """Стохастический градиент с Гауссовым шумом."""
    noise = np.random.normal(0, NOISE_SIGMA, size=x.shape)
    return grad_f(x) + noise

# --- Реализация алгоритмов ---

def sgd(
    n_iters,
    x_init=X_INIT,
    prox_center=None,
    prox_lambda=0,
    strong_convexity=MU,
    smoothness=L
):
    """
    Базовый решатель: Стохастический градиентный спуск (SGD).
    Может решать как исходную задачу, так и проксимальную подзадачу.
    
    Args:
        n_iters (int): Количество итераций.
        x_init (np.array, optional): Начальная точка.
        prox_center (np.array, optional): Центр для проксимального члена.
        prox_lambda (float, optional): Коэффициент регуляризации.
        strong_convexity (float): Параметр сильной выпуклости текущей задачи.
        smoothness (float): Параметр гладкости текущей задачи.

    Returns:
        np.array: Найденное решение.
        int: Количество вызовов градиента.
        list: Траектория движения (история точек).
    """
    x = x_init.copy()
    grad_calls = 0
    trajectory = [x.copy()]

    # Используем постоянный шаг для простоты
    step_size = 1.0 / smoothness

    for _ in range(n_iters):
        # Вычисляем стохастический градиент для f(x)
        grad_val = stochastic_grad_f(x)
        grad_calls += 1
        
        # Добавляем градиент от проксимального члена, если он есть
        if prox_center is not None and prox_lambda > 0:
            grad_val += prox_lambda * (x - prox_center)
            
        x -= step_size * grad_val
        trajectory.append(x.copy())
        
    return x, grad_calls, trajectory

def robust_distance_estimator(
    m_trials, 
    n_iters_sgd,
    start_point,
    prox_center=None,
    prox_lambda=0,
    strong_convexity=MU,
    smoothness=L
):
    """
    Реализация "Робастной оценки расстояния" (Algorithm 1 в статье).
    Этот метод вызывает "слабый" решатель (SGD) m раз и выбирает лучший результат.

    Args:
        m_trials (int): Количество запусков SGD.
        n_iters_sgd (int): Количество итераций в каждом запуске SGD.
        start_point (np.array): Начальная точка для слабого оракула.
        prox_center (np.array, optional): Центр для проксимального члена.
        prox_lambda (float, optional): Коэффициент регуляризации.
        strong_convexity (float): Параметр сильной выпуклости текущей задачи.
        smoothness (float): Параметр гладкости текущей задачи.

    Returns:
        np.array: Оцененное решение.
        int: Общее количество вызовов градиента.
        list: Траектория лучшего запуска SGD.
    """
    weak_oracle = lambda: sgd(
        n_iters=n_iters_sgd, 
        x_init=start_point,
        prox_center=prox_center, 
        prox_lambda=prox_lambda, 
        strong_convexity=strong_convexity, 
        smoothness=smoothness
    )
    
    total_grad_calls = 0
    
    # 1. Сгенерировать m точек с помощью слабого оракула (SGD)
    points = []
    trajectories = []
    for _ in range(m_trials):
        point, grad_calls, trajectory = weak_oracle()
        points.append(point)
        trajectories.append(trajectory)
        total_grad_calls += grad_calls
    
    points = np.array(points)
    
    # 2. Найти точку, которая является центром наименьшего шара,
    # содержащего > m/2 других точек.
    if len(points) == 1:
        return points[0], total_grad_calls, trajectories[0]

    pairwise_distances = squareform(pdist(points))
    
    min_radius = float('inf')
    best_point_idx = -1
    
    target_count = len(points) // 2 + 1
    
    for i in range(len(points)):
        # Для каждой точки найти радиус, содержащий > m/2 точек
        sorted_distances = np.sort(pairwise_distances[i])
        radius = sorted_distances[target_count - 1]
        
        if radius < min_radius:
            min_radius = radius
            best_point_idx = i
            
    return points[best_point_idx], total_grad_calls, trajectories[best_point_idx]

def prox_boost_alg(
    target_epsilon,
    target_p_fail,
    n_iters_sgd_weak=20,
):
    """
    Реализация основного алгоритма BoostAlg (proxBoost) из статьи.

    Args:
        target_epsilon (float): Желаемая точность итогового решения.
        target_p_fail (float): Желаемая вероятность ошибки.
        n_iters_sgd_weak (int): Кол-во итераций для "слабого" решателя SGD.

    Returns:
        np.array: Итоговое решение.
        list: История ошибки (значение функции) по мере итераций.
        list: История количества вызовов градиента.
    """
    
    kappa = L / MU
    # Параметры алгоритма, как в статье (Corollary 6.4)
    T = int(np.ceil(np.log2(kappa)))
    if T == 0: T=1
    
    m = int(np.ceil(5 * np.log((T + 3) / target_p_fail))) # 18 - константа из статьи
    if m <= 1: m = 2
        
    # Точность для подзадач
    delta = target_epsilon / (4 + 2 * T)

    plot_error_hist = [f(X_INIT)]
    plot_gc_hist = [0]
    cumulative_grad_calls = 0
    
    # --- Этап I: Инициализация ---
    # Получаем x0 робастной оценкой для исходной задачи f(x)
    
    x_current, grad_calls_stage, _ = robust_distance_estimator(
        m_trials=m,
        n_iters_sgd=n_iters_sgd_weak,
        start_point=X_INIT,
    )
    
    # Добавляем точку после первого этапа
    cumulative_grad_calls += grad_calls_stage
    plot_gc_hist.append(cumulative_grad_calls)
    plot_error_hist.append(f(x_current))
    
    
    # --- Этап II: Проксимальные итерации ---
    # Геометрическая последовательность для лямбда
    lambdas = [MU * (2**i) for i in range(T + 1)]

    for j in range(T):
        prox_lambda = lambdas[j]
        prox_center = x_current
        
        # Обновленные параметры для подзадачи f(x) + lambda/2 * ||x-x_center||^2
        current_mu = MU + prox_lambda
        current_L = L + prox_lambda
        
        x_current, grad_calls_stage, _ = robust_distance_estimator(
            m_trials=m,
            n_iters_sgd=n_iters_sgd_weak,
            start_point=x_current,
            prox_center=prox_center,
            prox_lambda=prox_lambda,
            strong_convexity=current_mu,
            smoothness=current_L
        )
        cumulative_grad_calls += grad_calls_stage * m #ДОБАВИЛ МНОЖИТЕЛЬ m
        plot_gc_hist.append(cumulative_grad_calls)
        plot_error_hist.append(f(x_current))

    # --- Этап III: Финальная очистка ---
    # Последний шаг для получения гарантии на значение функции
    
    prox_lambda_final = lambdas[T]
    prox_center_final = x_current
    
    current_mu_final = MU + prox_lambda_final
    current_L_final = L + prox_lambda_final
    
    # На последнем шаге решаем задачу с большей точностью
    x_final, grad_calls_stage, _ = robust_distance_estimator(
        m_trials=m,
        n_iters_sgd=int(n_iters_sgd_weak), # Увеличим кол-во итераций
        start_point=x_current,
        prox_center=prox_center_final,
        prox_lambda=prox_lambda_final,
        strong_convexity=current_mu_final,
        smoothness=current_L_final
    )
    cumulative_grad_calls += grad_calls_stage
    plot_gc_hist.append(cumulative_grad_calls)
    plot_error_hist.append(f(x_final))
    
    return x_final, plot_error_hist, plot_gc_hist

def smooth_values(values, alpha=0.75):
    """Сглаживание ряда значений с помощью экспоненциального скользящего среднего."""
    if not values.size:
        return values
    smoothed = np.zeros_like(values, dtype=float)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i-1]
    return smoothed


def run_all_simulations(n_runs, target_error, target_p_fail):
    """Запускает симуляции для обоих алгоритмов и возвращает их истории."""
    
    all_boost_err_hists = []
    all_boost_gc_hists = []
    all_sgd_err_hists = []
    
    total_grad_calls_for_boost = 0

    for i in tqdm(range(n_runs), desc="Симуляции для графиков"):
        # 1. Запуск BoostAlg
        _, boost_err_hist, boost_gc_hist = prox_boost_alg(target_error, target_p_fail)
        
        if i == 0:
            total_grad_calls_for_boost = int(boost_gc_hist[-1])

        all_boost_err_hists.append(boost_err_hist)
        all_boost_gc_hists.append(boost_gc_hist)

        # 2. Запуск SGD с тем же бюджетом вызовов градиента
        x_sgd = X_INIT.copy()
        step_size = 1.0 / L
        current_sgd_err = [f(X_INIT)]
        for k in range(1, total_grad_calls_for_boost + 1):
            grad_val = stochastic_grad_f(x_sgd)
            x_sgd -= step_size * grad_val
            if k % 500 == 0 or k == total_grad_calls_for_boost:
                current_sgd_err.append(f(x_sgd))
        all_sgd_err_hists.append(current_sgd_err)
        
    return all_boost_err_hists, all_boost_gc_hists, all_sgd_err_hists, total_grad_calls_for_boost


def plot_convergence_results(
    all_boost_err_hists, all_boost_gc_hists, all_sgd_err_hists, 
    total_grad_calls_for_boost, target_error, n_runs
):
    """Обрабатывает результаты и строит график сходимости."""
    
    # --- Обработка и усреднение данных для BoostAlg ---
    # Все gc истории должны быть примерно одинаковы, берем первую как эталон
    common_gc_grid = all_boost_gc_hists[0]
    
    interpolated_boost_errs = []
    for err_hist, gc_hist in zip(all_boost_err_hists, all_boost_gc_hists):
        # Интерполируем каждую траекторию на общую сетку
        interpolated = np.interp(common_gc_grid, gc_hist, err_hist)
        interpolated_boost_errs.append(interpolated)
        
    boost_median = np.median(interpolated_boost_errs, axis=0)
    boost_q25 = np.percentile(interpolated_boost_errs, 25, axis=0)
    boost_q75 = np.percentile(interpolated_boost_errs, 75, axis=0)

    # --- Обработка и усреднение данных для SGD ---
    sgd_gc_hist = np.linspace(0, total_grad_calls_for_boost, len(all_sgd_err_hists[0]))
    sgd_median = np.median(all_sgd_err_hists, axis=0)
    
    # Сглаживаем медиану SGD для более чистого графика
    sgd_median_smoothed = smooth_values(sgd_median)

    sgd_q25 = np.percentile(all_sgd_err_hists, 25, axis=0)
    sgd_q75 = np.percentile(all_sgd_err_hists, 75, axis=0)
    
    # --- Построение графика ---
    plt.figure(figsize=(14, 8))
    
    # BoostAlg (proxBoost)
    plt.plot(common_gc_grid, boost_median, 'o-', label='BoostAlg (Median)', linewidth=2.5, markersize=8, color='C0')
    plt.fill_between(common_gc_grid, boost_q25, boost_q75, color='C0', alpha=0.2, label='BoostAlg (25-75% percentile)')

    # Обычный SGD
    plt.plot(sgd_gc_hist, sgd_median_smoothed, '-', label='SGD (Smoothed Median)', linewidth=2.5, color='C3')
    plt.fill_between(sgd_gc_hist, sgd_q25, sgd_q75, color='C3', alpha=0.2, label='SGD (25-75% percentile)')
    
    plt.axhline(y=target_error, color='k', linestyle=':', label=f'Целевая ошибка = {target_error}')
    
    plt.title(f'Сравнение сходимости BoostAlg и SGD (медиана по {n_runs} запускам)', fontsize=16)
    plt.xlabel('Количество вызовов градиента', fontsize=12)
    plt.ylabel('Ошибка (значение функции f(x))', fontsize=12)
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.legend(fontsize=12)
    plt.show()


# --- Проведение эксперимента и визуализация ---

if __name__ == "__main__":
    
    TARGET_ERROR = 0.001
    TARGET_P_FAIL = 0.05
    N_RUNS = 20 # Количество запусков для оценки вероятности и построения графиков

    # 1. Запуск симуляций для сбора данных
    boost_errs, boost_gcs, sgd_errs, total_gc = run_all_simulations(
        N_RUNS, TARGET_ERROR, TARGET_P_FAIL
    )

    # 2. Построение графика сходимости
    plot_convergence_results(boost_errs, boost_gcs, sgd_errs, total_gc, TARGET_ERROR, N_RUNS)

    # 3. Оценка вероятности ошибки
    print("\nОценка вероятности ошибки...")
    print(f"Бюджет вызовов градиента на один запуск: {total_gc}")
    
    boost_fails = 0
    sgd_fails = 0
    
    for _ in tqdm(range(N_RUNS), desc="Прогоняем симуляции для оценки надежности"):
        # Запуск BoostAlg
        sol_b, _, _ = prox_boost_alg(TARGET_ERROR, TARGET_P_FAIL)
        if f(sol_b) > TARGET_ERROR:
            boost_fails += 1
            
        # Запуск SGD
        sol_s, _, _ = sgd(n_iters=total_gc)
        if f(sol_s) > TARGET_ERROR:
            sgd_fails += 1
            
    # 4. Вывод результатов
    print("\n--- Результаты симуляции ---")
    print(f"Количество запусков: {N_RUNS}")
    print(f"Целевая ошибка: {TARGET_ERROR}\n")
    
    prob_boost_fail = boost_fails / N_RUNS
    prob_sgd_fail = sgd_fails / N_RUNS
    
    print(f"BoostAlg (proxBoost):")
    print(f"  Количество неудач (ошибка > {TARGET_ERROR}): {boost_fails}")
    print(f"  Эмпирическая вероятность неудачи: {prob_boost_fail:.2%} (целевая < {TARGET_P_FAIL:.2%})")
    
    print(f"\nОбычный SGD:")
    print(f"  Количество неудач (ошибка > {TARGET_ERROR}): {sgd_fails}")
    print(f"  Эмпирическая вероятность неудачи: {prob_sgd_fail:.2%}")

    if prob_boost_fail < prob_sgd_fail and prob_boost_fail <= TARGET_P_FAIL :
        print("\nВывод: BoostAlg демонстрирует значительно более высокую надежность,")
        print("достигая требуемой точности с гораздо большей вероятностью, чем обычный SGD,")
        print("при одинаковом бюджете на вычисление градиентов.")
    else:
        print("\nВывод: В данной симуляции BoostAlg не показал явного преимущества в надежности или не достиг цели.")
        print("Возможно, стоит поиграть с параметрами (уровень шума, кол-во итераций в слабом решателе).")
