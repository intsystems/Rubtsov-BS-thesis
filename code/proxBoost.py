import numpy as np
from typing import Callable, List, Tuple, Optional
import matplotlib.pyplot as plt
import time
import scipy.special

class BoostAlg:
    """
    Реализация алгоритма BoostAlg
    """
    
    def __init__(
        self, 
        func: Callable[[np.ndarray], float],
        grad: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray],
        mu: float,
        L: float,
        dimension: int = 100,
        noise_generator: Callable[[int], np.ndarray] = None,
        sigma: float = 1.0,
        max_iter_per_alg: int = 20000,
        flag: bool = True
    ):
        """
        Инициализация алгоритма BoostAlg.
        
        Параметры:
        ----------
        func : Callable[[np.ndarray], float]
            Целевая функция f(x)
        grad : Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]
            Стохастический градиент целевой функции ∇f(x, ξ)
        mu : float
            Параметр сильной выпуклости μ
        L : float
            Параметр гладкости L
        dimension : int
            Размерность пространства
        noise_generator : Callable[[int], np.ndarray]
            Функция генерации случайного вектора ξ
        sigma : float
            Параметр ограничения дисперсии E[||ξ||²] ≤ σ²
        """
        self.func = func
        self.grad = grad
        self.mu = mu
        self.L = L
        self.dimension = dimension
        self.noise_generator = noise_generator
        self.sigma = sigma
        self.max_iter_per_alg = max_iter_per_alg
        self.flag = flag
    
    def _phi_x(self, y: np.ndarray, x: np.ndarray, lambda_val: float) -> float:
        """
        Вычисляет значение функции phi_x(y) = f(y) + (λ/2) * ||y - x||^2.
        
        Параметры:
        ----------
        y : np.ndarray
            Точка y
        x : np.ndarray
            Центр x
        lambda_val : float
            Коэффициент штрафа λ
            
        Возвращает:
        -----------
        float
            Значение функции phi_x(y)
        """
        return self.func(y) + (lambda_val / 2) * np.sum((y - x) ** 2)
    
    def _grad_phi_x(self, y: np.ndarray, x: np.ndarray, lambda_val: float, noise: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Вычисляет стохастический градиент функции phi_x(y).
        
        Параметры:
        ----------
        y : np.ndarray
            Точка y
        x : np.ndarray
            Центр x
        lambda_val : float
            Коэффициент штрафа λ
        noise : Optional[np.ndarray]
            Случайный вектор ξ
            
        Возвращает:
        -----------
        np.ndarray
            Стохастический градиент функции phi_x(y)
        """
        return self.grad(y, noise) + lambda_val * (y - x)
    
    def _alg(self, delta: float, lambda_val: float, delta_in: float, x: np.ndarray, track_trajectory: bool = False) -> Tuple[np.ndarray, List[float], List[np.ndarray]]:
        """
        Реализация процедуры Alg(δ, λ, Δ, x) для приближенной минимизации phi_x(y).
        
        Параметры:
        ----------
        delta : float
            Требуемая точность
        lambda_val : float
            Коэффициент штрафа λ
        delta_in : float
            Верхняя граница начальной ошибки
        x : np.ndarray
            Центр квадратичного возмущения
        track_trajectory : bool
            Флаг для отслеживания траектории
            
        Возвращает:
        -----------
        Tuple[np.ndarray, List[float], List[np.ndarray]]
            Приближенный минимум, значения функции на каждой итерации, точки траектории
        """
        x_star_comp1 = lambda_val / (self.L + lambda_val) * x
        x_star_comp2 = lambda_val / (self.mu + lambda_val) * x
        x_star = np.array([x_star_comp1, x_star_comp2])
        y = np.copy(x)  # Начинаем с точки x
        func_values = []
        trajectory = []
        epsilon = np.sqrt(2*delta / (self.mu + lambda_val))
        alpha = 0.001  # Шаг SGD
        max_iterations = self.max_iter_per_alg
        
        for i in range(max_iterations):
            if self.noise_generator is not None:
                noise = self.noise_generator(self.dimension)
            else:
                noise = np.random.normal(0, 1, size=self.dimension)
            
            # Вычисляем стохастический градиент
            gradient = self._grad_phi_x(y, x, lambda_val, noise)
            
            
            # Проверка на валидность градиента и его ограничение
            if np.any(np.isnan(gradient)) or np.any(np.isinf(gradient)):
                # Если градиент содержит невалидные значения, пропускаем шаг
                print("Gradient is invalid")
                continue
                
            # Ограничиваем норму градиента для предотвращения расходимости
            grad_norm = np.linalg.norm(gradient)
            if grad_norm > 10.0:  # Максимальная допустимая норма градиента
                gradient = gradient * (10.0 / grad_norm)
            
            # Делаем шаг SGD
            y = y - alpha * gradient
            
            # Проверка на валидность точки
            if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                # Восстанавливаем предыдущее значение
                y = np.copy(x)
                break
            
            # Сохраняем значения
            if track_trajectory:
                func_value = self.func(y)
                # Проверяем, что значение функции валидно
                if not np.isnan(func_value) and not np.isinf(func_value):
                    func_values.append(func_value)
                    trajectory.append(np.copy(y))
            
            #sufficient_num_of_iter = max_iterations
            if np.linalg.norm(y - x_star) < epsilon/3 and self.flag:
                #self.max_iter_per_alg = len(func_values)
                #self.flag = False
                break
        
        return y, func_values, trajectory
    
    def _robust_distance_estimator(self, delta: float, lambda_val: float, delta_in: float, x: np.ndarray, m: int) -> np.ndarray:
        """
        Реализация алгоритма Alg-R(δ, λ, Δ, x, m).
        
        Параметры:
        ----------
        delta : float
            Требуемая точность
        lambda_val : float
            Коэффициент штрафа λ
        delta_in : float
            Верхняя граница начальной ошибки
        x : np.ndarray
            Центр квадратичного возмущения
        m : int
            Количество запросов к Alg
            
        Возвращает:
        -----------
        np.ndarray
            Точка, наиболее близкая к истинному минимуму по метрике радиуса
        """
        # Запрашиваем m раз Alg и собираем ответы
        points = []
        for _ in range(m):
            y, _, _ = self._alg(delta, lambda_val, delta_in, x)
            points.append(y)
        
        # Вычисляем радиусы для каждой точки
        radii = []
        for i in range(m):
            # Расстояния от yi до всех других точек
            distances = [np.linalg.norm(points[i] - points[j]) for j in range(m)]
            distances.sort()
            # Минимальный радиус r, для которого |Br(yi) ∩ Y| > m/2
            ri = distances[m//2]
            radii.append(ri)
        
        # Находим индекс с минимальным радиусом
        i_star = np.argmin(radii)
        
        # Возвращаем точку yi*
        return points[i_star]
    
    def run(self, delta: float, delta_in: float, x_in: np.ndarray, T: int, m: int, 
            track_trajectory: bool = False) -> Tuple[np.ndarray, List[float], List[np.ndarray]]:
        """
        Запуск алгоритма BoostAlg с отслеживанием всех внутренних итераций.
        
        Параметры:
        ----------
        delta : float
            Требуемая точность
        delta_in : float
            Верхняя граница начальной ошибки
        x_in : np.ndarray
            Начальная точка
        T : int
            Количество итераций - максимальное количество итераций
        m : int
            Количество запросов к Alg для робастной оценки расстояния
        track_trajectory : bool
            Флаг для отслеживания траектории
            
        Возвращает:
        -----------
        Tuple[np.ndarray, List[float], List[np.ndarray]]
            Итоговая точка, значения функции на каждой итерации, точки траектории
        """
        # Инициализация
        lambda_prev = 0
        delta_prev = delta_in
        x_prev = np.copy(x_in)
        
        # Списки для хранения значений
        all_func_values = [self.func(x_prev)]
        all_trajectory = [np.copy(x_prev)]
        
        # Возвращаем исходную последовательность λj 
        lambda_sequence = [2**j * self.mu for j in range(T+1)]  # Стандартная последовательность из теории
        
        # Алгоритм BoostAlg
        for j in range(T+1):
            # Запрашиваем m раз Alg и собираем ответы и все итерации
            all_points = []
            all_inner_values = []  # Список для хранения всех значений функции
            all_inner_traj = []    # Список для хранения всех траекторий
            
            for i in range(m):
                # Запускаем Alg с отслеживанием траектории
                y, inner_values, inner_traj = self._alg(delta/9, lambda_prev, delta_prev, x_prev, track_trajectory=True)
                all_points.append(y)
                
                # Сохраняем значения функции и траектории отдельно для каждого запуска
                if track_trajectory:
                    all_inner_values.append(inner_values)
                    all_inner_traj.append(inner_traj)
            
            # Находим лучшую точку среди m запусков
            radii = []
            for i in range(m):
                # Расстояния от yi до всех других точек
                distances = [np.linalg.norm(all_points[i] - all_points[j]) for j in range(m)]
                distances.sort()
                # Минимальный радиус r, для которого |Br(yi) ∩ Y| > m/2
                ri = distances[m//2]
                radii.append(ri)
            
            # Находим индекс с минимальным радиусом
            i_star = np.argmin(radii)
            x_j = all_points[i_star]
            
            # Добавляем в общие списки только значения и траектории от лучшего запуска
            if track_trajectory and all_inner_values and i_star < len(all_inner_values):
                all_func_values.extend(all_inner_values[i_star])
                all_trajectory.extend(all_inner_traj[i_star])
                '''
                # Проверяем, что значения не содержат NaN или бесконечностей
                valid_values = [v for v in all_inner_values[i_star] if not np.isnan(v) and not np.isinf(v)]
                if valid_values:
                    all_func_values.extend(valid_values)
                    
                valid_indices = [i for i, v in enumerate(all_inner_values[i_star]) 
                                if not np.isnan(v) and not np.isinf(v)]
                if valid_indices:
                    valid_traj = [all_inner_traj[i_star][i] for i in valid_indices]
                    all_trajectory.extend(valid_traj)
                '''
            
            # Вычисляем новую верхнюю границу ошибки (строго по формуле из алгоритма 7)
            delta_j = delta * ((self.L + lambda_prev) / (self.mu + lambda_prev))
            for i in range(j):
                delta_j += lambda_sequence[i] / (self.mu + lambda_sequence[i-1])
            
            # Обновляем для следующей итерации
            lambda_prev = lambda_sequence[j]
            delta_prev = delta_j
            x_prev = x_j
            print(len(all_func_values))
            if len(all_func_values) > max_iter:
                break
        
        # Финальный шаг - очистка
        final_ratio = (self.mu + lambda_prev) / (self.L + lambda_prev)
        
        # Снова запускаем с отслеживанием траектории
        all_points = []
        all_inner_values = []
        all_inner_traj = []
        
        for i in range(m):
            y, inner_values, inner_traj = self._alg(final_ratio * delta/9, lambda_prev, delta_prev, x_prev, track_trajectory=True)
            all_points.append(y)
            
            # Сохраняем значения функции и траектории отдельно для каждого запуска
            if track_trajectory:
                all_inner_values.append(inner_values)
                all_inner_traj.append(inner_traj)
        
        # Находим финальную точку
        radii = []
        for i in range(len(all_points)):
            distances = [np.linalg.norm(all_points[i] - all_points[j]) for j in range(len(all_points))]
            distances.sort()
            ri = distances[len(all_points)//2]
            radii.append(ri)
        
        i_star = np.argmin(radii)
        x_final = all_points[i_star]
        
        # Добавляем в общие списки только значения и траектории от лучшего запуска
        if track_trajectory and all_inner_values and i_star < len(all_inner_values):
            # Проверяем, что значения не содержат NaN или бесконечностей
            valid_values = [v for v in all_inner_values[i_star] if not np.isnan(v) and not np.isinf(v)]
            if valid_values:
                all_func_values.extend(valid_values)
                
            valid_indices = [i for i, v in enumerate(all_inner_values[i_star]) 
                            if not np.isnan(v) and not np.isinf(v)]
            if valid_indices:
                valid_traj = [all_inner_traj[i_star][i] for i in valid_indices]
                all_trajectory.extend(valid_traj)

        
        
        return x_final, all_func_values, all_trajectory


# Генераторы шума 
def gaussian_noise_generator(dimension):
    """Генератор шума с нормальным распределением."""
    return np.random.normal(0, 1, size=dimension)

def weibull_noise_generator(dimension):
    """Генератор шума с распределением Вейбулла."""
    c = 0.2
    # Параметр alpha
    alpha = 1.0 / np.sqrt(scipy.special.gamma(1 + 2/c) - (scipy.special.gamma(1 + 1/c))**2)
    
    # Генерируем случайные числа
    values = np.random.weibull(c, size=dimension) * alpha
    
    # Сдвигаем для нулевого среднего
    shift = alpha * scipy.special.gamma(1 + 1/c)
    values = values - shift
    
    return values

def burr_noise_generator(dimension):
    """Генератор шума с распределением Бурра XII типа."""
    c = 1.0
    d = 2.3
    
    # Генерируем равномерно распределенные случайные числа
    u = np.random.uniform(0, 1, size=dimension)
    
    # Преобразуем в распределение Бурра через обратную функцию CDF
    values = np.power(np.power(1 - u, -1/d) - 1, 1/c)
    
    # Бета-функция
    def beta_func(r):
        return scipy.special.beta(d - r/c, 1 + r/c)
    
    # Находим среднее и дисперсию 
    mu1 = d * beta_func(1)
    mu2 = d * beta_func(2)
    variance = mu2 - mu1**2
    
    # Нормализуем
    values = (values - mu1) / np.sqrt(variance)
    
    return values

# Добавляем функцию сглаживания перед её использованием
def smooth_trajectory(trajectory, window_size=5):
    """Сглаживает траекторию с помощью скользящего среднего."""
    smoothed = np.zeros_like(trajectory)
    n = len(trajectory)
    
    for i in range(n):
        start = max(0, i - window_size // 2)
        end = min(n, i + window_size // 2 + 1)
        smoothed[i] = np.mean(trajectory[start:end], axis=0)
    
    return smoothed

# Генерация случайной точки в шаре радиуса 1
def random_point_in_ball(dimension, radius=1.0):
    """Генерирует случайную точку в шаре заданного радиуса."""
    # Генерируем точку из нормального распределения
    point = np.random.normal(0, 1, size=dimension)
    # Нормализуем вектор
    point = point / np.linalg.norm(point)
    # Умножаем на случайную длину в пределах радиуса
    r = np.random.random() ** (1.0 / dimension) * radius
    return point * r

# Пример использования
if __name__ == "__main__":
    # Определяем функцию f(x) = ||x||²/2 и стохастический градиент ∇f(x, ξ) = x + ξ
    L = 100.0
    mu = 0.0
    def f(x):
        return 0.5 * (L *x[0]**2 + mu * x[1]**2)
    
    def stochastic_grad_f(x, noise=None):
        if noise is None:
            return x
        return x + noise
    
    # Параметры экспериментов
    dimension = 2      # Размерность 2 для визуализации
    delta = 0.001       # Требуемая точность
    delta_in = 1.0     # Начальная верхняя граница ошибки
    T = 30             # Количество итераций BoostAlg
    m = 30             # Количество запросов для робастной оценки расстояния
    max_iter = 30000  # Увеличено с 30000 до 100000
    gamma = 0.001      # Шаг SGD
    
    # Генерируем случайную начальную точку из шара радиуса 1
    x0 = random_point_in_ball(dimension, radius=1.0)
    print(f"Начальная точка: {x0}")
    x_star = np.array([0.0, 0.0])

    def regularized_f(x):
        return f(x) + delta * (np.linalg.norm(x - x0))**2 /2 / np.linalg.norm(x0 - x_star)**2
    
    def regularized_grad_f(x, noise=None):
        return stochastic_grad_f(x, noise) + delta * (x - x0) / np.linalg.norm(x0 - x_star)**2
    
    # Список генераторов шума
    noise_generators = [
        (gaussian_noise_generator, "Нормальное распределение"),
        (weibull_noise_generator, "Распределение Вейбулла"),
        (burr_noise_generator, "Распределение Бурра")
    ]
    
    # Словарь для хранения результатов
    results = {}
    
    # Проводим эксперименты для каждого генератора шума
    for noise_gen, name in noise_generators:
        print(f"\nЗапуск экспериментов с шумом: {name}")
        
        # Создаем экземпляр BoostAlg с соответствующим генератором шума
        boost_alg = BoostAlg(
            regularized_f, regularized_grad_f, 
            mu=delta / np.linalg.norm(x0 - x_star)**2, L=L, dimension=dimension,
            noise_generator=noise_gen, 
            sigma=np.sqrt(dimension)
        )
        
        # Запускаем простой SGD для сравнения
        start_time = time.time()
        
        # Запускаем SGD
        x = np.copy(x0)
        sgd_trajectory = [np.copy(x)]
        sgd_func_values = [f(x)]
        
        for i in range(max_iter):
            # Генерируем шум
            noise = noise_gen(dimension)
            
            # Делаем шаг SGD
            x = x - gamma * (x + noise)
            
            # Сохраняем значения
            sgd_func_values.append(f(x))
            sgd_trajectory.append(np.copy(x))
        
        sgd_time = time.time() - start_time
        print(f"Время выполнения SGD: {sgd_time:.4f} секунд")
        print(f"Итоговое значение функции SGD: {sgd_func_values[-1]:.8f}")
        
        # Запускаем BoostAlg
        start_time = time.time()
        result, boost_func_values, boost_trajectory = boost_alg.run(
            delta, delta_in, x0, T, m, track_trajectory=True
        )
        boost_time = time.time() - start_time
        
        print(f"Время выполнения BoostAlg: {boost_time:.4f} секунд")
        print(f"Итоговое значение функции BoostAlg: {f(result):.8f}")
        
        # Сохраняем результаты
        results[name] = {
            'sgd_trajectory': sgd_trajectory,
            'sgd_func_values': sgd_func_values,
            'sgd_time': sgd_time,
            'boost_trajectory': boost_trajectory,
            'boost_func_values': boost_func_values,
            'boost_time': boost_time,
            'final_point': result
        }
    
    # Визуализация результатов - три графика невязки и три графика траекторий
    fig = plt.figure(figsize=(18, 12))

    # Определим цвета для каждого распределения (для SGD)
    colors = {
        "Нормальное распределение": "blue",
        "Распределение Вейбулла": "red",
        "Распределение Бурра": "green"
    }

    # Создаем три подграфика для невязок
    for i, (noise_gen, name) in enumerate(noise_generators):
        plt.subplot(2, 3, i+1)
        
        data = results[name]
        color = colors[name]
        
        # Для SGD - нормализуем относительно начальной невязки
        sgd_values = np.array(data['sgd_func_values'])
        initial_value = sgd_values[0]  # Начальная невязка
        normalized_sgd_values = sgd_values / initial_value  # Нормализация
        
        # Вычисляем вероятности для SGD
        sgd_prob_1e2 = np.mean(normalized_sgd_values[5000:] < 1e-2)
        sgd_prob_1e3 = np.mean(normalized_sgd_values[5000:] < 1e-3)
        sgd_prob_1e4 = np.mean(normalized_sgd_values[5000:] < 1e-4)
        
        # Создаем метку с вероятностями
        sgd_label = f'SGD (P<1e-2: {sgd_prob_1e2:.3f}, P<1e-3: {sgd_prob_1e3:.3f}, P<1e-4: {sgd_prob_1e4:.3f})'
        
        # Рисуем график SGD с обновленной легендой
        plt.plot(np.arange(len(normalized_sgd_values)), normalized_sgd_values, 
                 color=color, linestyle='-', label=sgd_label)
        
        # Для BoostAlg
        if data['boost_func_values']:
            boost_values = np.array(data['boost_func_values'])
            initial_value = boost_values[0]
            normalized_boost_values = boost_values[:min(len(boost_values), max_iter)] / initial_value
            
            # Вычисляем вероятности для BoostAlg
            boost_prob_1e2 = np.mean(normalized_boost_values[5000:] < 1e-2)
            boost_prob_1e3 = np.mean(normalized_boost_values[5000:] < 1e-3)
            boost_prob_1e4 = np.mean(normalized_boost_values[5000:] < 1e-4)
            
            # Создаем метку с вероятностями
            boost_label = f'BoostAlg (P<1e-2: {boost_prob_1e2:.3f}, P<1e-3: {boost_prob_1e3:.3f}, P<1e-4: {boost_prob_1e4:.3f})'
            
            plt.plot(np.arange(len(normalized_boost_values)), normalized_boost_values, 
                    color='black', linestyle='-', label=boost_label)
        
        plt.xlabel('Итерации')
        plt.ylabel('f(x)/f(x₀)')
        plt.title(f'Сходимость для {name}')
        plt.yscale('log')  # Логарифмическая шкала для лучшей визуализации
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)  # Уменьшаем размер шрифта в легенде для лучшей читаемости


    # Создаем три подграфика для траекторий
    for i, (noise_gen, name) in enumerate(noise_generators):
        plt.subplot(2, 3, i+4)
        
        data = results[name]
        color = colors[name]
        
        # Траектория SGD
        sgd_traj = np.array(data['sgd_trajectory'])
        
        # Отбираем часть точек для визуализации
        sample_step = len(sgd_traj) // 1000 if len(sgd_traj) > 1000 else 1
        sgd_traj_sampled = sgd_traj[::sample_step]
        
        # Рисуем траекторию SGD
        plt.plot(sgd_traj_sampled[:, 0], sgd_traj_sampled[:, 1], 
                 color=color, linestyle='-', label='SGD', alpha=0.6)
        
        # Отмечаем начальную и конечную точки
        plt.scatter(sgd_traj[0][0], sgd_traj[0][1], color=color, s=100, marker='o', label='Начало')
        plt.scatter(sgd_traj[-1][0], sgd_traj[-1][1], color=color, s=100, marker='*', label='Конец SGD')
        
        # Траектория BoostAlg с дополнительным сглаживанием
        if data['boost_trajectory']:
            boost_traj = np.array(data['boost_trajectory'])
            
            # Применяем сглаживание траектории
            smoothed_boost_traj = smooth_trajectory(boost_traj, window_size=50)
            
            # Отбираем часть точек для визуализации
            sample_step = len(smoothed_boost_traj) // 1000 if len(smoothed_boost_traj) > 1000 else 1
            boost_traj_sampled = smoothed_boost_traj[::sample_step]
            
            # Рисуем сглаженную траекторию BoostAlg
            plt.plot(boost_traj_sampled[:, 0], boost_traj_sampled[:, 1], 
                     color='black', linestyle='--', label='BoostAlg', alpha=0.8)
            
            # Отмечаем конечную точку
            plt.scatter(smoothed_boost_traj[-1][0], smoothed_boost_traj[-1][1], 
                        color='black', s=100, marker='*', label='Конец BoostAlg')
        
        # Отмечаем оптимальную точку (0, 0)
        plt.scatter(0, 0, color='red', s=150, marker='x', label='Оптимум')
        
        # Добавляем дополнительную информацию
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        
        plt.xlabel('x_1')
        plt.ylabel('x_2')
        plt.title(f'Траектории для {name}')
        plt.grid(True, alpha=0.3)
        
        # Устанавливаем одинаковый масштаб по осям
        plt.axis('equal')
        
        # Добавляем легенду
        plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('sgd_vs_boostalg_trajectories.png')
    plt.show()