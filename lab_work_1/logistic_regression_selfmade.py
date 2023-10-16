import numpy as np


class SelfmadeLogisticRegression:  # Определяем класс SelfmadeLogisticRegression.
    def __init__(self, learning_rate=0.1, n_iterations=5000,
                 reg_strength=1):  # Конструктор класса с параметрами по умолчанию.
        self.learning_rate = learning_rate  # Устанавливаем скорость обучения.
        self.n_iterations = n_iterations  # Устанавливаем количество итераций обучения.
        self.reg_strength = reg_strength  # Устанавливаем коэффициент регуляризации.
        self.models_weights = {}  # Инициализируем пустой словарь для хранения весов моделей.
        self.models_biases = {}  # Инициализируем пустой словарь для хранения смещений моделей.

    def sigmoid(self, z):  # Определяем метод для расчета сигмоидной функции.
        return 1 / (1 + np.exp(-z))  # Возвращаем значение сигмоидной функции.

    def softmax(self, z):  # Определяем метод для расчета softmax функции.
        exp_z = np.exp(z - np.max(z, axis=1,
                                  keepdims=True))  # Вычисляем экспоненты и вычитаем максимальное значение для стабильности.
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)  # Возвращаем нормализованные вероятности по классам.

    def log_loss(self, y, predictions):  # Определяем метод для расчета функции потерь log loss.
        epsilon = 1e-15  # Очень маленькое значение для стабильности.
        predictions = np.clip(predictions, epsilon,
                              1 - epsilon)  # Обрезаем предсказания для избежания выхода за границы логарифма.
        loss = - np.sum(y * np.log(predictions))  # Рассчитываем log loss.
        return np.mean(loss)  # Возвращаем среднее значение log loss по всем примерам.

    def fit(self, X, y):  # Определяем метод для обучения модели.
        unique_classes = np.unique(y)  # Получаем уникальные классы в целевой переменной.
        n_classes = len(unique_classes)  # Получаем количество классов.
        n_samples, n_features = X.shape  # Получаем количество примеров и признаков.

        for i, cls in enumerate(unique_classes):  # Итерируемся по уникальным классам.
            binary_y = (y == cls).astype(int)  # Преобразуем многоклассовую задачу в бинарную для текущего класса.
            weights, bias = self.train_model(X, binary_y)  # Обучаем модель для бинарной классификации.
            self.models_weights[i] = weights  # Сохраняем веса модели для текущего класса.
            self.models_biases[i] = bias  # Сохраняем смещение модели для текущего класса.

    def train_model(self, X, y):  # Определяем метод для обучения бинарной модели.
        n_samples, n_features = X.shape  # Получаем количество примеров и признаков.
        weights = np.zeros(n_features)  # Инициализируем веса нулями.
        bias = 0  # Инициализируем смещение нулем.

        for _ in range(self.n_iterations):  # Итерируемся по заданному числу итераций.
            model = np.dot(X, weights) + bias  # Вычисляем модель.
            predictions = self.sigmoid(model)  # Применяем сигмоиду к модели для получения предсказаний.

            dw = (1 / n_samples) * (np.dot(X.T, (
                    predictions - y)) + 2 * self.reg_strength * weights)  # Рассчитываем градиент по весам.
            db = (1 / n_samples) * np.sum(predictions - y)  # Рассчитываем градиент по смещению.

            weights -= self.learning_rate * dw  # Обновляем веса.
            bias -= self.learning_rate * db  # Обновляем смещение.

        return weights, bias  # Возвращаем обученные веса и смещение.

    def predict(self, X):  # Определяем метод для предсказания классов.
        predictions = np.zeros((X.shape[0], len(self.models_weights)))  # Инициализируем массив для предсказаний.

        for i, (weights, bias) in enumerate(zip(self.models_weights.values(),
                                                self.models_biases.values())):  # Итерируемся по сохраненным весам и смещениям.
            model_predictions = np.dot(X, weights) + bias  # Вычисляем предсказания для текущей модели.
            predictions[:, i] = model_predictions  # Сохраняем предсказания в массив.

        return np.argmax(self.softmax(predictions), axis=1)  # Возвращаем индексы классов с наибольшей вероятностью.


def load_data(file_path):  # Определяем функцию для загрузки данных.
    with open(file_path, 'r') as file:  # Открываем файл для чтения.
        lines = file.readlines()  # Считываем все строки из файла.

    header = lines[0].strip().split(',')  # Получаем заголовок данных.
    data_lines = [line.strip().split(',') for line in lines[1:]]  # Преобразуем строки данных в список списков.

    train_features = [list(map(float, line[:-2])) for line in data_lines if
                      line[-1] == 'train']  # Извлекаем признаки для тренировочных данных.
    train_labels = [int(line[-2]) for line in data_lines if
                    line[-1] == 'train']  # Извлекаем метки классов для тренировочных данных.

    test_features = [list(map(float, line[:-2])) for line in data_lines if
                     line[-1] == 'test']  # Извлекаем признаки для тестовых данных.
    # Добавляем лейбл -1 к тестовым данным
    test_labels = [-1] * len(test_features)  # Создаем список из -1 для меток тестовых данных.

    return (
        np.array(train_features),  # Возвращаем массив тренировочных признаков.
        np.array(train_labels),  # Возвращаем массив меток тренировочных данных.
        np.array(test_features),  # Возвращаем массив тестовых признаков.
        np.array(test_labels)  # Возвращаем массив меток тестовых данных.
    )


def save_predictions(predictions, file_path):  # Определяем функцию для сохранения предсказаний.
    np.savetxt(file_path, predictions, fmt='%d',
               delimiter=',')  # Сохраняем предсказания в файл с форматом '%d' и разделителем ','.


x_train, y_train, x_test, tt_test = load_data('input.txt')  # Загружаем данные из файла 'input.txt'.

model = SelfmadeLogisticRegression(n_iterations=5000)  # Создаем экземпляр модели с указанными параметрами.
model.fit(x_train, y_train)  # Обучаем модель на тренировочных данных.

selfmade_predictions = model.predict(x_test)  # Предсказываем классы для тестовых данных.

save_predictions(selfmade_predictions, 'output.txt')  # Сохраняем предсказания в файл 'output.txt'.
