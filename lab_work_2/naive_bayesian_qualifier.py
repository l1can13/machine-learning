import numpy as np  # Импорт библиотеки для работы с массивами и матрицами


class SelfmadeNaiveBayesClassifier:
    """
    Наивный Байесовский классификатор.

    Attributes:
        num_labels (int): Количество уникальных меток.
        label_data (list): Данные для каждой метки.
        label_means (list): Средние значения для каждого признака для каждой метки.
        label_stds (list): Стандартные отклонения для каждого признака для каждой метки.
        label_priors (list): Априорные вероятности для каждой метки.
    """

    def __init__(self):
        """
        Инициализация объекта класса.
        """
        self.num_labels = 0  # Инициализация количества уникальных меток
        self.label_data = []  # Инициализация списка данных для каждой метки
        self.label_means = []  # Инициализация списка средних значений для каждого признака и метки
        self.label_stds = []  # Инициализация списка стандартных отклонений для каждого признака и метки
        self.label_priors = []  # Инициализация списка априорных вероятностей для каждой метки

    def fit(self, training_features, training_labels):
        """
        Обучение классификатора на основе предоставленных данных.

        Parameters:
            training_features (numpy.ndarray): Массив тренировочных признаков.
            training_labels (numpy.ndarray): Массив меток для тренировочных данных.

        Returns:
            None
        """
        self.num_labels = len(np.unique(training_labels))  # Определение количества уникальных меток
        self.label_data = [training_features[training_labels == label] for label in
                           range(self.num_labels)]  # Разделение данных для каждой метки
        self.label_means = [np.nanmean(data, axis=0) if len(data) > 0 else np.zeros(training_features.shape[1]) for data
                            in self.label_data]  # Расчет средних значений
        self.label_stds = [np.nanstd(data, axis=0) if len(data) > 0 else np.ones(training_features.shape[1]) for data in
                           self.label_data]  # Расчет стандартных отклонений
        total_samples = len(training_labels)
        self.label_priors = [len(data) / total_samples for data in self.label_data]  # Расчет априорных вероятностей

    def predict(self, test_features):
        """
        Предсказание меток для тестовых данных.

        Parameters:
            test_features (numpy.ndarray): Массив тестовых признаков.

        Returns:
            list: Список предсказанных меток.
        """
        predictions = []  # Инициализация списка предсказанных меток
        for test_instance in test_features:
            probabilities = []  # Инициализация списка вероятностей
            for label in range(self.num_labels):
                prior_probability = self.label_priors[label]  # Получение априорной вероятности для метки
                for feature_index in range(len(test_instance)):
                    probability_density = self.calculate_probability(test_instance[feature_index],
                                                                     self.label_means[label][feature_index],
                                                                     self.label_stds[label][feature_index])
                    prior_probability *= probability_density  # Обновление апостериорной вероятности
                probabilities.append(prior_probability)
            predicted_label = np.argmax(probabilities)  # Определение метки с наибольшей вероятностью
            predictions.append(predicted_label)
        return predictions

    @staticmethod
    def calculate_probability(value, mean, std):
        """
        Вычисление вероятности с использованием нормального распределения.

        Parameters:
            value (float): Значение признака.
            mean (float): Среднее значение признака для метки.
            std (float): Стандартное отклонение признака для метки.

        Returns:
            float: Рассчитанная вероятность.
        """
        if not std:
            return 1
        exponent = np.exp(-((value - mean) ** 2) / (2 * std ** 2))
        return (1 / (std * np.sqrt(2 * np.pi))) * exponent


def load_data(file_path):
    """
    Загрузка данных из файла.

    Parameters:
        file_path (str): Путь к файлу с данными.

    Returns:
        tuple: Кортеж с данными (train_features, train_labels, test_features, test_labels).
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    header = lines[0].strip().split(',')
    data_lines = [line.strip().split(',') for line in lines[1:]]
    train_features = [list(map(float, line[:-2])) for line in data_lines if line[-1] == 'train']
    train_labels = [int(line[-2]) for line in data_lines if line[-1] == 'train']
    test_features = [list(map(float, line[:-2])) for line in data_lines if line[-1] == 'test']
    test_labels = [-1] * len(test_features)
    return (
        np.array(train_features),
        np.array(train_labels),
        np.array(test_features),
        np.array(test_labels)
    )


def save_predictions(predictions, file_path):
    """
    Сохранение предсказаний в файл.

    Parameters:
        predictions (list): Список предсказанных меток.
        file_path (str): Путь к файлу для сохранения.

    Returns:
        None
    """
    np.savetxt(file_path, predictions, fmt='%d', delimiter='\n')


# Загрузка данных из входного файла
train_features, train_labels, test_features, _ = load_data("input.txt")

# Создание экземпляра классификатора
classifier = SelfmadeNaiveBayesClassifier()

# Обучение классификатора на тренировочных данных
classifier.fit(train_features, train_labels)

# Предсказание меток для тестовых данных
predictions = classifier.predict(test_features)

# Сохранение предсказаний в выходной файл
save_predictions(predictions, "output.txt")
