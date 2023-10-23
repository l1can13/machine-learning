import numpy as np


class SelfmadeKNearestNeighbors:
    """
    Простая реализация алгоритма k-ближайших соседей.

    Parameters:
        k (int): Количество соседей для учета при предсказании.

    Attributes:
        k (int): Количество соседей.
        X_train (numpy.ndarray): Обучающие данные.
        y_train (numpy.ndarray): Метки классов для обучающих данных.
    """

    def __init__(self, k=5):
        """
        Инициализация объекта SelfmadeKNearestNeighbors.

        Parameters:
            k (int, optional): Количество соседей для учета при предсказании.
                               По умолчанию установлено значение 5.
        """
        self.k = k

    def fit(self, X_train, y_train):
        """
        Обучение модели на обучающих данных.

        Parameters:
            X_train (list or numpy.ndarray): Обучающие данные.
            y_train (list or numpy.ndarray): Метки классов для обучающих данных.

        Returns:
            None
        """
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def predict(self, X_test):
        """
        Предсказание меток классов для тестовых данных.

        Parameters:
            X_test (list or numpy.ndarray): Тестовые данные.

        Returns:
            numpy.ndarray: Предсказанные метки классов для тестовых данных.
        """
        X_test = np.array(X_test)

        predictions = []
        for x in X_test:
            distances = np.sum((self.X_train - x) ** 2, axis=1)
            nearest_neighbors_indices = np.argpartition(distances, self.k)[:self.k]
            k_nearest_labels = self.y_train[nearest_neighbors_indices]

            unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
            predicted_label = unique_labels[np.argmax(counts)]
            predictions.append(predicted_label)

        return np.array(predictions)


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


# Загрузка данных
train_features, train_labels, test_features, _ = load_data('input.txt')

# Создание и обучение модели
knn = SelfmadeKNearestNeighbors(k=5)
knn.fit(train_features, train_labels)

# Предсказание меток для тестовых данных
test_predictions = knn.predict(test_features)

# Сохранение предсказаний в файл
save_predictions(test_predictions, 'output.txt')
