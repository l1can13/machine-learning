import numpy as np


class SelfmadeLogisticRegression:
    def __init__(self, learning_rate=0.1, n_iterations=5000, reg_strength=1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.reg_strength = reg_strength
        self.models_weights = {}
        self.models_biases = {}

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def log_loss(self, y, predictions):
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        loss = - np.sum(y * np.log(predictions))
        return np.mean(loss)

    def fit(self, X, y):
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        n_samples, n_features = X.shape

        for i, cls in enumerate(unique_classes):
            binary_y = (y == cls).astype(int)
            weights, bias = self.train_model(X, binary_y)
            self.models_weights[i] = weights
            self.models_biases[i] = bias

    def train_model(self, X, y):
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        bias = 0

        for _ in range(self.n_iterations):
            model = np.dot(X, weights) + bias
            predictions = self.sigmoid(model)

            dw = (1 / n_samples) * (np.dot(X.T, (predictions - y)) + 2 * self.reg_strength * weights)
            db = (1 / n_samples) * np.sum(predictions - y)

            weights -= self.learning_rate * dw
            bias -= self.learning_rate * db

        return weights, bias

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.models_weights)))

        for i, (weights, bias) in enumerate(zip(self.models_weights.values(), self.models_biases.values())):
            model_predictions = np.dot(X, weights) + bias
            predictions[:, i] = model_predictions

        return np.argmax(self.softmax(predictions), axis=1)


def load_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    header = lines[0].strip().split(',')
    data_lines = [line.strip().split(',') for line in lines[1:]]

    train_features = [list(map(float, line[:-2])) for line in data_lines if line[-1] == 'train']
    train_labels = [int(line[-2]) for line in data_lines if line[-1] == 'train']

    test_features = [list(map(float, line[:-2])) for line in data_lines if line[-1] == 'test']
    # Добавляем лейбл -1 к тестовым данным
    test_labels = [-1] * len(test_features)

    return (
        np.array(train_features),
        np.array(train_labels),
        np.array(test_features),
        np.array(test_labels)
    )


def save_predictions(predictions, file_path):
    np.savetxt(file_path, predictions, fmt='%d', delimiter=',')


x_train, y_train, x_test, tt_test = load_data('input.txt')

model = SelfmadeLogisticRegression(n_iterations=5000)
model.fit(x_train, y_train)

selfmade_predictions = model.predict(x_test)

save_predictions(selfmade_predictions, 'output.txt')
