import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

def load_and_preprocess_data():
    iris = load_iris()
    x = iris.data  
    y = iris.target.reshape(-1, 1)  

    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    y_onehot = encoder.fit_transform(y)

    scaler = preprocessing.MinMaxScaler()
    x_scaled = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y_onehot, random_state=1, test_size=0.2
    )
    _, _, y_train_label, y_test_label = train_test_split(
        x_scaled, iris.target, random_state=1, test_size=0.2
    )
    return x_train, x_test, y_train, y_test, y_train_label, y_test_label, scaler

def softmax(z):
    if z.ndim == 1:
        z = z - np.max(z)
        exp_z = np.exp(z)
        return exp_z / exp_z.sum()
    else:
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

def cross_entropy_loss(y_true, y_pred, epsilon=1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    n_samples = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred)) / n_samples
    return loss

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = softmax(self.z2)
        return self.a2

    def backward(self, x, y_true, y_pred, learning_rate=0.1):
        n_samples = x.shape[0]

        dz2 = y_pred - y_true
        dW2 = np.dot(self.a1.T, dz2) / n_samples
        db2 = np.sum(dz2, axis=0, keepdims=True) / n_samples

        dz1 = np.dot(dz2, self.W2.T) * relu_derivative(self.z1)
        dW1 = np.dot(x.T, dz1) / n_samples
        db1 = np.sum(dz1, axis=0, keepdims=True) / n_samples

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

def main():
    x_train, x_test, y_train, y_test, y_train_label, y_test_label, scaler = load_and_preprocess_data()

    input_size = 4       
    hidden_size = 8      
    output_size = 3      
    epochs = 1500        
    learning_rate = 0.5  

    nn = NeuralNetwork(input_size, hidden_size, output_size)
    losses = []

    print(f"输入维度:{input_size} | 隐层神经元:{hidden_size} | 输出类别:{output_size}")

    for epoch in range(epochs):
        y_pred = nn.forward(x_train)
        loss = cross_entropy_loss(y_train, y_pred)
        nn.backward(x_train, y_train, y_pred, learning_rate)

        if epoch % 100 == 0:
            losses.append(loss)
            print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

    train_pred = np.argmax(nn.forward(x_train), axis=1)
    test_pred = np.argmax(nn.forward(x_test), axis=1)
    train_acc = accuracy_score(y_train_label, train_pred)
    test_acc = accuracy_score(y_test_label, test_pred)

    print(f"训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")

if __name__ == "__main__":
    main()