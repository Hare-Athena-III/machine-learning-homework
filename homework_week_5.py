from sklearn import preprocessing

import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

 

# 一、对数据集进行处理，使其符合训练要求

# 二、划分训练集与测试集

def load_and_preprocess_data():

    """

    加载Iris数据集并进行预处理，为三分类Softmax回归做准备。

    返回归一化后的特征、one-hot编码的标签，以及用于后续反归一化的scaler。

    """

    # 定义数据类型，与文档中处理iris.data的方式一致

    dtype = np.dtype([('sepal_length', np.float32),

                     ('sepal_width',np.float32),

                     ('petal_length', np.float32),

                     ('petal_width', np.float32),

                     ('class','U15')])

    data = np.loadtxt('iris.data', delimiter=',', dtype=dtype)

 

    # 分离特征和标签

    x = np.column_stack([data['sepal_length'],

                         data['sepal_width'],

                         data['petal_length'],

                         data['petal_width']])

    # 将类别字符串转换为数值标签 0, 1, 2

    y = np.array([0 if cls=='Iris-setosa' else 1 if cls=='Iris-versicolor' else 2

                  for cls in data['class']])

    y = y.reshape(-1, 1)  # 转换为列向量 (N, 1)

 

    # 为三分类生成one-hot标签

  

    encoder = OneHotEncoder(sparse_output=False, categories='auto')

    y_onehot = encoder.fit_transform(y)  # 形状变为 (N, 3)

 

    # 特征归一化

    m = preprocessing.MinMaxScaler()

    x_scaled = m.fit_transform(x)

 

    # 划分训练集和测试集，测试集占比20%

    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_onehot, random_state=1, test_size=0.2)

    # 同时返回数值标签版本用于最终的准确率计算（与one-hot对应）

    _, _, y_train_label, y_test_label = train_test_split(x_scaled, y, random_state=1, test_size=0.2)

 

    return x_train, x_test, y_train, y_test, y_train_label, y_test_label, m

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def model(x,w,b):
    z = np.dot(x,w)+b
    return softmax(z)

def cross_entropy_loss(y_true, y_pred, eps=1e-15):
    y_pred = np.clip(y_pred, eps, 1-eps)
    n = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred)) / n

def gradients(x, y_true, y_pred):
    n = x.shape[0]
    dw = np.dot(x.T, y_pred - y_true) / n
    db = np.sum(y_pred - y_true, axis=0) / n
    return dw, db

def gradient_descent(x, y_true, w, b, lr):
    y_pred = model(x, w, b)
    dw, db = gradients(x, y_true, y_pred)
    w = w - lr * dw
    b = b - lr * db
    loss = cross_entropy_loss(y_true, y_pred)
    return w, b, loss

def predict(x, w, b):
    prob = model(x, w, b)
    pred = np.argmax(prob, axis=1)
    return pred, prob

def main():
    x_train, x_test, y_train, y_test, y_train_label, y_test_label, scaler = load_and_preprocess_data()

    n_samples, n_features = x_train.shape
    n_classes = y_train.shape[1]

    np.random.seed(1)
    w = np.random.randn(n_features, n_classes) * 0.01
    b = np.zeros(n_classes)

    lr = 0.1
    iters = 1000
    for i in range(iters):
        w, b, loss = gradient_descent(x_train, y_train, w, b, lr)
        if i % 100 == 0:
            print(f"Iter {i:3d} | Loss: {loss:.4f}")

    train_pred, _ = predict(x_train, w, b)
    test_pred, _ = predict(x_test, w, b)

    train_acc = accuracy_score(y_train_label.flatten(), train_pred)
    test_acc = accuracy_score(y_test_label.flatten(), test_pred)

    print(f"训练集准确率: {train_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    print("权重 w:\n", w)
    print("偏置 b:", b)

if __name__ == "__main__":
    main()
