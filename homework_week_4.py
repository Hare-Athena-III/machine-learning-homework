from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
def main():
    #分离特征列与花型
    def iris_type(s):
        it={'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}
        return it[s]
    data=np.loadtxt('iris.data',delimiter=',',converters={4:iris_type})    #文件与py文件处于同一目录下
    x,y=np.split(data,(4,),axis=1) #x为特征矩阵 y为花型向量
    #划分花型为one hot向量
    encoder=OneHotEncoder(sparse_output=False)#或者 encoder=OneHotEncoder()
    encoder.fit(y)
    targets=encoder.transform(y)
    #特征归一化
    m=preprocessing.MinMaxScaler()
    x_scaled=m.fit_transform(x)
    #划分训练集和测试集 测试集占比20%
    x_train,x_test,y_train,y_test=train_test_split( x_scaled,targets,random_state=1,test_size=0.2)

    print("训练集特征形状：", x_train.shape)
    print("测试集特征形状：", x_test.shape)
    print("训练集标签形状：", y_train.shape)
    print("测试集标签形状：", y_test.shape)

    train_idx = (y_train[:,0] == 1) | (y_train[:,1] == 1)
    test_idx = (y_test[:,0] == 1) | (y_test[:,1] == 1)

    x_train_bin = x_train[train_idx]
    y_train_bin = y_train[train_idx][:, :2]  
    x_test_bin = x_test[test_idx]
    y_test_bin = y_test[test_idx][:, :2]

    def sigmoid(z):
        return 1/(1+np.exp(-z))
    
    def model(x,w,b):
        return sigmoid(np.dot(x,w)+b)

    def loss(y_pred, y_true):
        m = len(y_true)
        return -np.sum(y_true * np.log(y_pred + 1e-8)) / m

    def grad(x, y_true, y_pred):
        m = x.shape[0]
        dw = np.dot(x.T, y_pred - y_true) / m
        db = np.sum(y_pred - y_true) / m
        return dw, db
    
    def gd(x,y,lr=0.005,epoch=1000):
        n_features = x.shape[1]
        n_classes = y.shape[1]
        w=np.zeros((n_features,n_classes))
        b=np.zeros(n_classes)
        for _ in range(epoch):
            y_pred = model(x,w,b)
            dw,db = grad(x,y,y_pred)
            w -= lr * dw
            b -= lr * db
        return w,b

    def predict(x,w,b):
        yp=model(x,w,b)
        return np.argmax(yp,axis=1)
    
    w, b = gd(x_train_bin, y_train_bin)

    yt = predict(x_train_bin, w, b)
    ytt = np.argmax(y_train_bin, axis=1)
    train_acc = np.mean(yt == ytt)

    yp = predict(x_test_bin, w, b)
    ytt2 = np.argmax(y_test_bin, axis=1)
    test_acc = np.mean(yp == ytt2)
    print("train_acc=",train_acc)
    print("test_acc=",test_acc)
main()