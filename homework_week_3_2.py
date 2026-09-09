import numpy as np

x1=np.random.randn(100)
x2=np.random.randn(100)
x3=np.random.randn(100)

y=3*x1+4*x2+5*x3+6+np.random.randn(100)

a1,a2,a3,b=np.random.randn(4)
lr=0.0001

for _ in range(50000):
    y_pred=a1*x1+a2*x2+a3*x3+b
    loss=0.5*np.sum((y-y_pred)**2)/100
    da1=-np.sum(x1*(y-y_pred))/100
    da2=-np.sum(x2*(y-y_pred))/100
    da3=-np.sum(x3*(y-y_pred))/100
    db=-np.sum(y-y_pred)/100
    a1-=lr*da1
    a2-=lr*da2
    a3-=lr*da3
    b-=lr*db

print(f"loss={loss:.4f}")
print(f"a1={a1:.2f}, a2={a2:.2f}, a3={a3:.2f}, b={b:.2f}")