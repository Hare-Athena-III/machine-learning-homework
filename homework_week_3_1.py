import numpy as np

x=np.linspace(-100,100,100)
y=3.0*x+4.0+np.random.randn(100)*0.05

a,b=np.random.randn(),np.random.randn()
lr=0.0001

for _ in range(50000):
    y_pred=a*x+b
    loss=0.5*np.sum((y-y_pred)**2)/100
    da=-np.sum(x*(y-y_pred))/100
    db=-np.sum(y-y_pred)/100
    a-=lr*da
    b-=lr*db
print(f"loss={loss:.4f}")
print(f"a={a:.2f}, b={b:.2f}")