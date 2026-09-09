def f(x, y):
    return (x + y - 3) ** 2 + (x + 2 * y - 5) ** 2 + 2
def d_f_x(x,y):
    return 2 * (x + y - 3) + 2 * (x + 2 * y - 5)
def d_f_y(x,y):
    return 2 * (x + y - 3) + 4 * (x + 2 * y - 5)
learning_rate = 0.1
max_loop = 500000
tolerance = 0.00000001
x_init = 10.0 
y_init = 10.0

x = x_init
y = y_init
f_xy_pre= f(x,y)
for i in range(max_loop):
    df_dx = d_f_x(x,y)
    df_dy = d_f_y(x,y)
    x = x - learning_rate*df_dx
    y = y - learning_rate*df_dy
    print(x,y)
    f_xy_cur = f(x,y)
    if abs(f_xy_cur-f_xy_pre)<tolerance:
        break 
    f_xy_pre = f_xy_cur
print('initial x =',x_init)
print('initial y =',y_init)
print('arg min f(x,y) of x =',x)
print('arg min f(x,y) of y =',y)
print('f(x,y) =',f(x,y))
