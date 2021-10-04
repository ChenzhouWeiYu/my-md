# 向前向后欧拉法

<font size=4>
    &emsp;&emsp; 对于初值问题
    \begin{equation}
        \frac{\text{d}y}{\text{d}t} = 10y\left(1-y\right), \quad \text{for}\quad 0 \leq t \leq 1
    \end{equation}
    其中
    \begin{equation}
        y\left(0\right)=0.01
    \end{equation}
    <br>
    
<font size=4>
    使用向前向后欧拉法求数值解，并在同一个坐标系下画出精确解和时间节点为4，16，64，256时的数值解。


```python
import numpy as np
import matplotlib.pyplot as plt
```

<font size=4>
    其真解为
    \begin{equation}
        y = \frac{e^{10t}}{e^{10t}+99}
    \end{equation}
    设时间节点$N$，则时间步长$h=\frac{1}{N-1}$，$t_{i}=ih=\frac{i}{N-1}$，令$y_i=y\left(t_i\right)$


```python
t = np.linspace(0,1,1000)
y = np.exp(10*t)/(np.exp(10*t)+99)
plt.figure()
plt.plot(t,y)
plt.show()
```


![png](output_4_0.png)


<font size=4> 
    向前欧拉法：直接迭代
    \begin{equation}
        y_{i+1}=y_{i}+h\cdot 10y_{i}\left(1-y_{i}\right), \quad i=0,1,\cdots,N-1
    \end{equation}
    其中，$y_0=0.01$
    <br>
    向后欧拉法：隐格式，需要解方程
    \begin{equation}
        y_{i+1}=y_{i}+h\cdot 10y_{i+1}\left(1-y_{i+1}\right), \quad i=0,1,\cdots,N-1
    \end{equation}
    其中，$y_0=0.01$， 移项化简得到如下形式：
    \begin{equation}
        \left(1-10h\right)y_{i+1} + 10hy_{i+1}^2=y_{i}
    \end{equation}
    公式法解上述方程，并舍去负根后得到：
    \begin{equation}
        y_{i+1} = \frac{10h-1}{20h} + \frac{\sqrt{(10h-1)^2+40hy_{i}}}{20h}
    \end{equation}
    


```python
def Forward_Euler(N):
    t = np.linspace(0,1,N)
    y = [0.01]
    h = 1/(N-1)
    for i in range(N-1):
        y0 = y[-1]
        y.append( y0 + h* 10*y0*(1-y0) )
    return t,np.array(y)

def Backward_Euler(N):
    t = np.linspace(0,1,N)
    y = [0.01]
    h = 1/(N-1)
    for i in range(N-1):
        y0 = y[-1]
        y.append( (10*h-1)/(20*h) + ((10*h-1)**2+40*h*y0)**0.5/(20*h) )
    return t,np.array(y)
```


```python
# 定性 验证程序正确性
N = 2**20
ft,fy = Forward_Euler(N)
bt,by = Backward_Euler(N)
plt.figure()
plt.plot(t,y,label='True_Solution')
plt.plot(ft,fy,'-.',label='Forward_Euler')
plt.plot(bt,by,'-.',label='Backward_Euler')
plt.legend(loc=0)
plt.show()
```


![png](output_7_0.png)


# 画图


```python
plt.figure(figsize=(16,12))
for i in range(2):
    for j in range(2):
        plt.subplot(2,2,2*i+j+1)
        N = 2**(4*i+2*j+2)
        ft,fy = Forward_Euler(N)
        bt,by = Backward_Euler(N)
        tt = np.linspace(0,1,N)
        ty = np.exp(10*tt)/(np.exp(10*tt)+99)
        plt.plot(t,y,label='True Solution')
        plt.plot(ft,fy,'-.',label='Forward Euler')
        plt.plot(bt,by,'-.',label='Backward Euler')
        plt.legend(loc=0,fontsize=15)
        plt.title('N = '+str(N),fontsize=15)
plt.show()
```


![png](output_9_0.png)



```python

```
