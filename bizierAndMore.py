# from scipy.special import comb
import numpy as np
import matplotlib.pyplot as plt
from typing import List
# from tools import 

def comb(n, m):
    """排列组合"""
    list = [1, 1]
    for i in range(2, n+1):
        list.append(list[i-1]*i)
    return list[n] // (list[m] * list[n-m])

def QT(controlPoints, t):
    """
    返回贝塞尔曲线上 t 对应的点的坐标
    """
    n = len(controlPoints)-1
    Bt = np.zeros(2, np.float64)
    for i in range(len(controlPoints)): # 四个点使用 3 的组合, 即 C(3,0), C(3,1), C(3,2), C(3,3)
        Bt = Bt + comb(n,i) * np.power(1-t,n-i) * np.power(t,i) * np.array(controlPoints[i])
    return Bt

def QT_(controlPoints, t):
    """
    Q(t) 函数的一阶导数
    """
    n = len(controlPoints)-1
    Bt = np.zeros(2, np.float64)
    for i in range(len(controlPoints)): 
        # Bt = Bt + comb(n,i) * np.power(1-t,n-i) * np.power(t,i) * np.array(controlPoints[i])
        c1 = - (n-i) * np.power(1-t,n-i-1) * np.power(t,i)
        c2 = i * np.power(1-t,n-i) * np.power(t,i-1)
        Bt = Bt + comb(n,i) * (c1 + c2) * np.array(controlPoints[i])
    return Bt

"""高斯勒让德积分公式的求积系数和求积节点"""
W = [0.5688888888888889, 0.4786286704993665, 0.4786286704993665, 0.2369268850561891, 0.2369268850561891]
X = [0.0000000000000000, -0.5384693101056831, 0.5384693101056831, -0.9061798459386640, 0.9061798459386640]

def LT(controlPoints, t) -> float:
    """
    返回 点 QT(t) 到贝塞尔曲线起点的距离
    """
    ret = float(0)
    n = W.__len__()
    t_half = t/2
    for i in range(n):
        qt_ = QT_(controlPoints, t_half*(X[i]+1))
        ret += W[i] * np.sqrt(qt_[0]*qt_[0] + qt_[1]*qt_[1])
    ret *= t_half
    return ret

def LT_(controlPoints, t):
    """
    LT() 的一阶导数
    """
    qt_ = QT_(controlPoints, t)
    return np.sqrt(qt_[0]*qt_[0] + qt_[1]*qt_[1])

def newton(controlPoints, start_x, target_y, tol) -> float:
    """
    牛顿迭代法求解满足 target_y = LT(t) 的 t
    start_x: t 的初始值
    target_y: 此处指目标距离
    tol: 与 target_y 的最大允许误差
    """
    fx = LT(controlPoints, start_x)
    # print("start: ", start_x)
    while abs( fx - target_y ) > tol:
        dfx = LT_(controlPoints, start_x)
        start_x = start_x - (fx - target_y) / dfx
        fx = LT(controlPoints, start_x)
        # print(start_x)
    return start_x

def getInterpolationPoints(controlPoints, tList):
    """
    给出 t 的列表，返回贝塞尔曲线上对应点的坐标
    """
    n = len(controlPoints)-1
    interPoints = []
    for t in tList:
        Bt = np.zeros(2, np.float64)
        for i in range(len(controlPoints)):
            Bt = Bt + comb(n,i) * np.power(1-t,n-i) * np.power(t,i) * np.array(controlPoints[i])
        # Bt = [t*50, LT(controlPoints, t)]
        # Bt = [t*50, LT_(controlPoints, t)]
        interPoints.append(list(Bt))

    return interPoints


def getControlPointList(pointsArray):
    """
    求解贝塞尔曲线的控制点
    这个算法可以自行设计, 控制点会决定曲线的形状
    """
    points = np.array(pointsArray)
    index = points.shape[0] - 2

    res = []
    for i in range(index):

        tmp = points[i:i+3]
        p1 = tmp[0]
        p2 = tmp[1]
        p3 = tmp[2]
        
        l12 = np.sqrt(np.sum((p2 - p1)**2))
        l23 = np.sqrt(np.sum((p3 - p2)**2))
        p12_norm = (p2 - p1) / l12
        p23_norm = (p3 - p2) / l23
        e = (p12_norm + p23_norm) / np.sqrt(np.sum((p12_norm + p23_norm)**2))
        c1 = p2 - e * l12 * 0.2
        c2 = p2 + e * l23 * 0.2
        res.append(c1)
        res.append(c2)

    pFirst = points[0] + 0.1*(res[0] - points[0])
    pEnd = points[-1] + 0.1*(res[-1] - points[-1])
    res.insert(0,pFirst)
    res.append(pEnd)

    return np.array(res)

# def getControlPointList(pointsArray, k1=-1, k2=1):
#     """
#     求解贝塞尔曲线的控制点
#     这个算法可以自行设计, 控制点会决定曲线的形状
#     """
#     points = np.array(pointsArray)
#     index = points.shape[0] - 2

#     res = []
#     for i in range(index):
#         tmp = points[i:i+3]
#         p1 = tmp[0]
#         p2 = tmp[1]
#         p3 = tmp[2]

#         if k1 == -1:
#             l1 = np.sqrt(np.sum((p1-p2)**2))
#             l2 = np.sqrt(np.sum((p2-p3)**2))
#             k1 = l1/(l1+l2)
#             k2 = l2/(l1+l2)

#         p01 = k1*p1 + (1-k1)*p2
#         p02 = (1-k2)*p2 + k2*p3
#         p00 = k2*p01 + (1-k2)*p02
        
#         sub = p2 - p00
#         p12 = p01 + sub
#         p21 = p02 + sub

#         res.append(p12)
#         res.append(p21)
#     pFirst = points[0] + 0.1*(res[0] - points[0])
#     pEnd = points[-1] + 0.1*(res[-1] - points[-1])
#     res.insert(0,pFirst)
#     res.append(pEnd)

#     return np.array(res)

def getPositionsFromDist(dists: List[float], points: List[List[float]], controlP: np.ndarray) -> List[List[float]]:
    """
    我们按照一个划定的路线行进，本函数输入在路线上行进的距离，返回与距离一一对应的坐标
    
    dists 表示距离列表
    points 和 controlP 用于限制路线
    
    返回坐标列表
    """
    l = len(points) - 1
    figure = plt.figure()
    j = 0
    extraLen = 0
    retPoints = []
    for i in range(l):
        controlPoints = np.array([points[i], controlP[2*i], controlP[2*i+1], points[i+1]])
        total_len = LT(controlPoints, 1)
        
        tList = []
        for dist in dists[j:]:
            if dist - extraLen > total_len:
                j += len(tList)
                extraLen += total_len
         
                # plt.scatter(range(len(tList)),tList)
                retPoints += getInterpolationPoints(controlPoints, tList)
                break
            else:
                target_y = dist - extraLen
                start_t = target_y / total_len
                t = newton(controlPoints, start_t, target_y, 0.0000000001)
                tList.append(t)
        
    x = np.array(retPoints)[:,0]
    y = np.array(retPoints)[:,1]
    plt.scatter(x,y)
    plt.plot(x, y)
    plt.scatter(np.array(points)[:,0],np.array(points)[:,1])
    plt.show()
    return retPoints
        
def example1():
    """
        贝塞尔曲线是关于 t 的函数, 其中 t 的范围为 [0,1]
        本示例 t 均匀增长, 画出 t 均匀增长时曲线上点的位置
    """
    points = [[1,1],[3,6],[6,3],[8,0],[11,6],[12,12]]
    controlP = getControlPointList(points)
    l = len(points) - 1
    figure = plt.figure()
    t = np.linspace(0,1,12)
    for i in range(l):
        p = np.array([points[i], controlP[2*i], controlP[2*i+1], points[i+1]])
        interPoints = getInterpolationPoints(p, t)
        x = np.array(interPoints)[:,0]
        y = np.array(interPoints)[:,1]
        plt.scatter(x,y)
        plt.plot(x, y)
            
    plt.scatter(np.array(points)[:,0],np.array(points)[:,1],color='gray')
    plt.show()
    
def example2():
    """
        贝塞尔曲线是关于 t 的函数, 其中 t 的范围为 [0,1]
        
        本示例首先求出贝塞尔曲线的长度 s 随 t 变化的公式 LT(controlPoints, t), 当 t 为 0 时该函数值为 0, 当 t 为 1 时该函数值为贝塞尔曲线全长
        
        我们用牛顿迭代法求解与指定 s 对应的 t, 再使用求解出的 t 算出坐标, 从而求解与指定长度 s 对应的曲线上的坐标
    """
    points = [[1,1],[3,6],[6,3],[8,0],[11,6],[12,12]]
    controlP = getControlPointList(points)
    l = len(points) - 1
    
    figure = plt.figure()
    for i in range(l):
        p = np.array([points[i], controlP[2*i], controlP[2*i+1], points[i+1]])
        
        total_len = LT(p, 1)
        target_ys = np.linspace(0, total_len, max(2, math.ceil(total_len/0.5)))
        print(target_ys)
        
        ts = []
        for target_y in target_ys:
            start_x = target_y / total_len
            x = newton(p, start_x, target_y, 0.0000001)
            ts.append(x)

        interPoints = getInterpolationPoints(p, ts)
        x = np.array(interPoints)[:,0]
        y = np.array(interPoints)[:,1]
        plt.scatter(x,y)
        plt.plot(x,y)
            
    plt.scatter(np.array(points)[:,0],np.array(points)[:,1],color='gray')
    plt.show()

def example3():
    """
    是 example2 的扩展版, 我们连接了多段 贝塞尔曲线, 这意味着我们给出的长度是由多段贝塞尔曲线相加得到的（而不是在单段贝塞尔曲线上）
    """
    dists = np.linspace(0,30,68)
    print(dists)
    points = [[1,1],[3,6],[6,3],[8,0],[11,6],[12,12]]
    controlP = getControlPointList(points)
    retPoints = getPositionsFromDist(dists, points, controlP)
    print(retPoints.__len__())

import math
if __name__ == '__main__':
    example1()      # 对应 图 2
    example2()      # 对应 图 3
    example3()      # 对应 图 4
    
    
