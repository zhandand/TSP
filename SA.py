import random
import math
import time
import timeit

import numpy as np
import matplotlib.pyplot as plt
from TSPInstance import TSPInstance



class SA(object):
    def __init__(self, tsp):
        # 传入数据集
        self.__tspInstance = tsp
        # 设置初始温度
        self.T0 = 4000
        # 设置结束温度
        self.Tend = 1e-4
        # 设置降温系数
        self.rate = 0.9995
        # 城市数量
        self.num_city = self.__tspInstance.city_num
        self.scores = []
        self.location = np.transpose(np.vstack((self.__tspInstance.x, self.__tspInstance.y)))
        # fruits中存每一个个体是下标的list
        self.fires = []
        # 计算不同城市之间的距离
        self.dis_mat = self.compute_dis_mat(self.num_city, self.location)
        self.fire = self.greedy_init(self.dis_mat,100,self.num_city)
        # 显示初始化后的路径
        init_pathlen = 1. / self.compute_pathlen(self.fire)
        init_best = self.location[self.fire]
        # 存储存储每个温度下的最终路径，画出收敛图
        self.iter_x = [0]
        self.iter_y = [1. / init_pathlen]
        # 定义最优距离和最优路径
        self.best_length = None
        self.best_path = None
        # 指定的epoch
        self.iter = []
        # 指定epoch对应的tour，作画图用
        self.paths = []





    # 贪婪的初始化一条初始路径
    def greedy_init(self, dis_mat, num_total, num_city):
        start_index = 0
        result = []
        for i in range(num_total):
            rest = [x for x in range(0, num_city)]
            # 所有起始点都已经生成了
            if start_index >= num_city:
                start_index = np.random.randint(0, num_city)
                result.append(result[start_index].copy())
                continue
            current = start_index
            rest.remove(current)
            # 找到一条最近邻路径
            result_one = [current]
            while len(rest) != 0:
                tmp_min = math.inf
                tmp_choose = -1
                # 找到距离当前城市最近的城市
                for x in rest:
                    if dis_mat[current][x] < tmp_min:
                        tmp_min = dis_mat[current][x]
                        tmp_choose = x

                current = tmp_choose
                result_one.append(tmp_choose)
                # 将加入result的点从reset中移除
                rest.remove(tmp_choose)
            result.append(result_one)
            start_index += 1
        pathlens = self.compute_paths(result)
        # 找到总距离最短路径对应的的索引
        sortindex = np.argsort(pathlens)
        index = sortindex[0]
        # 返回该索引对应的路径作为初始路径
        return result[index]

    # 初始化一条随机路径
    def random_init(self, num_city):
        tmp = [x for x in range(num_city)]
        return tmp

    # 计算不同城市之间的距离
    def compute_dis_mat(self, num_city, location):
        # 初始化距离矩阵
        dis_mat = np.zeros((num_city, num_city))
        # 计算所有城市两两之间的距离
        for i in range(num_city):
            for j in range(num_city):
                # 设置城市到自己本身的距离为inf
                if i == j:
                    dis_mat[i][j] = np.inf
                    continue
                a = location[i]
                b = location[j]
                # 计算第i个城市到第j个城市之间的距离
                tmp = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)]))
                dis_mat[i][j] = tmp
        return dis_mat

    # 计算路径长度
    def compute_pathlen(self, tour):
        dis = 0.0
        for i in range(self.num_city - 1):
            dis += self.dis_mat[tour[i]][tour[i+1]]
        dis += self.dis_mat[tour[self.num_city-1]][tour[0]]
        return dis

    # 计算一个温度下产生的一个群体的长度
    def compute_paths(self, paths):
        result = []
        for one in paths:
            length = self.compute_pathlen(one)
            result.append(length)
        return result

    # 产生一个新的解：随机交换两个元素的位置
    def get_new_fire(self, fire):
        fire = fire.copy()
        t = [x for x in range(len(fire))]
        # 随机选取两个城市
        a, b = np.random.choice(t, 2)
        # 将这两个城市之间的路径倒置
        fire[a:b] = fire[a:b][::-1]
        return fire

    # 退火策略，根据温度变化有一定概率接受差的解
    def eval_fire(self, raw, get, temp):
        len1 = self.compute_pathlen(raw)
        len2 = self.compute_pathlen(get)
        dc = len2 - len1

        p = np.exp(-dc / temp)
        # 如果新的路径路程更短，则更新
        if len2 < len1:
            return get, len2
        # 如果新的路径路程没有比原来更短，则以一定的概率更新
        elif np.random.rand() <= p:
            return get, len2
        else:
            return raw, len1

    # 模拟退火总流程
    def sa(self):
        count = 0
        # 记录最优解路径
        best_path = self.fire
        # 记录最优解总距离
        best_length = self.compute_pathlen(self.fire)

        # 持续迭代直至当前温度小于等于结束温度
        while self.T0 > self.Tend:

            if count % 5000 == 0:
                self.iter.append(count)
                self.paths.append(best_path)
            count += 1
            # 产生在这个温度下的随机解
            tmp_new = self.get_new_fire(self.fire.copy())
            # 根据温度判断是否选择这个解
            self.fire, file_len = self.eval_fire(best_path, tmp_new, self.T0)
            # 更新最优解
            # if file_len < best_length:
            #     best_length = file_len
            #     best_path = self.fire
            best_length = file_len
            best_path = self.fire
            # 降低温度
            self.T0 *= self.rate
            # 记录路径收敛曲线
            self.iter_x.append(count)
            self.iter_y.append(best_length)
            # print(count, best_length)
        return best_length, best_path

    def run(self):
        start = time.time()
        self.best_length, self.best_path = self.sa()
        end = time.time()
        print("time:%fs"%(end-start))
        return self.best_length, self.best_path

