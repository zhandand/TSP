"""
    TSPInstance.py contains a class TSPInstance
"""
import math

import matplotlib.pyplot as plt
import numpy as np


class TSPInstance(object):
    def __init__(self, dirpath, datasetName) -> None:
        """
        约定数据集文件名为 datasetName + '.txt'
            对应最佳结果文件名为 datasetName + '.opt.txt'

        Args:
            dirpath (_type_): 数据父目录
            datasetName (_type_): 数据集名字
        """
        self.datasetName = datasetName
        self.data_name = dirpath + datasetName + '.txt'
        self.ideal_name = dirpath + datasetName + '.opt.txt'
        self.load_dataset(self.data_name)
        self.load_optdata(self.ideal_name)
        assert len(self.optimaltour) == len(self.x)
        self.city_num = len(self.optimaltour)
        self.distance_graph = self.get_distance_graph()

    def load_dataset(self, filepath):
        """
        read the position of points 
        assume thats avaliable information starts at line 6 
        Args:
            filepath (_type_): _description_
        """
        with open(filepath, "r") as f:
            data = f.readlines()
        self.x = []
        self.y = []
        for line in data[6:-1]:
            points = line.split()
            self.x.append(float(points[1]))
            self.y.append(float(points[2]))
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        assert self.x.shape == self.y.shape

    def load_optdata(self, filepath):
        """
        read the ideal ranking of points
        assume thats avaliable information starts at line 4 
        Args:
            filepath (_type_): _description_
        """
        with open(filepath, "r") as f:
            data = f.readlines()
        self.optimaltour = []
        for line in data[4:-1]:
            self.optimaltour.append(int(line)-1)
        self.optimaltour = np.array(self.optimaltour)

    @property
    def citynum(self):
        """
            returns the number of city
        """
        return self.city_num

    def __getitem__(self, i):
        """
            return the coordinate of a point given index
        Args:
            i (_type_): index

        Returns:
            _type_: coordiante
        """
        return (self.x[i], self.y[i])

    def get_distance(self, i, j):
        """
            calculates the Euclidean Distance of two given points
        Args:
            i (_type_): index of points 1
            j (_type_): index of points 2
        Returns:
            the distance
        """
        deltaX = self.x[i] - self.x[j]
        deltaY = self.y[i] - self.y[j]
        return math.sqrt(pow(deltaX, 2) + pow(deltaY, 2))
    
    def get_tour_distance(self, tour):
        """
            returns the whole distance of the specific tour
        Args:
            tour (list): _description_
        """
        dis = 0.0
        for i in range(self.city_num-1):
            dis += self.distance_graph[tour[i]][tour[i+1]]
        dis += self.distance_graph[tour[self.city_num-1]][tour[0]]
        return dis

    def get_distance_graph(self):
        """
            returns the distance graph of the cities
        """
        distance_graph = np.zeros((self.citynum, self.citynum))
        for i in range(self.citynum):
            for j in range(i, self.citynum):
                distance_graph[i][j] = self.get_distance(i,j)
                distance_graph[j][i] = self.get_distance(i,j)
        return distance_graph


    def plot_tour(self, tour):
        """
            draw the path of the tour
        Args:
            tour (list): _description_
        """
        for i in range(self.city_num):
            x0, y0 = self.__getitem__(i)
            plt.scatter(int(x0), int(y0), s=10, c='c')
        for i in range(len(tour)-1):
            x1, y1 = self.__getitem__(int(tour[i]))
            x, y = self.__getitem__(int(tour[i+1]))
            plt.plot([x1, x], [y1, y], c='b')
        x2, y2 = self.__getitem__(int(tour[0]))
        x3, y3 = self.__getitem__(int(tour[len(tour)-1]))

        plt.plot([x2, x3], [y2, y3], c='b')
        plt.xlabel('x label')
        plt.ylabel('y label')
        plt.title("City access sequence diagram")
        plt.plot()
        plt.savefig('./result/' + self.datasetName + '.png', bbox_inches='tight')


if __name__ == "__main__":
    instance = TSPInstance('/mnt/4ta/gzzhan/projects/TSP/dataset/', 'a280')
    # print(instance.citynum)
    # print(instance.evaluate(instance.optimaltour))
    # print(instance.optimaltour)
    instance.plot_tour(instance.optimaltour)
    # print(instance.get_distance_graph())
    assert instance.get_tour_distance(instance.optimaltour) == instance.evaluate(instance.optimaltour)

