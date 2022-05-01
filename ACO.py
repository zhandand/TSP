"""
解决TSP问题的蚁群算法（Ant Colony Optimization）
"""
import math
import random
import sys
from multiprocessing import Pool
from importlib_metadata import re

import numpy as np
from tqdm import tqdm

from TSPInstance import TSPInstance
from utils import draw_quality

# (city_num, ant_num) = (50, 50)
# parameter
# (ALPHA, BETA, RHO, Q) = (1.0, 2.0, 0.5, 100.0)
# pheromone_graph = [[1.0 for _ in range(city_num)] for _ in range(city_num)]
# distance_graph = [[1.0 for _ in range(city_num)] for _ in range(city_num)]


class Ant(object):
    __ALPHA = 0.0
    __BETA = 0.0
    __Q = 1.0
    __city_num = 0

    def __init__(self, city_num, ALPHA, BETA, Q) -> None:
        super().__init__()
        Ant.setHyperparameter(city_num, ALPHA, BETA, Q)
        self.__initialize()

    def __initialize(self):
        self.__current_city = random.randint(
            0, Ant.get_city_num()-1)       # 随机初始化城市
        self.__path = [self.__current_city]                         # ant的轨迹
        self.__total_distance = 0.0
        self.__step = 1                                           # 移动次数
        self.__avaliable_city = [
            True for i in range(Ant.get_city_num())]   # 未访问城市
        self.__avaliable_city[self.__current_city] = False

    @classmethod
    def setHyperparameter(cls, city_num, alpha, beta, Q):
        cls.__ALPHA = alpha
        cls.__BETA = beta
        cls.__Q = Q
        cls.__city_num = city_num

    @classmethod
    def getAlpha(cls):
        return cls.__ALPHA

    @classmethod
    def getBeta(cls):
        return cls.__BETA

    @classmethod
    def getQ(cls):
        """
        Returns:
            _type_: The pheromone released by the ant cycle once 
        """
        return cls.__Q

    @classmethod
    def get_city_num(cls):
        return cls.__city_num

    def __rand_choose(self, p):
        """
            Roulette for choosing city
        Args:
            p (_type_): probability of choosing one city

        Returns:
            _type_: the index of the chosen city
        """
        x = np.random.rand()
        for i, prob in enumerate(p):
            x -= prob
            if x <= 0:
                break
        return i

    def __choose_next_city(self, distance_graph, pheromone_graph):
        """
            choose next city according to pheromone
        Args:
            distance_graph (np.array[[]]): [city_num, city_num]
            pheromone_graph (np.array[[]]): [city_num, city_num]
        """
        select_cities_prob = [0.0 for _ in range(Ant.get_city_num())]

        for i in range(Ant.get_city_num()):
            if self.__avaliable_city[i] is True:
                try:
                    # the probability is proportional to pheromone, inversely proportional to distance
                    # denominator can't be 0
                    select_cities_prob[i] = pow(pheromone_graph[self.__current_city][i], Ant.getAlpha()) * pow(
                        1.0/(distance_graph[self.__current_city][i]+0.00001), Ant.getBeta())
                except ZeroDivisionError as e:
                    print('Ant ID: {ID}, current city: {current}, target city: {target}'.format(
                        ID=self.ID, current=self.__current_city, target=i))
                    sys.exit(1)
        select_cities_prob = np.array(select_cities_prob)
        select_cities_prob = select_cities_prob / \
            np.sum(select_cities_prob)  # normalization
        next_city = self.__rand_choose(select_cities_prob)
        return next_city

    def __move(self, next_city, distance):
        """
            ant moves to the next city
        Args:
            next_city (_type_): _description_
            distance (_type_): the distance from current city to the next city
        """
        self.__path.append(next_city)
        self.__avaliable_city[next_city] = False
        self.__total_distance += distance
        self.__current_city = next_city
        self.__step += 1

    def __release_pheromone(self):
        """
            release pheromone on the road
        """
        pheromone = np.zeros([len(self.__path), len(self.__path)])
        for i in range(len(self.__path)-1):
            cur_city = self.__path[i]
            next_city = self.__path[i+1]
            pheromone[cur_city][next_city] = Ant.getQ() / \
                self.__total_distance
            pheromone[next_city][cur_city] = pheromone[cur_city][next_city]
        pheromone[0][self.__path[-1]] = Ant.getQ() / \
            self.__total_distance
        pheromone[self.__path[-1]][0] = Ant.getQ() / \
            self.__total_distance
        return pheromone

    def run(self, distance_graph, pheromone_graph):
        """
            find one circle according to pheromone
        Args:
            distance_graph (np.array[[]]): [city_num, city_num]
            pheromone_graph (np.array[[]]): [city_num, city_num]
        Returns:
            path: the path which the ant traverses
            total_distance: the path which the ant traverses
            pheromone: the pheromone which the ant releases
        """
        self.__initialize()
        # search all the cities
        while self.__step < Ant.get_city_num():
            next_city = self.__choose_next_city(
                distance_graph, pheromone_graph)
            self.__move(
                next_city, distance_graph[self.__current_city][next_city])
        self.__total_distance += distance_graph[self.__path[-1]
                                                ][self.__path[0]]
        return self.__path, self.__total_distance, self.__release_pheromone()


class ACO(object):
    def __init__(self, dirpath='./dataset/', datasetName='a280') -> None:
        self.__tspInstance = TSPInstance(dirpath, datasetName)
        self.__dataset = datasetName

        self.__iter = 500
        self.__ant_num = 50   # the number of ants
        self.__alpha = 2      # Pheromone importance factor
        self.__beta = 2       # Important factor of heuristic function
        self.__rho = 0.1      # Pheromone volatilization factor
        self.__Q = 100.0
        self.__generate_ants_population()

        self.__city_num = len(self.__tspInstance.get_distance_graph())
        self.__pheromone_graph = np.ones(
            [self.__city_num, self.__city_num])    # 全0初始化会在轮盘赌时出错

    def __generate_ants_population(self):
        """
            generate ants population
        """
        self.__ants = []
        for i in range(self.__ant_num):
            self.__ants.append(
                Ant(self.__tspInstance.city_num, self.__alpha, self.__beta, self.__Q))

    def __single_ant_run(self, ant_id):
        """
            for multiple process
        Args:
            ant_id (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.__ants[ant_id].run

    def run(self):
        best_path = None
        shortest_distance = math.inf
        quality = []
        for epoch in tqdm(range(1, self.__iter+1)):
            release_pheromone = []
            all_path = []
            all_distance = []
            # tandem
            # for ant in self.__ants:
            #     path, total_distance, pheromone = ant.run(
            #         self.__tspInstance.get_distance_graph(), self.__pheromone_graph)
            #     release_pheromone.append(pheromone)
            #     if total_distance < shortest_distance:
            #         shortest_distance = total_distance
            #         best_path = path

            # Parallel
            p = Pool(5)
            results = []
            for ant in self.__ants:
                result = p.apply_async(ant.run, args=(
                    self.__tspInstance.get_distance_graph(), self.__pheromone_graph))
                results.append(result)
            for res in results:
                path, total_distance, pheromone = res.get()
                release_pheromone.append(pheromone)
                if total_distance < shortest_distance:
                    shortest_distance = total_distance
                    best_path = path
            
            # update pheromone graph
            self.__pheromone_graph = (1-self.__rho) * self.__pheromone_graph + np.vstack(
                release_pheromone).reshape(self.__ant_num, self.__city_num, self.__city_num).sum(axis=0)
            quality.append(shortest_distance /
                           self.__tspInstance.optTourDistance)
            if epoch % 50 == 0:
                self.__tspInstance.plot_tour(
                    tour=path, name='Epoch ' + str(epoch))
                print("-"*20 + " epoch " + str(epoch) + "-"*20)
                print('quality: {}'.format(shortest_distance /
                                           self.__tspInstance.optTourDistance))

        draw_quality(quality, './result/' +
                     self.__dataset + '/', self.__dataset)


if __name__ == '__main__':
    aco = ACO()
    aco.run()
