# Ant Colony Optimization to solve Traveling Salesman Problem
# Contains two classes:
#        Ant.py
#        ACO.py

import math
import random
import sys
# from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from TSPInstance import TSPInstance
from utils import draw_quality, timer


class Ant(object):
    # Instance of ant in colony
    # times of pheromone intensity when choosing city
    __ALPHA = 0.0
    # times of visibility when choosing city
    __BETA = 0.0
    # coefficient when updating pheromone intensity
    __Q = 1.0
    # #cities
    __city_num = 0

    def __init__(self, city_num, ALPHA, BETA, Q):
        super().__init__()
        Ant.setHyperparameter(city_num, ALPHA, BETA, Q)
        self.__initialize()

    def __initialize(self):
        # 随机初始化城市
        self.__current_city = random.randint(
            0, Ant.get_city_num()-1)
        # ant的轨迹
        self.__path = [self.__current_city]
        # 总距离
        self.__total_distance = 0.0
        # 移动次数
        self.__step = 1
        # 未访问城市
        self.__avaliable_city = [
            True for i in range(Ant.get_city_num())]
        self.__avaliable_city[self.__current_city] = False

    @classmethod
    def setHyperparameter(cls, city_num, alpha, beta, Q):
        #    set the hyperparameters for ants
        # Args:
        #    city_num (_type_): #cities
        #    alpha (_type_): Pheromone importance factor
        #    beta (_type_): Important factor of heuristic function
        #    Q (_type_): coefficient when updating pheromone intensity
        cls.__ALPHA = alpha
        cls.__BETA = beta
        cls.__Q = Q
        cls.__city_num = city_num

    @classmethod
    def getAlpha(cls):
        # Returns:
        #     _type_: Pheromone importance factor
        return cls.__ALPHA

    @classmethod
    def getBeta(cls):
        # Returns:
        #    _type_: Important factor of heuristic function
        return cls.__BETA

    @classmethod
    def getQ(cls):
        # Returns:
        #    _type_: The pheromone released by the ant cycle once 
        return cls.__Q

    @classmethod
    def get_city_num(cls):
        # Returns:
        #    _type_: #cities
        return cls.__city_num

    def __rand_choose(self, p):
        #    Roulette for choosing city
        # Args:
        #    p (_type_): probability of choosing one city
        #
        # Returns:
        #    _type_: the index of the chosen city
        x = np.random.rand()
        for i, prob in enumerate(p):
            x -= prob
            if x <= 0:
                break
        return i

    def __choose_next_city(self, distance_graph, pheromone_graph):
        #    choose next city according to pheromone
        # Args:
        #     distance_graph (np.array[[]]): [city_num, city_num]
        #     pheromone_graph (np.array[[]]): [city_num, city_num]
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
        # normalization
        select_cities_prob = select_cities_prob / \
            np.sum(select_cities_prob)
        next_city = self.__rand_choose(select_cities_prob)
        return next_city

    def __move(self, next_city, distance):
        #     ant moves to the next city
        # Args:
        #     next_city (_type_): _description_
        #     distance (_type_): the distance from current city to the next city
        self.__path.append(next_city)
        self.__avaliable_city[next_city] = False
        self.__total_distance += distance
        self.__current_city = next_city
        self.__step += 1

    def __release_pheromone(self):
        # release pheromone on the road
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
        #    find one circle according to pheromone
        # Args:
        #    distance_graph (np.array[[]]): [city_num, city_num]
        #    pheromone_graph (np.array[[]]): [city_num, city_num]
        # Returns:
        #    path: the path which the ant traverses
        #    total_distance: the path which the ant traverses
        #    pheromone: the pheromone which the ant releases
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
    def __init__(self, tspInstance):
        self.__tspInstance = tspInstance

        self.__iter = 150
        # the number of ants
        self.__ant_num = 50
        # Pheromone importance factor
        self.__alpha = 2
        # Important factor of heuristic function
        self.__beta = 2
        # Pheromone volatilization factor
        self.__rho = 0.1
        self.__Q = 100.0
        self.__generate_ants_population()

        self.__city_num = len(self.__tspInstance.get_distance_graph())
        # 全0初始化会在轮盘赌时出错
        self.__pheromone_graph = np.ones(
            [self.__city_num, self.__city_num])
        # 每一轮迭代的current distance / optimal distance
        self.quality = []
        # 所有epoch中最短距离
        self.global_opt_distance = math.inf
        # 所有epoch中最好的path
        self.global_opt_path = None
        # 指定的epoch
        self.epoch = []
        # 指定epoch对应的tour，作画图用
        self.paths = []

    def __generate_ants_population(self):
        # generate ants population
        self.__ants = []
        for i in range(self.__ant_num):
            self.__ants.append(
                Ant(self.__tspInstance.city_num, self.__alpha, self.__beta, self.__Q))

    @timer
    def run(self, result_path):
        for epoch in tqdm(range(1, self.__iter+1)):
            best_path = None
            shortest_distance = math.inf
            release_pheromone = []
            # tandem
            for ant in self.__ants:
                path, total_distance, pheromone = ant.run(
                    self.__tspInstance.get_distance_graph(), self.__pheromone_graph)
                release_pheromone.append(pheromone)
                if total_distance < shortest_distance:
                    shortest_distance = total_distance
                    best_path = path

            # update pheromone graph
            self.__pheromone_graph = (1-self.__rho) * self.__pheromone_graph + np.vstack(
                release_pheromone).reshape(self.__ant_num, self.__city_num, self.__city_num).sum(axis=0)
            quality = shortest_distance / self.__tspInstance.optTourDistance
            self.quality.append(quality)
            if self.global_opt_distance / self.__tspInstance.optTourDistance > quality:
                self.global_opt_distance = shortest_distance
                self.global_opt_path = path
            if epoch % 20 == 0 or (epoch < 20 and epoch % 4 == 0):
                self.epoch.append(epoch)
                self.paths.append(best_path)
                # self.__tspInstance.plot_tour(
                #     tour=best_path, name='@Epoch ' + str(epoch))
                print("-"*20 + " epoch " + str(epoch) + "-"*20)
                print('quality: {}'.format(quality))


if __name__ == '__main__':
    datasets = ['att48', 'berlin52']
    for dataset in datasets:
        print("-"*20 + dataset + "-"*20)
        tspInstance = TSPInstance(dataset)
        aco = ACO(tspInstance)
        aco.run()
