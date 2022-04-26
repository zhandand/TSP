"""
解决TSP问题的蚁群算法（Ant Colony Optimization）
"""
import random
import sys

import numpy as np

(city_num, ant_num) = (50, 50)
# parameter
(ALPHA, BETA, RHO, Q) = (1.0, 2.0, 0.5, 100.0)
pheromone_graph = [[1.0 for _ in range(city_num)] for _ in range(city_num)]
distance_graph = [[1.0 for _ in range(city_num)] for _ in range(city_num)]


class Ant(object):
    def __init__(self, id) -> None:
        super().__init__()
        self.id = id
        self.__initialize()

    def __initialize(self):
        self.current_city = random.randint(0, city_num-1)       # 随机初始化城市
        self.path = [self.current_city]                         # ant的轨迹
        self.total_distance = 0.0
        self.step = 1                                           # 移动次数
        self.avaliable_city = [True for i in range(city_num)]   # 未访问城市
        self.avaliable_city[self.current_city] = False
    
    def rand_choose(self, p):
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

    def __choose_next_city(self):
        select_cities_prob = [0.0 for _ in range(city_num)]

        for i in range(city_num):
            if self.avaliable_city[i] is True:
                try:
                    # the probability is proportional to pheromone, inversely proportional to distance
                    select_cities_prob[i] = pow(pheromone_graph[self.current_city][i], ALPHA) * pow(
                        1.0/pow(distance_graph[self.current_city][i]), BETA)
                except ZeroDivisionError as e:
                    print('Ant ID: {ID}, current city: {current}, target city: {target}'.format(
                        ID=self.ID, current=self.current_city, target=i))
                    sys.exit(1)
        select_cities_prob = np.array(select_cities_prob)
        select_cities_prob = select_cities_prob / np.sum(select_cities_prob) # normalization
        next_city = self.rand_choose(self,select_cities_prob)
        return next_city
    
    def __move(self, next_city):
        """
            ant moves to the next city
        Args:
            next_city (_type_): _description_
        """
        self.path.append(next_city)
        self.avaliable_city[next_city] = False
        self.total_distance += distance_graph[self.current_city][next_city]
        self.current_city = next_city
        self.step +=1
    
    def search_food(self):
        """
            search one circle
        """
        self.__initialize()
        # search all the cities
        while self.step < city_num:
            next_city = self.__choose_next_city()
            self.__move(next_city)
        self.total_distance += distance_graph[self.path[-1]][self.path[0]]


