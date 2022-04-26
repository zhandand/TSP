"""
解决TSP问题的蚁群算法（Ant Colony Optimization）
"""
import random

(city_num, ant_num) = (50, 50)
(alpha, beta) = (1.0, 2.0)
pheromone_graph = [[1.0 for col in range(city_num)] for row in range(city_num)]
distance_graph = [[1.0 for col in range(city_num)] for row in range(city_num)]


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
    
    def __choose_next_city(self):
        next_city = -1
        select_cities_prob = [0.0 for _ in range(city_num)]
        total_prob = 0.0

        for i in range(city_num):
            if self.avaliable_city[i] is True:
                select_cities_prob[i] = pow(pheromone_graph[self.current_city][i],alpha)* pow(1/pow())