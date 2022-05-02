import timeit

from PSO import PSO
from TSPInstance import TSPInstance
import matplotlib.pyplot as plt

import numpy as np


class Evaluation(object):
    def __init__(self, dirpath='./dataset/', datasetName='a280', modelType="PSO") -> None:
        self.__tspInstance = TSPInstance(dirpath, datasetName)
        if modelType == "PSO":
            self.__model = PSO(self.__tspInstance)
            start = timeit.default_timer()
            self.__model.run()
            end = timeit.default_timer()
            # Best_path, Best = self.__model.run()
            array = np.transpose(np.vstack((self.__tspInstance.x, self.__tspInstance.y)))

            self.evaluate_cost(self.__model.best_l, self.__tspInstance.optTourDistance)
            self.draw_path("PSO", self.__model.location[self.__model.best_path])
            self.draw_path("OPT", array[self.__tspInstance.optimaltour])
            self.get_time(start, end)
            self.draw_iter("PSO", self.__model.iter_x, self.__model.iter_y)
            # self.draw_path("OPT", self.__tspInstance.get_distance_graph)
        elif modelType == "ACO":
            print("ACO")

    def evaluate_cost(self, ours, opt):
        print("估计最小的总代价: %.2f 目前最优的总代价: %.2f " % (ours, opt))

    def draw_path(self, title, point_array):
        """
                画个结果图
                point_array默认的size是城市数目*2
                Args:

         """
        point_array = np.vstack([point_array, point_array[0]])  # 形成回路
        fig, axs = plt.subplots(1, 1, sharex=False, sharey=False)
        axs.scatter(point_array[:, 0], point_array[:, 1])
        axs.plot(point_array[:, 0], point_array[:, 1])
        axs.set_title(title)
        plt.show()

    def get_time(self, start, end):
        """
                程序运行时间
                point_array默认的size是城市数目*2
                Args:

         """
        assert (start <= end)
        print("程序运行时间：%.2fs" % (end - start))
        plt.show()

    def draw_iter(self, title, iterations, best_record):
        fig, axs = plt.subplots(1, 1, sharex=False, sharey=False)
        axs.plot(iterations, best_record)
        axs.set_title(title)
        plt.show()


if __name__ == "__main__":
    evaluator = Evaluation(datasetName='st70', modelType="PSO")