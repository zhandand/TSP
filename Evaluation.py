
from PSO import PSO
from ACO import ACO
from TSPInstance import TSPInstance
import matplotlib.pyplot as plt
import os

import numpy as np


class Evaluation(object):
    def __init__(self, dirpath='./dataset/', datasetName='a280', modelType="PSO") -> None:
        self.init_result_path(modelType, datasetName)
        self.__tspInstance = TSPInstance(dirpath, datasetName)
        coordinates = np.transpose(
            np.vstack((self.__tspInstance.x, self.__tspInstance.y)))
        print("{} is running on {}".format(modelType, datasetName))
        if modelType == "PSO":
            self.__model = PSO(self.__tspInstance)
            # ! 建议在模型的run方法中使用timer装饰器
            self.__model.run(self.result_path)
            self.evaluate_cost(self.__model.best_l,
                               self.__tspInstance.optTourDistance)
            assert len(self.__model.iter_y) != 0

            self.draw_quality("PSO Quality", self.__model.iter_x, self.__model.iter_y/self.__tspInstance.optTourDistance)

            for i in range( len( self.__model.epoch)  ):
                self.draw_tour( self.__model.location[ self.__model.paths[i]],  "City access sequence @" + str(
                    self.__model.epoch[i]),"@" + str(self.__model.epoch[i]))
            self.draw_tour(
                self.__model.location[self.__model.best_path], "City access sequence @best", "best")
            # self.draw_path("OPT", self.__tspInstance.get_distance_graph)
        elif modelType == "ACO":
            self.__model = ACO(self.__tspInstance)
            self.__model.run(self.result_path)

            assert len(self.__model.quality) != 0
            assert len(self.__model.epoch) == len(self.__model.paths)

            self.evaluate_cost(self.__model.global_opt_distance,
                               self.__tspInstance.optTourDistance)

            self.draw_quality("ACO Quality", range(
                len(self.__model.quality)), self.__model.quality)

            for i in range(len(self.__model.epoch)):
                self.draw_tour(coordinates[self.__model.paths[i]], "City access sequence @" + str(
                    self.__model.epoch[i]), "@" + str(self.__model.epoch[i]))

            self.draw_tour(
                coordinates[self.__model.paths[i]], "City access sequence @best", "best")
        elif modelType == "Annealing":
            # todo: write your program here
            pass

    def evaluate_cost(self, cur, opt):
        """
            print distance then save in file
        Args:
            cur (_type_): current distance
            opt (_type_): optimal distance
        """
        str = "估计最小的总代价: %.2f 目前最优的总代价: %.2f 最佳quality: %.2f " % (cur, opt, cur / opt)
        print(str)
        with open(self.result_path + '/log.txt', mode='a+', encoding='utf-8') as f:
            f.write(str)

    def draw_tour(self, tour, title, filename):
        """
        draw the path according the the tour
        Args:
            tour (_type_): [[x,y] * #city]
            title (_type_): title for the image
            filename (_type_): filename for saving
        """
        tour = np.vstack([tour, tour[0]])  # 形成回路
        fig, axs = plt.subplots(1, 1, sharex=False, sharey=False)
        axs.scatter(tour[:, 0], tour[:, 1], s =10)
        axs.plot(tour[:, 0], tour[:, 1])
        axs.set_title(title)
        plt.savefig(self.result_path + '/' +
                    filename + '.png', bbox_inches='tight')
        plt.show()#先save后show
        plt.cla()
        plt.close()

    # todo: use timer decorater rather than this function, recommend delete this
    def get_time(self, start, end):
        """
                程序运行时间
                point_array默认的size是城市数目*2
                Args:

         """
        assert (start <= end)
        print("程序运行时间：%.2fs" % (end - start))

    def draw_quality(self, title, iterations, quality):
        """
        Args:
            title (str): title for the image
            iterations (list): epochs 
            quality (list): [current distance / optimal distance]
        """
        fig, axs = plt.subplots(1, 1, sharex=False, sharey=False)
        axs.plot(iterations, quality)
        axs.set_title(title)
        plt.savefig(self.result_path + 'quality' + '.png', bbox_inches='tight')
        plt.show()
        plt.cla()
        plt.close()

    def init_result_path(self, model, dataset):
        """
        init path for result, if not exist, then create it
        expect path as ./result/dataset/model 

        Args:
            model (_type_): model name 
            dataset (_type_): dataset name
        """
        if not os.path.exists('./result/' + dataset):
            os.mkdir('./result/' + dataset)
        if not os.path.exists('./result/' + dataset + "/" + model):
            os.mkdir('./result/' + dataset + "/" + model)
        self.result_path = './result/' + dataset + "/" + model


if __name__ == "__main__":
    # evaluator = Evaluation(datasetName='st70', modelType="PSO")
    datasets = ['att48', 'a280', 'berlin52', 'ch130', 'ch150', 'eil51', 'eil76', 'eil101', 'gr96', 'gr202', 'gr666', 'kroA100',
    'kroC100', 'kroD100','st70']
    # datasets = ['att48']
    for dataset in datasets:
        evaluator = Evaluation(datasetName=dataset, modelType="PSO")
