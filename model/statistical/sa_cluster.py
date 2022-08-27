import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import *


class SimulatedAnnealingKMeansAlter:
    r"""
    The modified version of KMeans clustering without definite centers.
    The optimization method adopted is Simulated Annealing algorithm
    """

    def __init__(self, k_val):
        self.data = []
        self.parent = []
        self.k_val = k_val
        self.inlier_loss_metric = lambda x: (0, [])
        self.outlier_loss_metric = lambda x, y: (0, [])
        self.groups = [[1]]

    def set_data(self, data):
        self.data = data
        self.parent = [0 for i in range(len(self.data))]

    def set_loss(self, inlier_loss_metric, outlier_loss_metric):
        self.inlier_loss_metric = inlier_loss_metric
        self.outlier_loss_metric = outlier_loss_metric

    def cluster_init(self):
        for i in range(len(self.data)):
            self.parent[i] = random.randint(0, self.k_val - 1)

    def generate_group(self):
        self.groups = []
        for i in range(self.k_val):
            self.groups.append([])
        for i in range(len(self.data)):
            # print(self.parent[i])
            self.groups[self.parent[i]].append([i, self.data[i]])


    def generate_group_custom(self, parents):
        group = []
        for i in range(self.k_val):
            group.append([])
        for i in range(len(self.data)):
            group[parents[i]].append([i, self.data[i]])
        return group

    def calc_loss(self, group):
        inlier_losses = 0
        outlier_losses = 0
        candidate = []
        for i in range(len(group)):
            group_loss, inlier_candidate = self.inlier_loss_metric(group[i])
            inlier_losses += group_loss
            candidate.extend(inlier_candidate)
        for i in range(len(group)):
            for j in range(len(group)):
                if i == j:
                    continue
                inter_loss, outlier_candidate = self.outlier_loss_metric(group[i], group[j])
                candidate.extend(outlier_candidate)
                outlier_losses += inter_loss
        return inlier_losses + outlier_losses, candidate

    def get_neighbour_sol(self, cur_parents, candidates):
        if len(candidates) > 0:
            idx: int = random.choice(candidates)
            idx2: int = random.choice(candidates)
        else:
            idx = random.randint(0, len(self.data) - 1)
            idx2 = random.randint(0, len(self.data) - 1)
        new_sol = [cur_parents[i] for i in range(len(cur_parents))]
        ft = random.randint(1, 2)
        if ft == 1:
            while new_sol[idx] != cur_parents[idx]:
                new_sol[idx] = random.randint(0, self.k_val - 1)
        if ft == 2:
            tmp = new_sol[idx]
            new_sol[idx] = new_sol[idx2]
            new_sol[idx2] = tmp
        return new_sol

    def simulated_annealing(self, ts, te, cd, iterations=10):
        best_losses = 1e250
        self.cluster_init()
        self.generate_group()
        best_sol = [self.parent[i] for i in range(len(self.data))]
        cur_losses = 1e233
        cur_sol = [self.parent[i] for i in range(len(self.data))]
        cur_candidate = []
        t = ts
        while t >= te:
            print("Losses,", best_losses, t)
            new_losses = 1e233
            new_sol = []
            new_can = []
            for i in range(iterations):
                ns = self.get_neighbour_sol(cur_sol, cur_candidate)
                temp_group = self.generate_group_custom(ns)
                nl, nc = self.calc_loss(temp_group)
                if nl < new_losses:
                    new_losses = nl
                    new_sol = ns
                    new_can = nc
            if new_losses < cur_losses:
                cur_losses = new_losses
                cur_sol = new_sol
                cur_candidate = new_can
                if new_losses < best_losses:
                    best_losses = new_losses
                    best_sol = new_sol
            else:
                delta_t = new_losses - cur_losses
                prob = math.exp(-delta_t / t)
                if prob < random.random():
                    cur_losses = new_losses
                    cur_sol = new_sol
                    cur_candidate = new_can
            t = t * cd
        return best_sol


class LCluster(SimulatedAnnealingKMeansAlter):
    def __init__(self, lanes):
        super(LCluster, self).__init__(lanes)
        self.set_loss(LCluster.inlier_loss, LCluster.outlier_loss)

    def cluster_init(self):
        maxx = -1e9
        minx = 1e9
        for i in range(len(self.data)):
            if self.data[i][0]>maxx:
                maxx = self.data[i][0]
            if self.data[i][0]<minx:
                minx = self.data[i][0]
        for i in range(len(self.data)):
            self.parent[i] = int((self.data[i][0]-minx)/(maxx-minx)*self.k_val)
            if self.parent[i] >= self.k_val:
                self.parent[i] = self.k_val-1

    @staticmethod
    def inlier_loss(x):
        losses = 0
        candidate = [x[i][0] for i in range(len(x))]
        xs = np.array([[x[i][1][0]] for i in range(len(x))])
        ys = np.array([[x[i][1][1]] for i in range(len(x))])
        if len(xs) == 0:
            return 1e20, []
        model = RANSACRegressor().fit(xs, ys)
        yp = [x[i][1][0] * model.estimator_.coef_[0] + model.estimator_.intercept_ for i in range(len(x))]
        losses_f = 0
        losses_can = -1
        for i in range(len(x)):
            ls = (yp[i] - x[i][1][1]) ** 2
            if ls > 25:
                ls += 1e2
            if ls > losses_f:
                losses_f = ls
                losses_can = x[i][0]
            losses += ls
        # candidate = [x[i][0] for i in range(len(x))]
        if losses_can != -1:
            for f in range(40):
                candidate.append(losses_can)

        for i in range(len(x)):
            for j in range(len(x)):
                if i != j and x[i][1][1] == x[j][1][1]:
                    losses += 1e3
                    for f in range(2):
                        candidate.append(i)
                        candidate.append(j)
        return losses, candidate

    @staticmethod
    def outlier_loss(x, y):
        coef = []
        inter = []
        loss = 0
        xs = np.array([[x[i][1][0]] for i in range(len(x))])
        ys = np.array([[x[i][1][1]] for i in range(len(x))])
        if len(xs) == 0:
            return 1e9, []
        modelx = LinearRegression().fit(xs, ys)
        coef.append(modelx.coef_[0])
        inter.append(modelx.intercept_)
        xs = np.array([[y[i][1][0]] for i in range(len(y))])
        ys = np.array([[y[i][1][1]] for i in range(len(y))])
        if len(xs) == 0:
            return 1e9, []
        modely = LinearRegression().fit(xs, ys)
        coef.append(modely.coef_[0])
        inter.append(modely.intercept_)
        for i in range(len(coef)):
            for j in range(i + 1, len(coef)):
                if abs(coef[i] - coef[j]) < 1e-6:
                    continue
                xp = (inter[j] - inter[i]) / (coef[i] - coef[j])
                yp = coef[i] * xp + inter[i]
                for k in range(len(x)):
                    if x[k][1][0] > xp and x[k][1][1] > yp:
                        loss += 1e6
        return 0, []

    def get_neighbour_sol(self, cur_parents, candidates):
        if len(candidates) > 0:
            idx: int = random.choice(candidates)
            idx2: int = random.choice(candidates)
        else:
            idx = random.randint(0, len(self.data) - 1)
            idx2 = random.randint(0, len(self.data) - 1)
        new_sol = [cur_parents[i] for i in range(len(cur_parents))]
        ft = random.randint(1, 3)
        if ft == 1:
            while new_sol[idx] != cur_parents[idx]:
                new_sol[idx] = random.randint(0, self.k_val - 1)
        if ft == 2:
            tmp = new_sol[idx]
            new_sol[idx] = new_sol[idx2]
            new_sol[idx2] = tmp
        if ft >= 3:
            coef = []
            inter = []
            group = self.generate_group_custom(new_sol)
            for j in range(len(group)):
                xs = np.array([[group[j][i][1][0]] for i in range(len(group[j]))])
                ys = np.array([[group[j][i][1][1]] for i in range(len(group[j]))])
                if len(xs) == 0:
                    return new_sol
                model = LinearRegression().fit(xs, ys)
                coef.append(model.coef_[0])
                inter.append(model.intercept_)
            for i in range(len(coef)):
                for j in range(i + 1, len(coef)):
                    if abs(coef[i] - coef[j]) < 1e-6:
                        continue
                    xp = (inter[j] - inter[i]) / (coef[i] - coef[j])
                    yp = coef[i] * xp + inter[i]
                    for k in range(len(new_sol)):
                        if new_sol[k] <= i and self.data[k][0] > xp and self.data[k][1] > yp:
                            new_sol[k] = random.randint(j, len(coef) - 1)

        return new_sol


if __name__ == "__main__":
    random.seed(time.time())
    points = []
    for i in range(30):
        points.append([2 * i + 1, i])
        points.append([-2 * (i - 100) + 30 , i])
        points.append([0.5*(i-50)+90, i])
    model = LCluster(3)
    model.set_data(points)
    result = model.simulated_annealing(200000, 1000, 0.9, 5)
    print(result)
    cla = []
    clb = []
    clc = []
    for i in range(len(result)):
        if result[i] == 0:
            cla.append(points[i])
        elif result[i] == 1:
            clb.append(points[i])
        else:
            clc.append(points[i])
    cla = np.array(cla)
    clb = np.array(clb)
    clc = np.array(clc)
    plt.scatter(cla[:, 0], cla[:, 1], marker="o")
    plt.scatter(clb[:, 0], clb[:, 1], marker="*")
    plt.scatter(clc[:, 0], clc[:, 1], marker="^")
    plt.show()
