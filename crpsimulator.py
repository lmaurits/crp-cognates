import itertools
import random
import scipy.stats

class CrpSimulator:

    def __init__(self, theta, w_mu, w_sigma, b_mu, b_sigma):

        self.theta = theta
        self.w_mu = w_mu
        self.w_sigma = w_sigma
        self.b_mu = b_mu
        self.b_sigma = b_sigma
        self.within_distances = []
        self.between_distances = []

    def draw_within(self):
        if not self.within_distances:
            a, b = (0.0 - self.w_mu) / self.w_sigma, (1.0 - self.w_mu) / self.w_sigma
            self.within_distances = list(scipy.stats.truncnorm(a,b,self.w_mu, self.w_sigma).rvs(1000))
        return self.within_distances.pop()

    def draw_between(self):
        if not self.between_distances:
            a, b = (0.0 - self.b_mu) / self.b_sigma, (1.0 - self.b_mu) / self.b_sigma
            self.between_distances = list(scipy.stats.truncnorm(a,b,self.b_mu, self.b_sigma).rvs(1000))
        return self.between_distances.pop()

    def simulate_datapoint(self, n):

        part = self.simulate_partition(n)
        matrix = self.simulate_matrix(part)
        return part, matrix

    def simulate_partition(self, n):

        part = []
        for i in range(0, n):
            if len(part) == 0:
                # First customer
                part.append([i])
            else:
                # Subsequent customer
                if random.random() <= self.theta /  (self.theta + i):
                    # Start new table
                    part.append([i])
                else:
                    # Sample old table
                    dist = [len(subset)/(self.theta + i) for subset in part]
                    dist = [p/sum(dist) for p in dist]
                    roll = random.random()
                    cumul = 0
                    for subset, p in zip(part, dist):
                        cumul += p
                        if cumul >= roll:
                            subset.append(i)
                            break
        assert sum([len(subset) for subset in part]) == n
        return part

    def simulate_matrix(self, part):
        
        matrix = []
        n = sum([len(subset) for subset in part])
        matrix = [[0.0 for i in range(0,n)] for j in range(0,n)]
        for x,y in itertools.combinations(range(0,n),2):
            if x == y:
                matrix[x][y] = 0.0
            else:
                if any([x in subset and y in subset for subset in part]):
                    matrix[x][y] = max(0.0, min(1.0,round(self.draw_within(),3)))
                else:
                    matrix[x][y] = max(0.0, min(1.0,round(self.draw_between(),3)))
                matrix[y][x] = matrix[x][y]
        return matrix

class BetaCrpSimulator(CrpSimulator):

    def __init__(self, theta, w_mu, w_sigma, b_mu, b_sigma):
        CrpSimulator.__init__(self, theta, w_mu, w_sigma, b_mu, b_sigma)

    def draw_within(self):
        if not self.within_distances:
            self.within_distances = list(scipy.stats.beta(self.w_mu, self.w_sigma).rvs(1000))
        return self.within_distances.pop()

    def draw_between(self):
        if not self.between_distances:
            self.between_distances = list(scipy.stats.beta(self.b_mu, self.b_sigma).rvs(1000))
        return self.between_distances.pop()
