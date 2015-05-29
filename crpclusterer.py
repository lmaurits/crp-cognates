import copy
import itertools
import math
import os.path
import pdb
import random
import sys

import scipy.stats

def safety_log(x):
    try:
        return math.log(x)
    except ValueError:
        return -9999.0

class Clusterer:

    def __init__(self, matrices):
        # Core data structures
        self.matrices = matrices
        self.partitions = []

        # Model params
        self.theta = 1.00
        self.within_mu = 0.25
        self.within_sigma = 0.1
        self.between_mu = 0.75
        self.between_sigma = 0.1

        # Caching stuff
        self.dirty_theta = True
        self.dirty_parts = []
        self.crp_likelihood = 0.0
        self.likelihoods = []
        self.w_lh_cache = {}
        self.b_lh_cache = {}
        self.update_lh_cache("within")
        self.update_lh_cache("between")

        # Controls
        self.verbose = True
        self.change_partitions = True
        self.change_params = True

    def init_partitions(self):
        """Construct initial partitions for all matrices.  This is not done
        randomly, but rather we try to start things off in a state which is
        likely to be not too terrible."""

        for matrix in self.matrices:
            # Start off by sticking 0 in it's own class
            part = [[0]]
            # Then, for everything else...
            for i in range(1,len(matrix)):
                assigned = False
                for bit in part:
                    # Put it in a class if it's close to the first member of that class
                    if matrix[i][bit[0]] <= 0.35:
                        bit.append(i)
                        assigned = True
                        break
                # If this item didn't get put in an existing class, put it in a new one
                if not assigned:
                    part.append([i])

            # Make sure there are no duplicates or missing values
            assert sum([len(bit) for bit in part]) == len(matrix)

            self.partitions.append(part)
            self.dirty_parts.append(True)
            self.likelihoods.append(0)

    def update_lh_cache(self, w_or_b):

        if w_or_b == "within":
            dist = scipy.stats.norm(loc=self.within_mu,scale=self.within_sigma)
        elif w_or_b == "between":
            dist = scipy.stats.norm(loc=self.between_mu,scale=self.between_sigma)

        distances = [i/100 for i in range(0,101)]
        lhs = dist.pdf(distances)
        lhs = [safety_log(l) for l in lhs]

        if w_or_b == "within":
            self.w_lh_cache = dict(zip(distances, lhs))
        elif w_or_b == "between":
            self.b_lh_cache = dict(zip(distances, lhs))

    def find_MAP(self, iterations=1000):
        """Attempt to find the partition and parameter values which
        maximimise the posterior probability of the data.  Runs for the
        specified number of iterations, or terminates after 100 consecutive
        failures to find an improved posterior."""

        self.poster = self.compute_posterior()
        self.failed_attempts = 0
        if self.verbose:
            print("\t".join("Prior Lh Poster Theta W_mu W_sigma B_mu B_sigma".split()))
        for i in range(0,iterations):
            self.snapshot()
            self.dirty_theta = False
            self.dirty_parts = [False for part in self.partitions]
            self.draw_proposal()
            new_poster = self.compute_posterior()
            if new_poster > self.poster:
                # Accept
                self.poster = new_poster
                self.failed_attempts = 0
                if self.verbose:
                    self.instrument()
            else:
                # Reject
                self.revert()
                self.failed_attempts +=1
                if self.failed_attempts == 100:
                    break

    def compute_posterior(self):
        return self.compute_prior() + self.compute_lh()

    def compute_prior(self):
        """Compute log prior on model parameters."""

        prior = 0

        # Prior on theta
        # A fairly arbitrary Gamma prior which is basically chosen
        # to trade off between gernally preferring lower theta over higher
        # theta, but not wanting *too* low of a theta.
        p = scipy.stats.gamma.pdf(self.theta, 4.0, loc=0.0, scale=1/2.0)
        prior += safety_log(p)

        # Prior on within_mu
        # (Beta distribution prior)
        dist = scipy.stats.beta(2, 5)
        prior += safety_log(dist.pdf(self.within_mu))

        # Prior on within_sigma
        # (exponential prior)
        prior += safety_log(0.01*math.exp(-1*0.01*self.within_sigma))

        # Prior on between_mu
        # (Beta distribution prior)
        dist = scipy.stats.beta(5, 2)
        prior += safety_log(dist.pdf(self.between_mu))

        # Prior on between_sigma
        # (exponential prior)
        prior += safety_log(0.01*math.exp(-1*0.01*self.between_sigma))
        self.prior = prior
        return prior

    def compute_lh(self):
        """Compute log likelihood of partition (under CRP process) and
        distance matrices (under sampling from two distributions according
        to the partition)."""

        lh = self.get_partition_lh()
        lh += self.get_matrix_lh()
        self.lh = lh
        return lh

    def get_partition_lh(self):
        """Compute the probability of the current partition according to the
        current CRP model parameters."""
        if not self.dirty_theta:
            return self.crp_likelihood
        lh = 0
        for matrix, part in zip(self.matrices, self.partitions):
            lh += safety_log( math.gamma(self.theta)*self.theta**len(part) / math.gamma(self.theta + len(matrix)) )
            for bit in part:
                lh += safety_log(math.gamma(len(bit)))
        self.crp_likelihood = lh
        return lh

    def get_matrix_lh(self):
        """Compute the probability of the data matrix according to the
        current partition and distance distribution parameters."""
        for i, (part, dirty, matrix) in enumerate(zip(self.partitions, self.dirty_parts, self.matrices)):
            if not dirty:
                continue
            lh = 0
            for x,y in itertools.combinations(range(0,sum([len(subset) for subset in part])),2):
                if x == y:
                    continue
                if any([x in bit and y in bit for bit in part]):
                    lh += self.w_lh_cache[matrix[x][y]]
                else:
                    lh += self.b_lh_cache[matrix[x][y]]
            self.likelihoods[i] = lh
        return sum(self.likelihoods)

    def snapshot(self):
        """Backup everything which is modified by drawing a proposal."""
        self.snapped_partitions = copy.deepcopy(self.partitions)
        self.snapped_crp_likelihood = self.crp_likelihood
        self.snapped_likelihoods = self.likelihoods[:]
        self.snapped_w_lh_cache = self.w_lh_cache.copy()
        self.snapped_b_lh_cache = self.b_lh_cache.copy()
        self.snapped_theta = self.theta
        self.snapped_within_mu = self.within_mu
        self.snapped_within_sigma = self.within_sigma
        self.snapped_between_mu = self.between_mu
        self.snapped_between_sigma = self.between_sigma

    def revert(self):
        """Restore from backup everything which is modified by drawing a
        proposal."""
        self.partitions = self.snapped_partitions
        self.crp_likelihood = self.snapped_crp_likelihood
        self.likelihoods = self.snapped_likelihoods
        self.w_lh_cache = self.snapped_w_lh_cache
        self.b_lh_cache = self.snapped_b_lh_cache
        self.theta = self.snapped_theta
        self.within_mu = self.snapped_within_mu
        self.within_sigma = self.snapped_within_sigma
        self.between_mu = self.snapped_between_mu
        self.between_sigma = self.snapped_between_sigma

    def draw_proposal(self):
        """Make a random change to the state space."""
        if self.change_params and self.change_partitions:
            roll = random.random()
            if 0 <= roll < 0.50:
                # Half the time, change the parameters
                self.move_change_params()
            else:
                # The other half, change the partition
                self.move_change_partition()
        elif self.change_params:
                self.move_change_params()
        elif self.change_partitions:
                self.move_change_partition()

    def move_change_params(self):
        """Choose one of the model parameters at random and multiply it by a
        Normally distributed random scale."""
        # Choose a scaling value
        roll = random.randint(1,4)
        if roll == 1:
            mult = random.normalvariate(1.0,0.05)
        elif roll == 2:
            mult = random.normalvariate(1.0,0.10)
        elif roll == 3:
            mult = random.normalvariate(1.0,0.20)
        else:
            mult = random.normalvariate(1.0,0.30)

        # Choose a parameter and scale it
        roll = random.random()
        if roll < 0.20:
            self.theta *= mult
            self.dirty_theta = True
            # Return now so that dirty_parts is not touched
            return
        elif 0.20 <= roll < 0.40:
            self.within_mu *= mult
            self.update_lh_cache("within")
        elif 0.40 <= roll < 0.60:
            self.within_sigma *= mult
            self.update_lh_cache("within")
        elif 0.60 <= roll < 0.80:
            self.between_mu *= mult
            self.update_lh_cache("between")
        else:
            self.between_sigma *= mult
            self.update_lh_cache("between")
        self.dirty_parts = [True for part in self.partitions]

    def move_change_partition(self):
        """Sample one of the partition changing moves at random and apply
        it."""

        moved = False
        while not moved:
            index = random.randint(0,len(self.partitions)-1)
            part = self.partitions[index]
            operator = random.sample(
                    (   self.move_merge,
                        self.move_split,
                        self.move_reassign,
                        self.move_swap,
                        self.move_shuffle,
                        self.move_smart),
                    1)[0]
            if operator == self.move_smart:
                matrix = self.matrices[index]
                moved = operator(part, matrix)
            else:
                moved = operator(part)
        self.dirty_parts[self.partitions.index(part)] = True

    def move_merge(self, part):
        """Choose two sets of the partition at random and merge them."""
        random.shuffle(part)
        if len(part) == 1:
            # If the partition is just one big set there's nothing to merge!
            return False
        newset = []
        newset.extend(part.pop())
        newset.extend(part.pop())
        part.append(newset)
        self.dirty_theta = True
        return True

    def move_split(self, part):
        """Choose a set of the partition at random and split it in two."""
        if all([len(bit) == 1 for bit in part]):
            # If all sets of the partition are singletons there's nothing to split!
            return False
        random.shuffle(part)
        partbit = part.pop()
        while len(partbit) == 1:
            part.append(partbit)
            random.shuffle(part)
            partbit = part.pop()
        if len(partbit) == 2:
            part.append([partbit[0],])
            part.append([partbit[1],])
        else:
            random.shuffle(partbit)
            pivot = random.randint(1,len(partbit)-2)
            part.append(partbit[0:pivot])
            part.append(partbit[pivot:])
        self.dirty_theta = True
        return True

    def move_reassign(self, part):
        """Choose a random element of a random set and move it to a new
        random set."""
        if len(part) == 1:
            # If the partition is just one big set then we can't do anything!
            return False
        bit_a, bit_b = random.sample(part,2)
        bit_b.append(bit_a.pop())
        if not bit_a:
            part.remove(bit_a)
        self.dirty_theta = True
        return True

    def move_swap(self, part):
        """Choose two random sets and swap a random element of one with a
        random element of the other."""
        if len(part) == 1:
            # We need at least two partitions
            return False
        bit_a, bit_b = random.sample(part,2)
        random.shuffle(bit_a)
        x_a = bit_a.pop()
        random.shuffle(bit_b)
        x_b = bit_b.pop()
        bit_a.append(x_b)
        bit_b.append(x_a)
        return True

    def move_shuffle(self, part):
        """Randomly shuffle elements among the sets of the partition, while
        keeping the number and sizes of partitions constant."""
        n = sum([len(bit) for bit in part])
        words = list(range(0,n))
        random.shuffle(words)
        new_part = []
        while part:
            bit = part.pop()
            new_bit = []
            for i in bit:
                new_bit.append(words.pop())
            new_part.append(new_bit)
        part.extend(new_part)
        self.dirty_theta = True
        return True

    def move_smart(self, part, mat):
        """For MAP searches: attempt a very smart move, which uses the
        distance matrices to make optimal choices."""
        if len(part) == 1:
            # Given only a single grouping, find the word
            # with the greatest mean distance to other words in the group
            # and remove it to form its own group
            bit = part[0]
            mean_dists = [sum([mat[i][j] for j in bit if i!= j])/(len(bit)-1) for i in bit]
            max_dist = max(mean_dists)
            max_index = mean_dists.index(max_dist)
            mover = bit[max_index]
            bit.remove(mover)
            part.append([mover])
        else:
            # Given multiple groupings, pick a random word from a random group
            # and put it in the group which minimises its mean distance to
            # other words in the group
            random.shuffle(part)
            partbit = part.pop()
            random.shuffle(partbit)
            mover = partbit.pop()
            mean_dists = [sum([mat[mover][j] for j in bit])/len(bit) for bit in part]
            min_dist = min(mean_dists)
            min_index = mean_dists.index(min_dist)
            part[min_index].append(mover)
            if partbit:
                part.append(partbit)
        self.dirty_theta = True
        return True

    def instrument(self):
        print("\t".join(["%.2f" % x for x in (self.prior, self.lh, self.poster, self.theta, self.within_mu, self.within_sigma, self.between_mu, self.between_sigma)]))

class BetaClusterer(Clusterer):

    def __init__(self, matrices):

        Clusterer.__init__(self, matrices)
        self.within_mu = 2
        self.within_sigma = 2
        self.between_mu = 2
        self.between_sigma = 2
        self.update_lh_cache("within")
        self.update_lh_cache("between")

    def compute_prior(self):
        """Compute log prior on model parameters."""

        prior = 0

        # Prior on theta
        # A fairly arbitrary Gamma prior which is basically chosen
        # to trade off between gernally preferring lower theta over higher
        # theta, but not wanting *too* low of a theta.
        p = scipy.stats.gamma.pdf(self.theta, 4.0, loc=0.0, scale=1/2.0)
        prior += safety_log(p)

        # Prior on within_mu
        # (Beta distribution prior)
        prior += safety_log(0.01*math.exp(-1*0.01*self.within_mu))

        # Prior on within_sigma
        # (exponential prior)
        prior += safety_log(0.01*math.exp(-1*0.01*self.within_sigma))

        # Prior on between_mu
        # (Beta distribution prior)
        prior += safety_log(0.01*math.exp(-1*0.01*self.between_mu))

        # Prior on between_sigma
        # (exponential prior)
        prior += safety_log(0.01*math.exp(-1*0.01*self.between_sigma))
        self.prior = prior
        return prior

    def update_lh_cache(self, w_or_b):

        if w_or_b == "within":
            dist = scipy.stats.beta(self.within_mu,self.within_sigma)
        elif w_or_b == "between":
            dist = scipy.stats.beta(self.between_mu,self.between_sigma)

        distances = [i/100 for i in range(0,101)]
        lhs = dist.pdf(distances)
        lhs = [safety_log(l) for l in lhs]

        if w_or_b == "within":
            self.w_lh_cache = dict(zip(distances, lhs))
        elif w_or_b == "between":
            self.b_lh_cache = dict(zip(distances, lhs))
