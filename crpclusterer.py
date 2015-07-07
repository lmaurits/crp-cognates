import copy
import itertools
import math
import os.path
import pdb
import random
import sys

import scipy.stats
from scipy.misc import comb as Choose

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
        self.theta = 0.25
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
        self.build_distance_list()
        self.update_lh_cache("within")
        self.update_lh_cache("between")

        self.op_norms = {}
        self.op_hits = {}

        # Controls
        self.verbose = True
        self.change_partitions = True
        self.change_params = True

    def init_partitions(self, method="thresh"):
        """Construct initial partitions for all matrices.  This is not done
        randomly, but rather we try to start things off in a state which is
        likely to be not too terrible."""

        for matrix in self.matrices:
            roll = random.randint(1,3)
            if method == "thresh" or (method == "rand" and roll == 1):
                # Start off by sticking 0 in it's own class
                part = [[0]]
                # Then, for everything else...
                for i in range(1,len(matrix)):
                    assigned = False
                    for bit in part:
                        # Put it in a class if it's close to the first member of that class
                        if matrix[i][bit[0]] <= 0.33:
                            bit.append(i)
                            assigned = True
                            break
                    # If this item didn't get put in an existing class, put it in a new one
                    if not assigned:
                        part.append([i])

                # Make sure there are no duplicates or missing values
                assert sum([len(bit) for bit in part]) == len(matrix)
            elif method == "lump" or (method == "rand" and roll == 2):
                part = [list(range(0,len(matrix))),]
            elif method == "split" or (method == "rand" and roll == 3):
                part = [[i] for i in range(0,len(matrix))]

            self.partitions.append(part)
            self.dirty_parts.append(True)
            self.likelihoods.append(0)

    def build_distance_list(self):
        self.all_distances = []
        for m in self.matrices:
            for r in m:
                for x in r:
                    self.all_distances.append(x)
        self.all_distances = list(set(self.all_distances))

    def update_lh_cache(self, w_or_b):

        if w_or_b == "within":
            a, b = (0.0 - self.within_mu) / self.within_sigma, (1.0 - self.within_mu) / self.within_sigma
            dist = scipy.stats.truncnorm(a, b,loc=self.within_mu,scale=self.within_sigma)
        elif w_or_b == "between":
            a, b = (0.0 - self.between_mu) / self.between_sigma, (1.0 - self.between_mu) / self.between_sigma
            dist = scipy.stats.truncnorm(a, b, loc=self.between_mu,scale=self.between_sigma)

        lhs = dist.pdf(self.all_distances)
        lhs = [safety_log(l) for l in lhs]

        if w_or_b == "within":
            self.w_lh_cache = dict(zip(self.all_distances, lhs))
        elif w_or_b == "between":
            self.b_lh_cache = dict(zip(self.all_distances, lhs))

    def find_MAP(self, iterations=1000, hookfunc=None):
        """Attempt to find the partition and parameter values which
        maximimise the posterior probability of the data.  Runs for the
        specified number of iterations, or terminates after 100 consecutive
        failures to find an improved posterior."""

        self.posterior = self.compute_posterior()
        self.failed_attempts = 0
        if self.verbose:
            print("\t".join("Prior Lh Poster Theta W_mu W_sigma B_mu B_sigma".split()))
        for i in range(0,iterations):
            self.snapshot()
            self.dirty_theta = False
            self.dirty_parts = [False for part in self.partitions]
            self.draw_proposal(map_mode=True)
            new_poster = self.compute_posterior()
            if new_poster > self.posterior:
                # Accept
                self.posterior = new_poster
                if self.verbose:
                    self.instrument()
            else:
                # Reject
                self.revert()
            if hookfunc:
                hookfunc()

    def sample_posterior(self, iterations, burnin, lag, filename=None):
        self.posterior = self.compute_posterior()
        if self.verbose:
            print("\t".join("Prior Lh Poster Theta W_mu W_sigma B_mu B_sigma".split()))
        for i in range(0, burnin):
            self.make_mcmc_move()
        if filename:
            fp = open(filename, "w")
            fp.write("\t".join("Sample Prior Lh Poster Theta W_mu W_sigma B_mu B_sigma".split())+"\n")
        iters = 0
        while iters < iterations:
            for i in range(0, lag):
                self.make_mcmc_move()
            if self.verbose:
                self.instrument()
            if filename:
                fp.write(("%d\t" % iters) + "\t".join(["%.6f" % x for x in (self.prior, self.lh, self.posterior, self.theta, self.within_mu, self.within_sigma, self.between_mu, self.between_sigma)])+"\n")
            yield (self.posterior, self.theta, self.within_mu, self.within_sigma, self.between_mu, self.between_sigma, self.partitions)
            iters += 1
        if filename:
            fp.close()

    def make_mcmc_move(self):
        self.snapshot()
        self.dirty_theta = False
        self.dirty_parts = [False for part in self.partitions]
        self.draw_proposal()
        old_poster = self.posterior
        new_poster = self.compute_posterior()
        try:
            poster_ratio = math.exp(new_poster - old_poster)
        except OverflowError:
            if new_poster >= old_poster:
                poster_ratio = 1.0
            else:
                poster_ratio = 0.0
        acceptance_prob = poster_ratio * self.proposal_ratio
        if acceptance_prob >= 1.0 or random.random() <= acceptance_prob:
            # Accept
            self.posterior = new_poster
            self.op_hits[self.operator] = self.op_hits.get(self.operator, 0) + 1
        else:
            # Reject
            self.revert()
        self.op_norms[self.operator] = self.op_norms.get(self.operator, 0) + 1

    def compute_posterior(self):
        return self.compute_prior() + self.compute_lh()

    def compute_prior(self):
        """Compute log prior on model parameters."""

        # Domain constraints
        if not (
                (0.5 <= self.theta <= 1.5) and
                (0.0 <= self.within_mu <= 1.0) and
                (0.0 <= self.between_mu <= 1.0)
                ):
            prior = safety_log(0.0)
            self.prior = prior
            return prior

        prior = 0
        # Prior on theta
        # A fairly arbitrary Gamma prior which is basically chosen
        # to trade off between gernally preferring lower theta over higher
        # theta, but not wanting *too* low of a theta.
        #if self.theta > 3.0:
        #    p = 0.0
        #else:
        #    p = scipy.stats.gamma.pdf(self.theta, 1.2128, loc=0.0, scale=1.0315)
        #prior += safety_log(p)

        # Prior on within_mu
        # (Beta distribution prior)
#        dist = scipy.stats.beta(2, 5)
#        prior += safety_log(dist.pdf(self.within_mu))

        # Prior on within_sigma
        # (exponential prior)
        prior += safety_log(3.0*math.exp(-1*3.0*self.within_sigma))

        # Prior on between_mu
        # (Beta distribution prior)
        dist = scipy.stats.beta(5, 2)
        prior += safety_log(dist.pdf(self.between_mu))

        # Prior on between_sigma
        # (exponential prior)
        prior += safety_log(3.0*math.exp(-1*3.0*self.between_sigma))
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
        self.dirty_theta = False
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

    def draw_proposal(self, map_mode=False):
        """Make a random change to the state space."""
        self.proposal_ratio = 1.0
        if self.change_params and self.change_partitions:
            roll = random.random()
            if roll < 0.33:
                # Half the time, change the parameters
                self.move_change_params()
            elif roll < 0.75:
                # The other half, change the partition
                #self.move_change_partition()
                self.move_change_partition()
            else:
                if map_mode:
                    self.move_smart()
                else:
                    self.move_change_many_things()
        elif self.change_params:
            self.move_change_params()
        elif self.change_partitions:
            self.move_change_partition()

    def move_change_params(self):
        """Choose one of the model parameters at random and multiply it by a
        Normally distributed random scale."""

        # This move is completely symmetric so:
        self.proposal_ratio *= 1.0
        mult = - 1.0

        # Choose a parameter and scale it
        roll = random.random()
        if roll < 0.1666:
            self.operator = "scale_theta"
            while mult < 0:
                mult = random.normalvariate(1.0,0.3)
            self.theta *= mult
            self.dirty_theta = True
            # Return now so that dirty_parts is not touched
            return
        elif 0.1666 <= roll < 0.3333:
            if random.random() < 0.5:
                self.operator = "scale_w_mu"
                while mult < 0:
                    mult = random.normalvariate(1.0,0.10)
                self.within_mu *= mult
            else:
                self.operator = "sample_w_mu"
                self.within_mu = random.random()
            self.update_lh_cache("within")
        elif 0.3333 <= roll < 0.5:
            if random.random() < 0.5:
                self.operator = "scale_w_sigma"
                while mult < 0:
                    mult = random.normalvariate(1.0,0.02)
                self.within_sigma *= mult
            else:
                self.operator = "sample_w_sigma"
                self.within_sigma =  scipy.stats.expon(scale=1/5.0).rvs(1)[0]
            self.update_lh_cache("within")
        elif 0.5 <= roll < 0.6666:
            if random.random() < 0.5:
                self.operator = "scale_b_mu"
                while mult < 0:
                    mult = random.normalvariate(1.0,0.10)
                self.between_mu *= mult
            else:
                self.operator = "sample_b_mu"
                self.between_mu = random.random()
            self.update_lh_cache("between")
        elif 0.6666 <= roll < 0.8333:
            if random.random() < 0.5:
                self.operator = "scale_b_sigma"
                while mult < 0:
                    mult = random.normalvariate(1.0,0.05)
                self.between_sigma *= mult
            else:
                self.operator = "sample_sigma"
                self.between_sigma =  scipy.stats.expon(scale=1/5.0).rvs(1)[0]
            self.update_lh_cache("between")
        else:
            self.operator = "multi_scale"
            while mult < 0:
                mult = random.normalvariate(1.0,0.05)
            if mult > 1:
                bigger = mult
                smaller = 2 - mult
            else:
                smaller = mult
                bigger = 2 - mult
            # Scale theta in a random direction
            # Push the means in the sensible directions
            # Make the sigmas larger to make the move more favourable
            #self.theta *= mult
            #self.dirty_theta = True
            self.within_mu *= smaller
            self.within_sigma *= bigger
            self.between_mu *= bigger
            self.between_sigma *= bigger
            self.update_lh_cache("within")
            self.update_lh_cache("between")
        self.dirty_parts = [True for part in self.partitions]

    def move_change_partition(self):
        """Sample one of the partition changing moves at random and apply
        it."""

        moved = False
        while not moved:
            index = random.randint(0,len(self.partitions)-1)
            part = self.partitions[index]
            if random.random() < 0.75:
                operator = random.sample(
                        (   self.move_reassign,
                            self.move_swap,
                            self.move_shuffle),
                        1)[0]
            else:
                operator = random.sample(
                        (   self.move_merge,
                            self.move_split),
                        1)[0]
            moved = operator(part)
        self.dirty_parts[index] = True

    def move_change_many_things(self):
        # Change parameters
        mult = random.normalvariate(1.0,0.10)
        self.theta *= mult
        self.dirty_theta = True
        if mult > 1:
            # If theta got bigger, we'll want to split some cognate sets
            operator = self.move_split
            self.within_sigma /= mult
            self.between_sigma *= mult
        elif mult < 1:
            # If theta got smaller, we'll want to merge sets
            operator = self.move_merge
            self.within_sigma *= mult
            self.between_sigma /= mult
        self.update_lh_cache("within")
        self.update_lh_cache("between")

        # Split or merge some partitions
        n = random.randint(1,len(self.partitions))
        indices = random.sample(range(0,len(self.partitions)),n)
        for index in indices:
            part = self.partitions[index]
            self.dirty_parts[index] = operator(part)

    def move_merge(self, part):
        """Choose two sets of the partition at random and merge them."""
        self.operator = "merge"
        random.shuffle(part)
        old_len = len(part)
        if len(part) == 1:
            # If the partition is just one big set there's nothing to merge!
            return False
        newset = []
        newset.extend(part.pop())
        mid_size = len(newset)
        newset.extend(part.pop())
        part.append(newset)
        self.dirty_theta = True
        self.proposal_ratio *= (1 / len(newset))*Choose(len(newset),mid_size) / Choose(old_len,2)
        return True

    def move_split(self, part):
        """Choose a set of the partition at random and split it in two."""
        self.operator = "split"
        if all([len(bit) == 1 for bit in part]):
            # If all sets of the partition are singletons there's nothing to split!
            return False
        random.shuffle(part)
        old_part_length = len(part)
        partbit = part.pop()
        old_set_length = len(partbit)
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
        self.proposal_ratio *= Choose(len(part),2) / ((1/old_part_length)*Choose(old_set_length,len(part[-2])))
        return True

    def move_reassign(self, part):
        """Choose a random element of a random set and move it to a new
        random set."""
        self.operator = "reassign"
        if len(part) == 1:
            # If the partition is just one big set then we can't do anything!
            return False
        bit_a, bit_b = random.sample(part,2)
        bit_b.append(bit_a.pop())
        if not bit_a:
            part.remove(bit_a)
        self.dirty_theta = True
        self.proposal_ratio = 1.0
        return True

    def move_swap(self, part):
        """Choose two random sets and swap a random element of one with a
        random element of the other."""
        self.operator = "swap"
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
        self.proposal_ratio = 1.0
        return True

    def move_shuffle(self, part):
        """Randomly shuffle elements among the sets of the partition, while
        keeping the number and sizes of partitions constant."""
        self.operator = "shuffle"
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
        self.proposal_ratio = 1.0
        return True

    def move_smart(self):
        """For MAP searches: attempt a very smart move, which uses the
        distance matrices to make optimal choices."""
        self.operator = "smart"
        index = random.randint(0,len(self.partitions)-1)
        part = self.partitions[index]
        mat = self.matrices[index]
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
        print("\t".join(["%.6f" % x for x in (self.posterior, self.theta, self.within_mu, self.within_sigma, self.between_mu, self.between_sigma, [len(p) for p in self.partitions].count(1), [len(p) for p in self.partitions].count(max([len(p) for p in self.partitions])))]))

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

        lhs = dist.pdf(self.all_distances)
        lhs = [safety_log(l) for l in lhs]

        if w_or_b == "within":
            self.w_lh_cache = dict(zip(self.all_distances, lhs))
        elif w_or_b == "between":
            self.b_lh_cache = dict(zip(self.all_distances, lhs))
