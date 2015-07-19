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
        return -999999.0

def stirling2(n,k):
    """Returns the stirling number Stirl2(n,k) of the second kind using recursion."""
    if k <= 1 or k == n:
        return 1
    elif k > n or n <= 0:
        return 0
    else:
        return stirling2(n-1, k-1) + k * stirling2(n-1, k)

def crp_lh(theta, partition):
    n = sum([len(s) for s in partition])
    lh = 0
    lh += safety_log( math.gamma(theta)*theta**len(partition) / math.gamma(theta + n) )
    for subset in partition:
        lh += safety_log(math.gamma(len(subset)))
    return lh

def simulate_crp(theta, n):
    part = []
    for i in range(0, n):
        if len(part) == 0:
            # First customer
            part.append([i])
        else:
            # Subsequent customer
            if random.random() <= theta /  (theta + i):
                # Start new table
                part.append([i])
            else:
                # Sample old table
                dist = [len(subset)/(theta + i) for subset in part]
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

        self.compute_mean_distance()

        # Priors
        self.theta_prior = scipy.stats.gamma(1.2128, loc=0.0, scale=1.0315)
        self.within_mu_prior = scipy.stats.beta(2,5)
        self.within_sigma_prior =  scipy.stats.expon(scale=1/12.0)
        self.between_mu_prior = scipy.stats.beta(5,2)
        self.between_sigma_prior =  scipy.stats.expon(scale=1/12.0)

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

    def compute_mean_distance(self):
        mean = 0
        norm = 0
        for m in self.matrices:
            for x,y in itertools.combinations(range(0,len(m)),2):
                mean += m[x][y]
                norm +=1
        self.mean_distance = mean/norm

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
                        if matrix[i][bit[0]] <= 0.50:
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

        max_poster = self.compute_posterior()
        self.failed_attempts = 0
        if self.verbose:
            print("\t".join("Prior Lh Poster Theta W_mu W_sigma B_mu B_sigma".split()))
        for i in range(0,iterations):
            self.snapshot()
            self.dirty_theta = False
            self.dirty_parts = [False for part in self.partitions]
            self.draw_proposal(map_mode=True)
            new_poster = self.compute_posterior()
            if new_poster > max_posterior:
                # Accept
                max_poster = new_poster
                if self.verbose:
                    self.instrument()
            else:
                # Reject
                self.revert()
            if hookfunc:
                hookfunc()

    def sample_posterior(self, iterations, burnin, lag, filename=None):
        self.compute_posterior()
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
        old_poster = self.posterior
        self.draw_proposal()
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
            self.op_hits[self.operator] = self.op_hits.get(self.operator, 0) + 1
        else:
            # Reject
            self.revert()
        self.op_norms[self.operator] = self.op_norms.get(self.operator, 0) + 1

    def compute_posterior(self):
        self.posterior = self.compute_prior() + self.compute_lh()
        return self.posterior

    def compute_prior(self):
        """Compute log prior on model parameters."""

        # Domain constraints
        if not (
           # No run-away params
                (0.0 <= self.theta <= 4.0) and
                (0.0 <= self.within_mu <= 1.0) and
                (0.0 <= self.between_mu <= 1.0)
            ) or (
           # Don't let *everything* be cognate to everything else...
               all([len(p)==1 for p in self.partitions])
            ) or (
           # ...but insist on at least one cognate
               not any([any([len(s)>1 for s in p]) for p in self.partitions])
            ):
           prior = safety_log(0.0)
           self.prior = prior
           return prior

        prior = 0

        # Prior on number of cognate classes
        no_cognate_classes = sum([len(p) for p in self.partitions])
        min_classes = len(self.partitions)
        max_classes = sum([len(m) for m in self.matrices])
        class_percentile = (no_cognate_classes - min_classes) / (max_classes - min_classes)
        mu = 0.264
        sigma = 0.1337
        a, b = (0.0 - mu) / sigma, (1.0 - mu) / sigma
        dist = scipy.stats.truncnorm(a, b,loc=mu,scale=sigma)
        prior += safety_log(dist.pdf(class_percentile))

        # Prior on theta
        # A fairly arbitrary Gamma prior which is basically chosen
        # to trade off between gernally preferring lower theta over higher
        # theta, but not wanting *too* low of a theta.
        prior += safety_log(self.theta_prior.pdf(self.theta))

        # Prior on within_mu
        # (Beta distribution prior)
        prior += safety_log(self.within_mu_prior.pdf(self.within_mu))

        # Prior on within_sigma
        # (exponential prior)
        prior += safety_log(self.within_sigma_prior.pdf(self.within_sigma))

        # Prior on between_mu
        # (Beta distribution prior)
        prior += safety_log(self.between_mu_prior.pdf(self.between_mu))

        # Prior on between_sigma
        # (exponential prior)
        prior += safety_log(self.between_sigma_prior.pdf(self.between_sigma))

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
        for part in self.partitions:
            lh += crp_lh(self.theta, part)
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
        self.snapped_prior = self.prior
        self.snapped_lh = self.lh
        self.snapped_posterior = self.posterior
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
        self.prior = self.snapped_prior
        self.lh = self.snapped_lh
        self.posterior = self.snapped_posterior
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
            if roll < 0.50:
                # Half the time, change the parameters
                self.move_change_params()
            elif roll < 0.75:
                # The other half, change the partition
                #self.move_change_partition()
                self.move_change_partition()
            elif roll < 0.99:
                if map_mode:
                    self.move_smart()
                else:
                    self.move_change_many_things()
            else:
                self.move_unstick()
        elif self.change_params:
            self.move_change_params()
        elif self.change_partitions:
            self.move_change_partition()

    def move_change_params(self):
        """Choose one of the model parameters at random and multiply it by a
        Normally distributed random scale."""

        # This move is completely symmetric so:
        mult = - 1.0

        # Choose a parameter and scale it
        roll = random.random()
        if roll < 0.1666:
            if random.random() < 0.5:
                self.operator = "scale_theta"
                self.proposal_ratio *= 1.0
                while mult < 0:
                    mult = random.normalvariate(1.0,0.1)
                self.theta *= mult
            else:
                self.operator = "sample_theta"
                old = self.theta
                self.theta = self.theta_prior.rvs(1)[0]
                self.proposal_ratio *= self.theta_prior.pdf(old) / self.theta_prior.pdf(self.theta)
            self.dirty_theta = True
            # Return now so that dirty_parts is not touched
            return
        elif 0.1666 <= roll < 0.3333:
            if random.random() < 0.5:
                self.operator = "scale_w_mu"
                self.proposal_ratio *= 1.0
                while mult < 0:
                    mult = random.normalvariate(1.0,0.10)
                self.within_mu *= mult
            else:
                self.operator = "sample_w_mu"
                old = self.within_mu
                self.within_mu = self.within_mu_prior.rvs(1)[0]
                self.proposal_ratio *= self.within_mu_prior.pdf(old) / self.within_mu_prior.pdf(self.within_mu)
            self.update_lh_cache("within")
        elif 0.3333 <= roll < 0.5:
            if random.random() < 0.5:
                self.operator = "scale_w_sigma"
                self.proposal_ratio *= 1.0
                while mult < 0:
                    mult = random.normalvariate(1.0,0.02)
                self.within_sigma *= mult
            else:
                self.operator = "sample_w_sigma"
                old = self.within_sigma
                self.within_sigma = self.within_sigma_prior.rvs(1)[0]
                self.proposal_ratio *= self.within_sigma_prior.pdf(old) / self.within_sigma_prior.pdf(self.within_sigma)
            self.update_lh_cache("within")
        elif 0.5 <= roll < 0.6666:
            if random.random() < 0.5:
                self.operator = "scale_b_mu"
                self.proposal_ratio *= 1.0
                while mult < 0:
                    mult = random.normalvariate(1.0,0.10)
                self.between_mu *= mult
            else:
                self.operator = "sample_b_mu"
                old = self.between_mu
                self.between_mu = self.between_mu_prior.rvs(1)[0]
                self.proposal_ratio *= self.between_mu_prior.pdf(old) / self.between_mu_prior.pdf(self.between_mu)
            self.update_lh_cache("between")
        elif 0.6666 <= roll < 0.8333:
            if random.random() < 0.5:
                self.operator = "scale_b_sigma"
                self.proposal_ratio *= 1.0
                while mult < 0:
                    mult = random.normalvariate(1.0,0.05)
                self.between_sigma *= mult
            else:
                self.operator = "sample_b_sigma"
                old = self.between_sigma
                self.between_sigma = self.between_sigma_prior.rvs(1)[0]
                self.proposal_ratio *= self.between_sigma_prior.pdf(old) / self.between_sigma_prior.pdf(self.between_sigma)
            self.update_lh_cache("between")
        else:
            if random.random() < 0.5:
                self.operator = "multi_scale"
                while mult < 0:
                    mult = random.normalvariate(1.0,0.05)
                # Scale theta and means in a random direction
                # Push the means in the sensible directions
                # Make the sigmas larger to make the move more favourable
                self.theta *= mult
                self.dirty_theta = True
                self.within_mu *= random.normalvariate(1.0, 0.05)
                self.between_mu *= random.normalvariate(1.0, 0.05)
                if mult > 1:
                    # We want to encourage splitting, so there
                    # will be more between distances
                    self.within_sigma /= mult
                    self.between_sigma *= mult
                else:
                    # We want to encourage lumping, so there
                    # will be more within distances
                    self.within_sigma *= mult
                    self.between_sigma /= mult
                self.update_lh_cache("within")
                self.update_lh_cache("between")
            else:
                self.operator = "multi_sample"
                old = self.theta
                self.theta = self.theta_prior.rvs(1)[0]
                self.proposal_ratio *= self.theta_prior.pdf(old) / self.theta_prior.pdf(self.theta)
                old = self.within_mu
                self.within_mu = self.within_mu_prior.rvs(1)[0]
                self.proposal_ratio *= self.within_mu_prior.pdf(old) / self.within_mu_prior.pdf(self.within_mu)
                old = self.within_sigma
                self.within_sigma = self.within_sigma_prior.rvs(1)[0]
                self.proposal_ratio *= self.within_sigma_prior.pdf(old) / self.within_sigma_prior.pdf(self.within_sigma)
                old = self.between_mu
                self.between_mu = self.between_mu_prior.rvs(1)[0]
                self.proposal_ratio *= self.between_mu_prior.pdf(old) / self.between_mu_prior.pdf(self.between_mu)
                old = self.between_sigma
                self.between_sigma = self.between_sigma_prior.rvs(1)[0]
                self.proposal_ratio *= self.between_sigma_prior.pdf(old) / self.between_sigma_prior.pdf(self.between_sigma)
                self.update_lh_cache("within")
                self.update_lh_cache("between")

        self.dirty_parts = [True for part in self.partitions]

    def sample_partition_operator(self):
        roll = random.random()
        if roll < 0.75:
            return random.sample(
                    (   self.move_reassign,
                        self.move_pluck,
                        self.move_swap,
                        self.move_shuffle),
                    1)[0]
        elif roll < 0.95:
            return random.sample(
                    (   self.move_merge,
                        self.move_split),
                    1)[0]
        else:
            return self.move_randomise

    def move_change_partition(self):
        """Sample one of the partition changing moves at random and apply
        it."""

        moved = False
        while not moved:
            operator = self.sample_partition_operator()
            index = random.randint(0,len(self.partitions)-1)
            part = self.partitions[index]
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
        self.dirty_parts = [True for p in self.partitions]

        # Randomly operate on some partitions
        n = random.randint(int(len(self.partitions)*0.25),len(self.partitions))
        indices = random.sample(range(0,len(self.partitions)),n)
        if random.random() < 0.5:
            # Apply the same operator to all the parts
            operator = self.sample_partition_operator()
            for part in self.partitions:
                operator(part)
        else:
            # Apply a different operator to each part
            for part in self.partitions:
                operator = self.sample_partition_operator()
                operator(part)

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
        partbit = ["Foo",]
        while len(partbit) == 1:
            partbit = random.sample(part,1)[0]
        part.remove(partbit)
        old_set_length = len(partbit)
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

    def move_randomise(self, part):
        self.operator = "randomise"
        self.proposal_ratio = 1.0
        # Sample a partition uniformly at random
        # Sample size of partition first
        dist = []
        n = sum([len(p) for p in part])
        for k in range(1,n):
            dist.append(stirling2(n, k))
        dist = [d/sum(dist) for d in dist]
        roll = random.random()
        cumul = 0
        for s, p in enumerate(dist):
            cumul += p
            if cumul >= roll:
                break
        size = s+1
        # Empty the partition
        while part:
            part.pop()
        # Each set of the partition requires at least 1 word...
        words = list(range(0,n))
        random.shuffle(words)
        for i in range(0,size):
            part.append([words.pop(),])
        # The remaining words can be assigned at random
        for w in words:
            index = random.randint(0,size-1)
            part[index].append(w)
        return True

    def move_reassign(self, part):
        """Choose a random element of a random set and move it to a new
        random set."""
        self.operator = "reassign"
        if len(part) == 1:
            # If the partition is just one big set then we can't do anything!
            return False
        bit_a, bit_b = random.sample(part,2)
        if len(bit_a) == 1:
            bit_a_single = True
        else:
            random.shuffle(bit_a)
            bit_a_single = False
        bit_b.append(bit_a.pop())
        if not bit_a:
            part.remove(bit_a)
        self.dirty_theta = True
        xp_to_x = 1/(len(bit_b)*Choose(len(part),2))   # reverse with reassign
        if bit_a_single:
            xp_to_x += 1/(len(part)*len(bit_b)) # reverse with pluck
        x_to_xp = 1/(len(bit_a)*Choose(len(part),2))   # forward with reassign
        self.proposal_ratio = xp_to_x / x_to_xp
        return True

    def move_pluck(self, part):
        """Choose a random element of a random set and put it into a new set all of its own."""
        self.operator = "pluck"
        if all([len(bit) == 1 for bit in part]):
            # No point if partition is all singletons
            return False
        bit = ["Foo",]
        while len(bit) == 1:
            bit = random.sample(part,1)[0]
        lenbit = len(bit)
        x = random.sample(bit, 1)[0]
        bit.remove(x)
        part.append([x,])
        self.dirty_theta = True
        xp_to_x = 1/Choose(len(part),2)   # reverse with reassign
        xp_to_x += 1/Choose(len(part),2)  # reverse with merge
        x_to_xp = 1/(len(part)*lenbit)    # forward with pluck
        if lenbit == 2:
            x_to_xp += 1/Choose(len(part),2)    # forward with split
        else:
            x_to_xp += 1/(Choose(len(part),2)*lenbit)    # forward with split
        self.proposal_ratio = xp_to_x / x_to_xp
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

    def move_unstick(self):
        self.operator = "unsticker"
        # Draw theta from prior
        old = self.theta
        self.theta = self.theta_prior.rvs(1)[0]
        self.proposal_ratio *= self.theta_prior.pdf(old) / self.theta_prior.pdf(self.theta)
        # Simulate all partitions from CRP process
        for i in range(0, len(self.partitions)):
            old_prob = math.exp(crp_lh(old, self.partitions[i]))
            self.partitions[i] = simulate_crp(self.theta, sum([len(s) for s in self.partitions[i]]))
            new_prob = math.exp(crp_lh(self.theta, self.partitions[i]))
            self.dirty_parts[i] = True
            self.proposal_ratio *= old_prob / new_prob
        # Sample both means from a narrow distribution centred on the empirical mean distance
        mean_dist = scipy.stats.norm(loc=self.mean_distance, scale=0.05)
        old = self.within_mu
        self.within_mu = scipy.stats.norm(loc=self.mean_distance, scale=0.05).rvs(1)[0]
        self.proposal_ratio *= mean_dist.pdf(old) / mean_dist.pdf(self.within_mu)
        old = self.between_mu
        self.between_mu = scipy.stats.norm(loc=self.mean_distance, scale=0.05).rvs(1)[0]
        self.proposal_ratio *= mean_dist.pdf(old) / mean_dist.pdf(self.between_mu)
        # Sample both variances from a narrow distribution centred somewhere in the higher region of the prior... 
        sigma_dist = scipy.stats.norm(loc=0.15, scale=0.025)
        old = self.within_sigma
        self.within_sigma = sigma_dist.rvs(1)[0]
        self.proposal_ratio *= sigma_dist.pdf(old) / sigma_dist.pdf(self.within_sigma)
        old = self.between_sigma
        self.between_sigma = sigma_dist.rvs(1)[0]
        self.proposal_ratio *= sigma_dist.pdf(old) / sigma_dist.pdf(self.between_sigma)
        # We have messed with ALL the things
        self.dirty_theta = True
        self.dirty_partitions = [True for p in self.partitions]
        self.update_lh_cache("within")
        self.update_lh_cache("between")
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
