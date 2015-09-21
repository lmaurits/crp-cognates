import itertools

class CladeCounter:
    
    def __init__(self):
        self.partitions = []
        self.counter = {}
        self.n = 0

    def update_counts(self, part):
        self.partitions.append(part)
        for bit in part:
            if len(bit) == 1:
                self.counter[tuple(bit)] = self.counter.get(tuple(bit),0) + 1
            else:
                for pair in itertools.combinations(bit, 2):
                    self.counter[tuple(sorted(pair))] = self.counter.get(tuple(sorted(pair)),0) + 1
        self.n += 1

    def get_consensus(self):
        self.n *= 1.0
        for p in self.counter:
            self.counter[p] /= self.n
        maxx = 0.0
        for part in self.partitions:
            prob = 1
            for bit in part:
                if len(bit) == 1:
                    prob *= self.counter[tuple(bit)]
                else:
                    for pair in itertools.combinations(bit, 2):
                        prob *= self.counter[tuple(sorted(pair))]
            if prob > maxx:
                maxx = prob
                consensus = part
        return consensus
