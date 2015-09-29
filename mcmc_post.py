import itertools

class CladeCounter:
    
    def __init__(self):
        self.size = 0
        self.partitions = []
        self.counter = {}
        self.n = 0

    def update_counts(self, part):
        if self.size == 0:
            # First part
            self.size = sum([len(b) for b in part])
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
        # Build greedy potentially inconsistent consensus
        cons = []
        for i in range(0, self.size):
            bit = [i]
            for j in range(0, self.size):
                if i == j:
                    continue
                if self.counter.get(tuple(sorted([i,j])),0) >= 0.75:
                    bit.append(j)
            if sorted(bit) not in cons:
                cons.append(sorted(bit))
        # Enforce consensus
        for i in range(0, self.size):
            N = sum([b.count(i) for b in cons])
            if N > 1:
                # We've put i in the consensus partition twice
                # Keep only the most probable one...
                probs = []
                for b in cons:
                    if i not in b:
                        probs.append(0.0)
                        continue
                    prob = 1.0
                    for j in b:
                        if i == j:
                            continue
                        prob *= self.counter[tuple(sorted([i,j]))]
                    probs.append(prob)
                max_index = probs.index(max(probs))
                for index, b in enumerate(cons):
                    if i in b and index != max_index:
                        b.remove(i)
        while [] in cons:
            cons.remove([])
        # Check
        assert all([sum([b.count(i) for b in cons]) == 1 for i in range(0, self.size)])
        return cons 

    def get_consensus_alt(self):
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
