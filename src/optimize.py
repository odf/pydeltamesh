class FordFulkerson(object):
    def __init__(self, capacity, source, sink):
        self._capacity = capacity
        self._source = source
        self._sink = sink
        self._residual = None
        self._maxFlow = None
        self._minCut = None

    @property
    def capacity(self):
        return self._capacity

    @property
    def source(self):
        return self._source

    @property
    def sink(self):
        return self._sink

    def residual(self):
        if self._residual is None:
            self._residual = _fordFulkersonResidual(
                self.capacity, self.source, self.sink
            )
        return self._residual

    def maxFlow(self):
        if self._maxFlow is None:
            self._maxFlow = _maxFlow(self.capacity, self.residual())
        return self._maxFlow

    def minCut(self):
        if self._minCut is None:
            self._minCut = _minCut(self.capacity, self.residual(), self.source)
        return self._minCut


def _fordFulkersonResidual(capacity, source, sink):
    residual = {}

    for v, w, c in capacity:
        residual.setdefault(v, {})[w] = c
        residual.setdefault(w, {})[v] = 0

    while True:
        # Do a BFS to find an augmenting path

        parent = { source: None }
        queue = [source]
        success = False

        while len(queue) and not success:
            v = queue.pop(0)

            for w, c in residual[v].items():
                if c > 0 and not w in parent:
                    parent[w] = v

                    if w == sink:
                        success = True
                        break
                    else:
                        queue.append(w)

        # If no augmenting path was found, terminate

        if not success:
            break

        # Determine the capacity of the path

        path_capacity = float('inf')
        w = sink
        while w != source:
            v = parent[w]
            path_capacity = min(path_capacity, residual[v][w])
            w = v

        # Update the residual network

        w = sink
        while w != source:
            v = parent[w]
            residual[v][w] -= path_capacity
            residual[w][v] += path_capacity
            w = v

    # Return the residual network - max flow and min cut can be derived

    return residual


def _maxFlow(capacity, residual):
    return [
        (v, w, c - residual[v][w])
        for v, w, c in capacity
        if residual[v][w] < c
    ]


def _minCut(capacity, residual, source):
    reached = set([source])
    queue = [source]

    while len(queue):
        v = queue.pop(0)

        for w, c in residual[v].items():
            if c > 0 and not w in reached:
                reached.add(w)
                queue.append(w)

    return [
        (v, w, c)
        for v, w, c in capacity
        if v in reached and not w in reached
    ]


def minimumWeightAssignment(weightMatrix):
    import numpy as np

    M = np.array(weightMatrix, copy=True)
    nrows, ncols = M.shape
    n = max(nrows, ncols)

    if nrows < n or ncols < n:
        T = M
        M = np.zeros_like(M, shape=(n, n))
        M[:nrows, :ncols] = T

    for i in range(n):
        M[i, :] -= np.min(M[i, :])

    for j in range(n):
        M[:, j] -= np.min(M[:, j])

    while True:
        network = _coverageNetwork(M)

        if len(network.minCut()) == n:
            return [
                (i, j)
                for ((r, i), (s, j), _) in network.maxFlow()
                if r == 1 and s == 2 and i < nrows and j < ncols
            ]
        else:
            cut = network.minCut()
            rowsCovered = set(
                i for ((r, _), (s, i), _) in cut if r == 0 and s == 1
            )
            colsCovered = set(
                j for ((r, j), (s, _), _) in cut if r == 2 and s == 3
            )
            d = min(
                M[i][j]
                for i in range(n) if not i in rowsCovered
                for j in range(n) if not j in colsCovered
            )

            M -= d
            for i in rowsCovered:
                M[i, :] += d
            for j in colsCovered:
                M[:, j] += d


def _coverageNetwork(M):
    nrows, ncols = M.shape

    source = (0, 0)
    sink = (3, 0)

    capacity = (
        [
            (source, (1, i), 1)
            for i in range(nrows)
        ]
        +
        [
            ((1, i), (2, j), 1)
            for i in range(nrows)
            for j in range(ncols)
            if M[i][j] == 0
        ]
        +
        [ 
            ((2, j), sink, 1)
            for j in range(ncols)
        ]
    )

    return FordFulkerson(capacity, source, sink)


def minimumWeightAssignmentBruteForce(weightMatrix):
    nrows = len(weightMatrix)
    ncols = len(weightMatrix[0])

    if nrows > ncols:
        import numpy as np
        M = np.transpose(weightMatrix)
        return [ (j, i) for i, j in minimumWeightAssignmentBruteForce(M) ]
    else:
        best = None

        for a in _selections(range(ncols), nrows):
            cost = sum(weightMatrix[i][a[i]] for i in range(nrows))
            if best is None or cost < best[1]:
                best = a, cost

        a = best[0]
        return [(i, a[i]) for i in range(nrows)]


def _selections(a, n):
    if n <= 0:
        yield []
    else:
        a = list(a)
        for i in range(len(a)):
            for s in _selections(a[:i] + a[i + 1:], n - 1):
                yield [a[i]] + s


def _testMinimumWeightAssignment(weightMatrix):
    def cost(assignment):
        return sum(weightMatrix[i][j] for i, j in assignment)

    expected = minimumWeightAssignmentBruteForce(weightMatrix)
    found = minimumWeightAssignment(weightMatrix)

    assert(cost(found) == cost(expected))


def _runTest():
    import numpy as np

    for run in range(100):
        nrows = 1 + np.random.randint(7)
        ncols = 1 + np.random.randint(7)
        weights = np.random.randint(100, size=(nrows, ncols))
        _testMinimumWeightAssignment(weights)


if __name__ == "__main__":
    _runTest()
