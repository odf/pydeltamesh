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
                if parent.get(w) is None and c > 0:
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


def maxFlow(capacity, residual):
    return [
        (v, w, c - residual[v][w])
        for v, w, c in capacity
        if residual[v][w] < c
    ]


if __name__ == "__main__":
    import pprint

    capacity = [
        ("source", "c1", 1),
        ("source", "c2", 1),
        ("source", "c3", 1),
        ("c1", "r1", 1),
        ("c1", "r2", 1),
        ("c2", "r3", 1),
        ("c3", "r3", 1),
        ("r1", "sink", 1),
        ("r2", "sink", 1),
        ("r3", "sink", 1),
    ]

    residual = _fordFulkersonResidual(capacity, "source", "sink")

    print("\nResidual:")
    pprint.pp(residual)

    print("\nMax flow:")
    pprint.pp(maxFlow(capacity, residual))
