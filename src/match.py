import numpy as np


# -- Imports that support type checking under Python 3

try:
    import typing
    have_typing = True
except ImportError:
    have_typing = False


if have_typing:
    from typing import Callable, Iterator, Optional, TypeVar

    T = TypeVar('T')
    VertexList = list[int]
    Face = list[int]
    FaceList = list[Face]
    OrientedEdge = tuple[int, int]
    Location = tuple[int, int]


# -- Helper types

class BufferedIterator(object):
    def __init__(self, gen):
        self._gen = gen
        self._buffer = []

    def _advance(self):
        try:
            val = next(self._gen)
            self._buffer.append(val)
            return True
        except StopIteration:
            return False

    def get(self, i):
        while len(self._buffer) <= i and self._advance():
            pass
        return self._buffer[i] if i < len(self._buffer) else None

    def result(self):
        while self._advance():
            pass
        return self._buffer[:]


Traversal = BufferedIterator


# -- Core data types

class Complex(object):
    def __init__(self, faces): # type: (FaceList) -> None
        self._faces = faces
        self._neighbors = _faceNeighbors(faces)
        self._degrees = _vertexDegrees(faces)

    @property
    def nrFaces(self): # type: () -> int
        return len(self._faces)

    def vertexDegree(self, v): # type: (int) -> int
        return self._degrees[v]

    def faceVertices(self, f): # type: (int) -> Face
        return self._faces[f]

    def faceNeighbors(self, f): # type: (int) -> list[Optional[Location]]
        return self._neighbors[f]



class Component(object):
    def __init__(self, complex, faceIndices):
        # type: (Complex, list[int]) -> None

        self._complex = complex
        self._faceIndices = faceIndices
        self._optimalTraversals = None # type: Optional[list[Traversal]]
        self._invariant = None # type: Optional[str]
        self._vertexOrders = None # type: Optional[list[VertexList]]

    @property
    def complex(self): # type: () -> Complex
        return self._complex

    @property
    def faceIndices(self): # type: () -> Iterator[int]
        return iter(self._faceIndices)

    @property
    def optimalTraversals(self): # type: () -> list[Traversal]
        if self._optimalTraversals is None:
            self._optimalTraversals = _optimalTraversals(self)
        return self._optimalTraversals

    @property
    def invariant(self): # type () -> str
        if self._invariant is None:
            self._invariant = " ".join([
                " ".join(map(str, f)) + " 0"
                for _, f in self.optimalTraversals[0].result()
            ])
        return self._invariant

    @property
    def vertexOrders(self): # type: () -> list[VertexList]
        if self._vertexOrders is None:
            self._vertexOrders = [
                _vertexOrder(t) for t in self.optimalTraversals
            ]
        return self._vertexOrders



class Topology(dict):
    def __init__(self, faces, vertices):
        # type: (FaceList, np.ndarray) -> None

        self._vertices = vertices

        for c in _components(Complex(faces)):
            self.setdefault(c.invariant, []).append(c.vertexOrders)

    def vertexPositions(self, indices): # type: (list[int]) -> np.ndarray
        return self._vertices[np.array(indices)]


# -- High level helper functions

def _components(complex): # type: (Complex) -> Iterator[Component]
    seen = set() # type: set[int]

    for f0 in range(complex.nrFaces):
        if not f0 in seen:
            seen.add(f0)
            queue = [f0]
            k = 0

            while k < len(queue):
                f = queue[k]
                k += 1

                for neighbor in complex.faceNeighbors(f):
                    if neighbor is not None:
                        g, _ = neighbor
                        if not g in seen:
                            seen.add(g)
                            queue.append(g)

            yield Component(complex, queue)


def _faceNeighbors(faces):
    # type: (FaceList) -> list[list[Optional[Location]]]

    edgeLocation = {} # type: dict[OrientedEdge, Location]

    for i, face in enumerate(faces):
        for k, (v, w) in enumerate(_cyclicPairs(face)):
            if (v, w) in edgeLocation:
                raise ValueError('an oriented edge appears more than once')
            else:
                edgeLocation[(v, w)] = (i, k)

    return [
        [edgeLocation.get((w, v)) for v, w in _cyclicPairs(face)]
        for face in faces
    ]


def _vertexDegrees(faces): # type: (FaceList) -> list[int]
    maxVertex = max(max(f) for f in faces)
    degree = [0] * (maxVertex + 1)

    for face in faces:
        for v in face:
            degree[v] += 1

    return degree


def _optimalTraversals(component): # type: (Component) -> list[Traversal]
    best = None # type: Optional[Traversal]
    result = [] # type: list[Traversal]

    for start in _startCandidates(component):
        candidate = BufferedIterator(
            _traverseAndRenumber(component.complex, *start)
        ) # type: Traversal
        if best is None:
            best = candidate
            result = [candidate]
        else:
            i = 0
            while True:
                a = best.get(i)
                b = candidate.get(i)
                if a is None:
                    assert(b is None)
                    result.append(candidate)
                    break
                elif a[1] < b[1]:
                    break
                elif b[1] < a[1]:
                    best = candidate
                    result = [candidate]
                    break
                else:
                    i += 1

    return result


def _startCandidates(component): # type: (Component) -> list[Location]
    complex = component.complex

    cost = {} # type: dict[int, int]
    for f in component.faceIndices:
        for v in complex.faceVertices(f):
            d = complex.vertexDegree(v)
            cost[d] = cost.get(d, 0) + 1

    d = min((c, i) for i, c in cost.items())[1]

    result = [] # type: list[Location]

    for f in component.faceIndices:
        vs = complex.faceVertices(f)

        for k, v in enumerate(vs):
            if complex.vertexDegree(v) == d:
                result.append((f, k))

    return result


def _traverseAndRenumber(complex, startFace, startOffset):
    # type: (Complex, int, int) -> Iterator[tuple[Face, Face]]

    vertexReIndex = {} # type: dict[int, int]
    nextVertex = 1
    queue = [(startFace, startOffset)]
    seen = set([startFace])

    while len(queue) > 0:
        f, k = queue.pop(0)
        vs = _rotate(k, complex.faceVertices(f))
        nbs = _rotate(k, complex.faceNeighbors(f))

        for v in vs:
            if not v in vertexReIndex:
                vertexReIndex[v] = nextVertex
                nextVertex += 1

        for nb in nbs:
            if nb is not None:
                fn, kn = nb
                if not fn in seen:
                    queue.append((fn, kn))
                    seen.add(fn)

        yield vs, [vertexReIndex[v] for v in vs]


def _vertexOrder(traversal): # type: (Traversal) -> VertexList
    faceData = traversal.result()
    vMax = max(max(f) for _, f in faceData)

    result = [0] * vMax
    for fOld, fNew in faceData:
        for v, w in zip(fOld, fNew):
            result[w - 1] = v

    return result


# -- Low level helper functions

def _cyclicPairs(indices): # type: (list[T]) -> Iterator[tuple[T, T]]
    return zip(indices, _rotate(1, indices))


def _rotate(i, items): # type: (int, list[T]) -> list[T]
    return items[i:] + items[:i]


# -- API functions

def topology(faces, vertices): # type: (FaceList, np.ndarray) -> Topology
    return Topology(faces, vertices)


if have_typing:
    CostFunction = Callable[[list[int], list[int]], float]


def match(topoA, topoB, metric=None):
    # type: (Topology, Topology, CostFunction) -> np.ndarray

    import optimize

    if metric is None:
        metric = lambda idcsA, idcsB: np.sum((
            topoA.vertexPositions(idcsA) - topoB.vertexPositions(idcsB)
        )**2)

    matchedVerticesInA = []
    matchedVerticesInB = []

    for key in topoA.keys():
        compA = topoA.get(key, [])
        compB = topoB.get(key, [])

        if len(compB) > 0:
            M = np.zeros((len(compA), len(compB)))

            for j, instA in enumerate(compA):
                for k, instB in enumerate(compB):
                    costs = [ metric(instA[0], orderB) for orderB in instB ]
                    M[j, k] = min(costs)

            assignment = optimize.minimumWeightAssignment(M)

            for j, k in assignment:
                instA = compA[j]
                instB = compB[k]
                # TODO avoid recomputing this?
                costs = [ metric(instA[0], orderB) for orderB in instB ]

                matchedVerticesInA.extend(compA[j][0])
                matchedVerticesInB.extend(compB[k][np.argmin(costs)])

    return np.transpose((matchedVerticesInA, matchedVerticesInB))
