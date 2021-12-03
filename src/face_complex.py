from hashlib import sha1
from typing import Generic, Iterator, Optional, TypeVar

import numpy as np

T = TypeVar('T')


# -- Simple type aliases

VertexList = list[int]
Face = list[int]
FaceList = list[Face]
OrientedEdge = tuple[int, int]
Location = tuple[int, int]


# -- Helper types

class BufferedIterator(Generic[T]):
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


Traversal = BufferedIterator[tuple[Face, Face]]



# -- Core data types

class Complex(object):
    def __init__(self, faces: FaceList):
        self._faces = faces
        self._neighbors = _faceNeighbors(faces)
        self._degrees = _vertexDegrees(faces)
        self._vertices = [v for v, m in enumerate(self._degrees) if m > 0]

    @property
    def nrVertices(self) -> int:
        return len(self._vertices)

    @property
    def vertices(self) -> Iterator[int]:
        return iter(self._vertices)

    @property
    def nrFaces(self) -> int:
        return len(self._faces)

    def vertexDegree(self, v: int) -> int:
        return self._degrees[v]

    def faceVertices(self, f: int) -> Face:
        return self._faces[f]

    def faceNeighbors(self, f: int) -> list[Optional[Location]]:
        return self._neighbors[f]



class Component(object):
    def __init__(self, complex: Complex, faceIndices: list[int]):
        self._complex = complex
        self._faceIndices = faceIndices
        self._optimalTraversals: Optional[list[Traversal]] = None
        self._invariant: Optional[str] = None
        self._fingerprint: Optional[str] = None
        self._vertexOrders: Optional[list[VertexList]] = None

    @property
    def nrFaces(self) -> int:
        return len(self._faceIndices)

    @property
    def complex(self) -> Complex:
        return self._complex

    @property
    def faceIndices(self) -> Iterator[int]:
        return iter(self._faceIndices)

    @property
    def optimalTraversals(self) -> list[Traversal]:
        if self._optimalTraversals is None:
            self._optimalTraversals = _optimalTraversals(self)
        return self._optimalTraversals

    @property
    def invariant(self) -> str:
        if self._invariant is None:
            self._invariant = " ".join([
                " ".join(map(str, f)) + " 0"
                for _, f in self.optimalTraversals[0].result()
            ])
        return self._invariant

    @property
    def fingerprint(self) -> str:
        if self._fingerprint is None:
            self._fingerprint = sha1(str.encode(self.invariant)).hexdigest()
        return self._fingerprint

    @property
    def vertexOrders(self) -> list[VertexList]:
        if self._vertexOrders is None:
            self._vertexOrders = [
                _vertexOrder(t) for t in self.optimalTraversals
            ]
        return self._vertexOrders



class Topology(dict[str, list[list[VertexList]]]):
    def __init__(self, faces: FaceList):
        for c in _components(Complex(faces)):
            self.setdefault(c.invariant, []).append(c.vertexOrders)



# -- High level helper functions

def _components(complex: Complex) -> Iterator[Component]:
    seen: set[int] = set()

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


def _faceNeighbors(faces: FaceList) -> list[list[Optional[Location]]]:
    edgeLocation: dict[OrientedEdge, Location] = {}

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


def _vertexDegrees(faces: FaceList) -> list[int]:
    maxVertex = max(max(f) for f in faces)
    degree = [0] * (maxVertex + 1)

    for face in faces:
        for v in face:
            degree[v] += 1

    return degree


def _optimalTraversals(component: Component) -> list[Traversal]:
    best: Optional[Traversal] = None
    result: list[Traversal] = []

    for start in _startCandidates(component):
        candidate: Traversal = BufferedIterator(
            _traverseAndRenumber(component.complex, *start)
        )
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


def _startCandidates(component: Component) -> list[Location]:
    complex = component.complex

    cost: dict[int, int] = {}
    for f in component.faceIndices:
        for v in complex.faceVertices(f):
            d = complex.vertexDegree(v)
            cost[d] = cost.get(d, 0) + 1

    d = min((c, i) for i, c in cost.items())[1]

    result: list[Location] = []

    for f in component.faceIndices:
        vs = complex.faceVertices(f)

        for k, v in enumerate(vs):
            if complex.vertexDegree(v) == d:
                result.append((f, k))

    return result


def _traverseAndRenumber(
    complex: Complex, startFace: int, startOffset: int
) -> Iterator [
    tuple[Face, Face]
]:
    vertexReIndex: dict[int, int] = {}
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


def _vertexOrder(traversal: Traversal) -> VertexList:
    faceData = traversal.result()
    vMax = max(max(f) for _, f in faceData)

    result = [0] * vMax
    for fOld, fNew in faceData:
        for v, w in zip(fOld, fNew):
            result[w - 1] = v

    return result


# -- Low level helper functions

def _cyclicPairs(indices: list[T]) -> Iterator[tuple[T, T]]:
    return zip(indices, _rotate(1, indices))


def _rotate(i: int, items: list[T]) -> list[T]:
    return items[i:] + items[:i]


# -- Main API and utility functions

def compare(base: Topology, morph: Topology, metric=None):
    for i, key in enumerate(morph.keys()):
        print("Component %d:" % i)

        morphPart = morph.get(key, [])
        basePart = base.get(key, [])

        print("  %d vertices" % len(morphPart[0][0]))
        print()

        if len(basePart) > 0:
            for j, morphInst in enumerate(morphPart):
                print("  Morph instance %d:" % j)

                for k, baseInst in enumerate(basePart):
                    print("    Base instance %d:" % k)

                    costs = [
                        metric(baseOrder, morphInst[0])
                        for baseOrder in baseInst
                    ]
                    print("      %s" % min(costs))

                    print()
            print()

        print()


def sumOfSquaresMetric(baseVerts: np.ndarray, morphVerts: np.ndarray):
    def fn(baseIndices: list[int], morphIndices: list[int]):
        basePos = baseVerts[np.array(baseIndices) - 1]
        morphPos = morphVerts[np.array(morphIndices) - 1]

        return (np.sum((basePos - morphPos)**2) / len(basePos))**0.5

    return fn


def loadAndProcessMesh(path):
    with open(path) as fp:
        data = obj.load(fp, path)

    faces = [ [ v for v in f['vertices'] ] for f in data['faces'] ]
    topo = Topology(faces)

    return topo, data


def display(*meshes):
    import numpy as np
    import polyscope as ps # type: ignore

    ps.init()

    for i in range(len(meshes)):
        mesh = meshes[i]
        ps.register_surface_mesh(
            "mesh_%02d" % i,
            np.array(mesh["vertices"]),
            [ [ v - 1 for v in f["vertices"] ] for f in mesh["faces"] ]
        )

    ps.show()


# -- Test script

if __name__ == '__main__':
    import sys
    import obj

    topoBase, dataBase = loadAndProcessMesh(sys.argv[1])

    symcounts = [
        [ [ len(vs) for vs in inst ] for inst in val ]
        for val in topoBase.values()
    ]
    print("Counts for base: %s" % " ".join(map(str, symcounts)))
    print()

    if len(sys.argv) > 2:
        topoMorph, dataMorph = loadAndProcessMesh(sys.argv[2])

        symcounts = [
            [ [ len(vs) for vs in inst ] for inst in val ]
            for val in topoMorph.values()
        ]
        print("Counts for morph: %s" % " ".join(map(str, symcounts)))
        print()

    metric = sumOfSquaresMetric(dataBase["vertices"], dataMorph["vertices"])
    compare(topoBase, topoMorph, metric)

    #display(dataBase, dataMorph)
