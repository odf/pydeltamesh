from typing import Iterator, Optional, TypeVar

T = TypeVar('T')
FaceList = list[list[int]]
OrientedEdge = tuple[int, int]
Location = tuple[int, int]


# -- Data type(s)

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

    def facesVertices(self, f: int) -> list[int]:
        return self._faces[f]

    def faceNeighbors(self, f: int) -> list[Optional[Location]]:
        return self._neighbors[f]



# -- API functions

def components(complex: Complex) -> Iterator[list[int]]:
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

            yield queue


# -- High level helper functions

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


def _startCandidates(
    complex: Complex, faceSelection: Optional[list[int]] = None
) -> list[
    tuple[int, int]
]:
    faces = faceSelection or range(complex.nrFaces)
    best = 0
    result: list[tuple[int, int]] = []

    for f in faces:
        vs = complex.facesVertices(f)

        for k, v in enumerate(vs):
            d = complex.vertexDegree(v)

            if d > best:
                best = d
                result = []

            if d == best:
                result.append((f, k))

    return result


def _traverseAndRenumber(
    complex: Complex, startFace: int, startOffset: int
) -> Iterator [
    tuple[list[int], list[int]]
]:
    vertexOrder: dict[int, int] = {}
    nextVertex = 1
    queue = [(startFace, startOffset)]
    seen = set([startFace])

    while len(queue) > 0:
        f, k = queue.pop(0)
        vs = _rotate(k, complex.facesVertices(f))
        nbs = _rotate(k, complex.faceNeighbors(f))

        for v in vs:
            if not v in vertexOrder:
                vertexOrder[v] = nextVertex
                nextVertex += 1

        for nb in nbs:
            if nb is not None:
                fn, kn = nb
                if not fn in seen:
                    queue.append((fn, kn))
                    seen.add(fn)

        yield vs, [vertexOrder[v] for v in vs]


# -- Low level helper functions

def _cyclicPairs(indices: list[T]) -> Iterator[tuple[T, T]]:
    return zip(indices, _rotate(1, indices))


def _rotate(i: int, items: list[T]) -> list[T]:
    return items[i:] + items[:i]


# -- Test script

if __name__ == '__main__':
    import sys
    import obj

    with open(sys.argv[1]) as fp:
        data = obj.load(fp)

    complex = Complex(
        [ [ v for v in f['vertices'] ] for f in data['faces'] ]
    )

    comps = list(components(complex))
    print("%d components" % len(comps))

    starts = list(_startCandidates(complex, comps[0]))
    print("Found %d start candidates" % len(starts))

    print(len(list(_traverseAndRenumber(complex, starts[0][0], starts[0][1]))))
