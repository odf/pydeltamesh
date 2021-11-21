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

    def degree(self, v: int) -> int:
        return self._degrees[v] if 0 <= v < len(self._degrees) else 0

    def facesVertices(self, f: int) -> list[int]:
        return self._faces[f] if 0 <= f < len(self._faces) else []

    def faceNeighbors(self, f: int) -> list[Optional[Location]]:
        return self._neighbors[f] if 0 <= f < len(self._neighbors) else []



# -- API functions

def components(complex: Complex) -> Iterator[FaceList]:
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

            yield [complex.facesVertices(f) for f in queue]


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


# -- Low level helper functions

def _cyclicPairs(indices: list[T]) -> Iterator[tuple[T, T]]:
    return zip(indices, indices[1:] + indices[:1])


# -- Test script

if __name__ == '__main__':
    import sys
    import obj

    with open(sys.argv[1]) as fp:
        data = obj.load(fp)

    complex = Complex(
        [ [ v for v in f['vertices'] ] for f in data['faces'] ]
    )

    print("%d components" % len(list(components(complex))))
