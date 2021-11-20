from typing import Iterator, Optional, TypeVar

T = TypeVar('T')
OrientedEdge = tuple[int, int]
Location = tuple[int, int]


class Complex(object):
    def __init__(self, faces: list[list[int]]):
        self._faces = faces
        self._neighbors = faceNeighbors(faces)
        self._degrees = vertexDegrees(faces)
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

    @property
    def faceVertices(self) -> Iterator[list[int]]:
        return iter(self._faces)

    def degree(self, v: int) -> int:
        return self._degrees[v] if 0 <= v < len(self._degrees) else 0

    def faceNeighbors(self, f: int) -> list[Optional[Location]]:
        return self._neighbors[f] if 0 <= f < len(self._neighbors) else []



def faceNeighbors(faces: list[list[int]]) -> list[list[Optional[Location]]]:
    edgeLocation: dict[OrientedEdge, Location] = {}

    for i, face in enumerate(faces):
        for k, (v, w) in enumerate(cyclicPairs(face)):
            if edgeLocation.get((v, w)) is None:
                edgeLocation[(v, w)] = (i, k)
            else:
                raise ValueError('an oriented edge appears more than once')

    return [
        [edgeLocation.get((w, v)) for v, w in cyclicPairs(face)]
        for face in faces
    ]


def vertexDegrees(faces: list[list[int]]) -> list[int]:
    maxVertex = max(max(f) for f in faces)
    degree = [0] * (maxVertex + 1)

    for face in faces:
        for v in face:
            degree[v] += 1

    return degree


def cyclicPairs(indices: list[T]) -> list[tuple[T, T]]:
    return list(zip(indices, indices[1:] + indices[:1]))


# -- Test script

if __name__ == '__main__':
    import sys
    import obj

    with open(sys.argv[1]) as fp:
        data = obj.load(fp)

    complex = Complex(
        [ [ v for v in f['vertices'] ] for f in data['faces'] ]
    )
