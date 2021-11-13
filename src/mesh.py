import numpy as np


# -- Code for reading an .obj file

def loadrawmesh(fp):
    vertices = []
    normals = []
    texverts = []
    faces = []
    materials = {}

    obj = None
    group = None
    material = None
    smoothinggroup = None

    for rawline in fp.readlines():
        line = rawline.strip()

        if len(line) == 0 or line[0] == '#':
            continue

        fields = line.split()
        cmd = fields[0]
        pars = fields[1:]

        if cmd == 'mtllib':
            # TODO if path is relative, start at .obj file path
            with open(pars[0]) as f:
                loadmaterials(f, materials)
        elif cmd == 'v':
            vertices.append([float(pars[0]), float(pars[1]), float(pars[2])])
        elif cmd == 'vt':
            texverts.append([float(pars[0]), float(pars[1])])
        elif cmd == 'vn':
            normals.append([float(pars[0]), float(pars[1]), float(pars[2])])
        elif cmd == 'o':
            obj = pars[0]
        elif cmd == 'g':
            group = pars[0]
        elif cmd == 'usemtl':
            material = pars[0]
        elif cmd == 's':
            smoothinggroup = pars[0]
        elif cmd == 'f':
            fv = []
            ft = []
            fn = []

            for i in range(len(pars)):
                v, vt, vn = map(int, (pars[i] + '/0/0').split('/')[:3])
                fv.append(v + (len(vertices) + 1 if v < 0 else 0))
                ft.append(vt + (len(texverts) + 1 if vt < 0 else 0))
                fn.append(vn + (len(normals) + 1 if vn < 0 else 0))

            faces.append({
                'vertices': fv,
                'texverts': ft,
                'normals': fn,
                'object': obj,
                'group': group,
                'material': material,
                'smoothinggroup': smoothinggroup
            })

    return {
        'vertices': np.array(vertices),
        'normals': np.array(normals),
        'texverts': np.array(texverts),
        'faces': faces,
        'materials': materials
    }


def loadmaterials(fp, target):
    material = None

    for rawline in fp.readlines():
        line = rawline.strip()

        if len(line) == 0 or line[0] == '#':
            continue

        fields = line.split()

        if fields[0] == 'newmtl':
            material = fields[1]
            target.setdefault(material, [])
        else:
            target[material].append(line)


# -- Half-edge based mesh implementation

from typing import Callable, Generic, Iterator, Optional, TypeVar

T = TypeVar('T')
Vertex = TypeVar('Vertex')
OrientedEdge = tuple[int, int]
Point = np.ndarray


class Mesh(Generic[Vertex]):
    def __init__(
        self,
        vertices: list[Vertex],
        atVertex: list[OrientedEdge],
        alongFace: list[OrientedEdge],
        alongBoundaryComponent: list[OrientedEdge],
        toFace: dict[OrientedEdge, int],
        toBoundaryComponent: dict[OrientedEdge, int],
        nextEdge: dict[OrientedEdge, OrientedEdge]
    ):
        self._vertices = vertices
        self._atVertex = atVertex
        self._alongFace = alongFace
        self._alongBoundaryComponent = alongBoundaryComponent
        self._toFace = toFace
        self._toBoundaryComponent = toBoundaryComponent
        self._next = nextEdge

    @property
    def nrVertices(self) -> int:
        return len(self._vertices)

    @property
    def nrFaces(self) -> int:
        return len(self._alongFace)

    @property
    def vertices(self) -> list[Vertex]:
        return self._vertices[:]

    def vertex(self, index: int) -> Optional[Vertex]:
        if 0 <= index < self.nrVertices:
            return self._vertices[index]
        else:
            return None

    def verticesInFace(self, e0: OrientedEdge) -> list[int]:
        return [e[0] for e in traceCycle(e0, self._next.get)]

    def faceIndices(self) -> list[list[int]]:
        return [
            canonicalCircular(self.verticesInFace(e))
            for e in self._alongFace
        ]

    def faceVertices(self) -> list[list[Vertex]]:
        return [
            [ self._vertices[i] for i in idcs ]
            for idcs in self.faceIndices()
        ]

    def boundaryIndices(self) -> list[list[int]]:
        return [
            canonicalCircular(self.verticesInFace(e))
            for e in self._alongBoundaryComponent
        ]

    def boundaryVertices(self) -> list[list[Vertex]]:
        return [
            [ self._vertices[i] for i in idcs ]
            for idcs in self.boundaryIndices()
        ]

    def edgeIndices(self) -> list[OrientedEdge]:
        return [(s, t) for (s, t) in self._next if s < t]

    def edgeVertices(self) -> list[tuple[Vertex, Vertex]]:
        return [
            (self._vertices[s], self._vertices[t])
            for (s, t) in self.edgeIndices()
        ]

    def _nextAroundVertex(self, e: OrientedEdge) -> OrientedEdge:
        return self._next[opposite(e)]

    def _vertexNeighbors(self, e: OrientedEdge) -> list[int]:
        return [
            t for (s, t) in traceCycle(e, self._nextAroundVertex)
        ][::-1]

    def vertexNeighbors(self, v: int, w: Optional[int] = None) -> list[int]:
        if w is None:
            return self._vertexNeighbors(self._atVertex[v])
        else:
            return self._vertexNeighbors((v, w))

    def vertexDegree(self, v: int) -> int:
        return len(self.vertexNeighbors(v))

    def neighborIndices(self) -> list[list[int]]:
        return [
            canonicalCircular(self._vertexNeighbors(e))
            for e in self._atVertex
        ]

    def neighborVertices(self) -> list[list[Vertex]]:
        return [
            [ self._vertices[i] for i in idcs ]
            for idcs in self.neighborIndices()
        ]


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


def opposite(e: OrientedEdge) -> OrientedEdge:
    return (e[1], e[0])


def fromOrientedFaces(
    vertexData: list[Vertex],
    faceLists: list[list[int]]
) -> Mesh[Vertex]:
    definedVertexSet = set(range(len(vertexData)))
    referencedVertexSet = set(concat(faceLists))

    if not definedVertexSet.issuperset(referencedVertexSet):
        raise ValueError('an undefined vertex appears in a face')
    elif not referencedVertexSet.issuperset(definedVertexSet):
        raise ValueError('one of the vertices does not appear in any face')
    elif any(len(f) < 2 for f in faceLists):
        raise ValueError('there is a face with fewer than 2 vertices')
    elif any(hasDuplicates(f) for f in faceLists):
        raise ValueError('a vertex appears more than once in the same face')

    orientedEdgeLists = [cyclicPairs(face) for face in faceLists]
    orientedEdges = concat(orientedEdgeLists)
    orientedEdgeSet = set(orientedEdges)

    if len(orientedEdgeSet) < len(orientedEdges):
        raise ValueError('an oriented edge appears more than once')

    boundaryEdges = [
        e for e in orientedEdges if opposite(e) not in orientedEdgeSet
    ]

    if hasDuplicates([a for a, b in boundaryEdges]):
        raise ValueError('a vertex appears more than once in a boundary')

    verticesAtBoundary = [e[0] for e in boundaryEdges]
    nextOnBoundary = dict(opposite(e) for e in boundaryEdges)

    boundaryLists = [
        cyclicPairs(canonicalCircular(c))
        for c in extractCycles(verticesAtBoundary, nextOnBoundary.get)
    ]

    atVertex = sorted(dict((s, (s, t)) for (s, t) in orientedEdges).values())

    toFace = dict(
        (e, i) for i in range(len(faceLists)) for e in orientedEdgeLists[i]
    )
    alongFace = sorted(dict((toFace[e], e) for e in toFace).values())

    toBoundaryComponent = dict(
        (e, i) for i in range(len(boundaryLists)) for e in boundaryLists[i]
    )
    alongBoundaryComponent = sorted(dict(
        (toBoundaryComponent[e], e) for e in toBoundaryComponent
    ).values())

    nextEdge = dict(
        e
        for cycles in [orientedEdgeLists, boundaryLists]
        for cycle in cycles
        for e in cyclicPairs(cycle)
    )

    return Mesh(
        vertexData,
        atVertex,
        alongFace,
        alongBoundaryComponent,
        toFace,
        toBoundaryComponent,
        nextEdge
    )


def triangulate(mesh: Mesh[Vertex]) -> Mesh[Vertex]:
    return fromOrientedFaces(
        mesh.vertices,
        [
            [f[0], f[i], f[i + 1]]
            for f in mesh.faceIndices()
            for i in range(1, len(f) - 1)
        ]
    )


def combine(meshes: list[Mesh[Vertex]]) -> Mesh[Vertex]:
    vertices = concat([m.vertices for m in meshes])

    faces: list[list[int]] = []
    offset = 0

    for m in meshes:
        for f in m.faceIndices():
            faces.append([i + offset for i in f])
        offset += m.nrVertices

    return fromOrientedFaces(vertices, faces)


def connectedComponents(mesh: Mesh[Vertex]) -> list[Mesh[Vertex]]:
    faceIndices = mesh.faceIndices()
    output: list[Mesh[Vertex]] = []
    seen: set[int] = set()

    for v0 in range(mesh.nrVertices):
        if not v0 in seen:
            seen.add(v0)
            queue = [v0]
            k = 0

            while k < len(queue):
                v = queue[k]
                k += 1

                for w in mesh.vertexNeighbors(v):
                    if not w in seen:
                        seen.add(w)
                        queue.append(w)

            vertexSet = set(queue)
            mapping = dict((v, i) for i, v in enumerate(queue))
            vertsOut = filterOptional([mesh.vertex(v) for v in queue])
            facesOut = [
                [mapping[v] for v in f]
                for f in faceIndices
                if f[0] in vertexSet
            ]
            output.append(fromOrientedFaces(vertsOut, facesOut))

    return output


def mapVertices(mesh: Mesh[Vertex], fn: Callable[[Vertex], T]) -> Mesh[T]:
    return fromOrientedFaces(
        [fn(v) for v in mesh.vertices],
        mesh.faceIndices()
    )


def subdivide(
    mesh: Mesh[Vertex],
    composeFn: Callable[[list[Vertex]], Vertex]
) -> Mesh[Vertex]:
    allEdges = mesh.edgeIndices()
    nrVerts = mesh.nrVertices
    nrEdges = len(allEdges)

    midPointIndex = {}
    for i in range(nrEdges):
        (u, v) = allEdges[i]
        midPointIndex[u, v] = midPointIndex[v, u] = i + nrVerts

    sourceLists = (
        [ [ v ] for v in mesh.vertices ] +
        [ [u, v] for (u, v) in mesh.edgeVertices() ] +
        mesh.faceVertices()
    )

    vertsOut = [composeFn(vs) for vs in sourceLists]

    facesOut = [
        [
            v,
            midPointIndex[v, w],
            k + nrVerts + nrEdges,
            midPointIndex[u, v]
        ]
        for k, f in enumerate(mesh.faceIndices())
        for (u, v, w) in cyclicTriples(f)
    ]

    return fromOrientedFaces(vertsOut, facesOut)


def subdivideSmoothly(
    mesh: Mesh[Vertex],
    vertexPosition: Callable[[Vertex], Point],
    isFixed: Callable[[Vertex], bool],
    toOutputVertex: Callable[[list[Vertex], Point], Vertex]
) -> Mesh[Vertex]:
    nrVertices = mesh.nrVertices
    nrEdges = len(mesh.edgeIndices())
    nrFaces = mesh.nrFaces

    def makeOutputVertex(indices, pos):
        return toOutputVertex([mesh.vertex(i) for i in indices], pos)

    meshSub = subdivide(mapVertices(mesh, vertexPosition), centroid)
    subFaceIndices = meshSub.faceIndices()

    facePoints = [
        makeOutputVertex(idcs, meshSub.vertex(k + nrVertices + nrEdges))
        for k, idcs in enumerate(subFaceIndices)
        if k < nrFaces
    ]

    boundaryIndicesSub = set(concat(meshSub.boundaryIndices()))
    neighborsSub = meshSub.neighborIndices()

    def edgePointPosition(index):
        k = index + nrVertices

        if k in boundaryIndicesSub:
            return meshSub.vertex(k)
        else:
            return centroid([meshSub.vertex(i) for i in neighborsSub[k]])

    edgePointPositions = [edgePointPosition(i) for i in range(nrEdges)]

    edgePoints = [
        makeOutputVertex([u, v], edgePointPosition(i))
        for i, (u, v) in enumerate(mesh.edgeIndices())
    ]

    def vertexPointPosition(index):
        posIn = meshSub.vertex(index)
        neighbors = neighborsSub[index]
        nrNeighbors = len(neighbors)

        if isFixed(mesh.vertex(index)):
            return posIn
        elif index in boundaryIndicesSub:
            a = centroid([
                meshSub.vertex(k)
                for k in neighbors
                if k in boundaryIndicesSub
            ])
            return (posIn + a) / 2.0
        else:
            a = centroid([ meshSub.vertex(k) for k in neighbors ])
            b = centroid([
                edgePointPositions[k - nrVertices] for k in neighbors
            ])
            return ((nrNeighbors - 3) * posIn + a + 2 * b) / nrNeighbors

    vertexPoints = [
        makeOutputVertex([i], vertexPointPosition(i))
        for i in range(nrVertices)
    ]

    return fromOrientedFaces(
        vertexPoints + edgePoints + facePoints,
        subFaceIndices
    )


def poleVertexIndices(mesh: Mesh[Vertex]) -> list[int]:
    boundary = set(concat(mesh.boundaryIndices()))
    degree = lambda v: len(mesh.vertexNeighbors(v))

    return [
        v for v in range(mesh.nrVertices)
        if degree(v) not in ([2, 3] if v in boundary else [4])
    ]


def coarseningTypes(
    seed: int, mesh: Mesh[Vertex]
) -> Optional[
    tuple[set[int], set[int], set[int]]
]:
    boundary = set(concat(mesh.boundaryIndices()))

    vertices: set[int] = set()
    edgeCenters: set[int] = set()
    faceCenters: set[int] = set()

    vertices.add(seed)
    queue = [seed]

    while len(queue):
        v = queue.pop()

        for w in mesh.vertexNeighbors(v):
            if w in vertices or w in faceCenters:
                return None

            edgeCenters.add(w)
            neighbors = mesh.vertexNeighbors(w, v)

            if w in boundary:
                if v not in boundary or len(neighbors) != 3:
                    return None
            elif len(neighbors) != 4:
                    return None

            for u in (neighbors[0], neighbors[-2]):
                if u in edgeCenters:
                    return None

                if u in boundary:
                    if w not in boundary or u in faceCenters:
                        return None
                    elif u not in vertices:
                        vertices.add(u)
                        queue.append(u)
                elif u in vertices:
                    return None
                else:
                    faceCenters.add(u)

            if w not in boundary:
                u = neighbors[1]
                if u in edgeCenters or u in faceCenters:
                    return None
                elif u not in vertices:
                    vertices.add(u)
                    queue.append(u)

    return (vertices, edgeCenters, faceCenters)


def invariant(mesh: Mesh[Vertex]) -> list[int]:
    iter = BufferedIterator[tuple[tuple[int, int], tuple[int, int]]]
    starts = optimalStartEdges(mesh)
    edges: iter = BufferedIterator(
        orientedEdgesInOrientedBreadthFirstOrder(starts[0], mesh)
    )
    return [ v for _, e in edges.result() for v in list(e) ]


def symmetries(mesh: Mesh[Vertex]) -> list[list[int]]:
    starts = optimalStartEdges(mesh)
    edges0 = BufferedIterator(
        orientedEdgesInOrientedBreadthFirstOrder(starts[0], mesh)
    ).result()

    result = []

    for s in starts:
        edges = BufferedIterator(
            orientedEdgesInOrientedBreadthFirstOrder(s, mesh)
        ).result()
        mapping = [-1] * mesh.nrVertices

        for i in range(len(edges)):
            e0 = edges0[i][0]
            e = edges[i][0]
            mapping[e0[0]] = e[0]
            mapping[e0[1]] = e[1]

        result.append(mapping)

    return result


def optimalStartEdges(mesh: Mesh[Vertex]) -> list[tuple[int, int]]:
    iter = BufferedIterator[tuple[tuple[int, int], tuple[int, int]]]
    best: Optional[iter] = None
    result: list[tuple[int, int]] = []

    for (v, w) in startEdgeCandidates(mesh):
        candidate: iter = BufferedIterator(
            orientedEdgesInOrientedBreadthFirstOrder((v, w), mesh)
        )
        if best is None:
            best = candidate
            result = [(v, w)]
        else:
            i = 0
            while True:
                a = best.get(i)
                b = candidate.get(i)
                if a is None:
                    assert(b is None)
                    result.append((v, w))
                    break
                elif a[1] < b[1]:
                    break
                elif b[1] < a[1]:
                    best = candidate
                    result = [(v, w)]
                    break
                else:
                    i += 1

    return result


def startEdgeCandidates(mesh: Mesh[Vertex]) -> list[tuple[int, int]]:
    degree = [mesh.vertexDegree(v) for v in range(mesh.nrVertices)]
    bestValue: Optional[tuple[int, int]] = None
    result: list[tuple[int, int]] = []

    for v in range(mesh.nrVertices):
        for w in mesh.vertexNeighbors(v):
            value = (degree[v], degree[w])
            if bestValue is None or value > bestValue:
                bestValue = value
                result = []

            if value == bestValue:
                result.append((v, w))

    return result


def orientedEdgesInOrientedBreadthFirstOrder(
    seed: tuple[int, int], mesh: Mesh[Vertex]
) -> Iterator[
    tuple[tuple[int, int], tuple[int, int]]
]:
    u, v = seed
    vertexOrder = { u: 0 }
    count = 1
    queue = [(u, v)]

    while len(queue) > 0:
        u, v = queue.pop(0)

        neighbors = rotate(-1, mesh.vertexNeighbors(u, v))

        for w in neighbors:
            if vertexOrder.get(w) is None:
                vertexOrder[w] = count
                count += 1
                queue.append((w, u))

        offset = int(np.argmin([vertexOrder[w] for w in neighbors]))

        for w in rotate(offset, neighbors):
            yield (u, w), (vertexOrder[u], vertexOrder[w])


# -- Various helper functions, mostly for lists

def extractCycles(
    items: list[T],
    advance: Callable[[T], Optional[T]]
) -> list[list[T]]:

    cycles: list[list[T]] = []
    seen: set[T] = set()

    for item in items:
        if item not in seen:
            cycle = traceCycle(item, advance)
            cycles.append(cycle)
            seen.update(cycle)

    return cycles


def traceCycle(start: T, advance: Callable[[T], Optional[T]]) -> list[T]:
    result: list[T] = []
    current: Optional[T] = start

    while True:
        if current is not None: # satisfy type checker
            result.append(current)
            current = advance(current)
        if current is None or current == start:
            return result


def rotate(i: int, items: list[T]) -> list[T]:
    return items[i:] + items[:i]


def cyclicPairs(indices: list[T]) -> list[tuple[T, T]]:
    return list(zip(indices, indices[1:] + indices[:1]))


def cyclicTriples(indices: list[T]) -> list[tuple[T, T, T]]:
    return list(zip(
        indices,
        indices[1:] + indices[:1],
        indices[2:] + indices[:2]
    ))


def canonicalCircular(items: list[T]) -> list[T]:
    best = items

    for i in range(1, len(items)):
        tmp = items[i:] + items[:i]
        if tmp < best:
            best = tmp

    return best


def hasDuplicates(items):
    tmp = sorted(items)
    return any(a == b for a, b in zip(tmp, tmp[1:]))


def concat(lists: list[list[T]]) -> list[T]:
    return [x for xs in lists for x in xs]


def centroid(points: list[Point]) -> Point:
    return np.average(points, axis=0)


def filterOptional(items: list[Optional[T]]) -> list[T]:
    return [x for x in items if x is not None]


# -- Test script

if __name__ == '__main__':
    import sys
    import polyscope as ps # type: ignore

    with open(sys.argv[1]) as fp:
        rawmesh = loadrawmesh(fp)

    mesh = fromOrientedFaces(
        rawmesh['vertices'],
        [ [ v - 1 for v in f['vertices'] ] for f in rawmesh['faces'] ]
    )

    print(
        "Mesh has %s vertices and %s faces" % (mesh.nrVertices, mesh.nrFaces)
    )
    print("Found %s poles." % len(poleVertexIndices(mesh)))

    seeds = [
        v for v in mesh.faceIndices()[0]
        if coarseningTypes(v, mesh) is not None
    ]
    print("Coarsening seeds: %s" % seeds)

    components = connectedComponents(mesh)
    print("Split into %s connected components" % len(components))

    mesh = components[0]

    print("Optimal start edges: %s" % optimalStartEdges(mesh))

    invar = invariant(mesh)
    if len(invar) > 400:
        print("Invariant: %s..." % invar[:400])
    else:
        print("Invariant: %s" % invar)

    syms = symmetries(mesh)
    print("Symmetries:")
    for s in syms:
        if len(s) > 16:
            print("  %s..." % s[:16])
        else:
            print("  %s" % s)

    '''
    ps.init()

    ps.register_surface_mesh(
        "mesh", np.array(mesh.vertices), mesh.faceIndices()
    )

    ps.show()
    #'''
