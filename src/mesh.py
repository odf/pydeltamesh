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
                fv.append(len(vertices) + 1 - v if v < 0 else v)
                ft.append(len(texverts) + 1 - vt if vt < 0 else vt)
                fn.append(len(normals) + 1 - vn if vn < 0 else vn)

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


# -- Chamber based mesh implementation

def loadchambermesh(fp):
    rawmesh = loadrawmesh(fp)

    return ChamberMesh(
        rawmesh['vertices'],
        rawmesh['faces'],
        rawmesh['normals'],
        rawmesh['texverts'],
        rawmesh['materials']
    )


class ChamberMesh(object):
    def __init__(
        self, vertices, faces, normals=None, texverts=None, materials=None
    ):
        self._vertices = vertices
        self._normals = normals
        self._texverts = texverts
        self._materials = materials

        self._faceobjects = [f['object'] for f in faces]
        self._facegroups = [f['group'] for f in faces]
        self._facematerials = [f['material'] for f in faces]
        self._facesmoothinggroups = [f['smoothinggroup'] for f in faces]

        rawchambers = makechambers([f['vertices'] for f in faces])
        chamberindex = rawchambers['chamberindex']

        self._sigma = rawchambers['sigma']
        self._nrchambers = nc = self._sigma.shape[1]
        self._chambervertex = np.full(nc, -1, dtype=np.int32)
        self._chambertexvert = np.full(nc, -1, dtype=np.int32)
        self._chambernormal = np.full(nc, -1, dtype=np.int32)
        self._chamberface = rawchambers['chamberface']
        self._facechamber = np.full(len(faces), -1, dtype=np.int32)

        for i in range(self._nrchambers):
            j = self._chamberface[i]
            if self._facechamber[j] < 0:
                self._facechamber[j] = i
            f = faces[j]
            k = chamberindex[i]
            self._chambervertex[i] = f['vertices'][k]
            self._chambertexvert[i] = f['texverts'][k]
            self._chambernormal[i] = f['normals'][k]

    def write(self, fp, basename=None):
        if basename is not None and len(self._materials) > 0:
            with open('%s.mtl' % basename, 'w') as f:
                for key in self._materials:
                    f.write('newmtl %s\n' % key)
                    f.write('\n'.join(self._materials[key]))
                    f.write('\n')

        for x, y, z in self._vertices:
            fp.write('v %.8f %.8f %.8f\n' % (x, y, z))
        for x, y, z in self._normals:
            fp.write('vn %.8f %.8f %.8f\n' % (x, y, z))
        for x, y in self._texverts:
            fp.write('vt %.8f %.8f\n' % (x, y))
        for key in sorted(self._materials):
            fp.write('usemtl %s\n' % key)

        parts = {}
        for i in range(len(self._faceobjects)):
            key = (
                self._faceobjects[i],
                self._facegroups[i],
                self._facematerials[i],
                self._facesmoothinggroups[i]
            )
            parts.setdefault(key, []).append(i)

        objp, grpp, matp, smgp = None, None, None, None

        for obj, grp, mat, smg in sorted(parts):
            if obj is not None and obj != objp:
                fp.write('o %s\n' % obj)
                objp = obj
            if grp is not None and grp != grpp:
                fp.write('g %s\n' % grp)
                grpp = grp
            if mat is not None and mat != matp:
                fp.write('usemtl %s\n' % mat)
                matp = mat
            if smg is not None and smg != smgp:
                fp.write('s %s\n' % smg)
                smgp = smg

            for i in parts[obj, grp, mat, smg]:
                ch0 = ch = self._facechamber[i]
                out = ['f']
                while True:
                    out.append('%s/%s/%s' % (
                        self._chambervertex[ch],
                        self._chambertexvert[ch] or '',
                        self._chambernormal[ch] or ''
                    ))
                    ch = self._sigma[1, self._sigma[0, ch]]
                    if ch == ch0:
                        break
                fp.write('%s\n' % ' '.join(out))


def makechambers(faces):
    edges = {}
    offset = [0]

    for i in range(len(faces)):
        f = faces[i]
        n = len(f)
        offset.append(offset[-1] + n)

        for j in range(n):
            u, v = f[j], f[(j + 1) % n]

            if edges.get((u, v)) is None:
                edges[u, v] = (i, j)
            else:
                raise RuntimeError(
                    'duplicate directed edge %s -> %s' % (u, v)
                )

    nc = 2 * len(edges)

    sop = np.full((3, nc), -1, dtype=np.int32)
    ch2face = np.full(nc, -1, dtype=np.int32)
    ch2index = np.full(nc, -1, dtype=np.int32)

    for i in range(len(faces)):
        f = faces[i]
        n = len(f)

        for j in range(n):
            jnext = (j + 1) % n
            k = 2 * (offset[i] + j)
            knext = 2 * (offset[i] + jnext)

            ch2face[k] = i
            ch2face[k + 1] = i
            ch2index[k] = j
            ch2index[k + 1] = jnext

            sop[0, k] = k + 1
            sop[0, k + 1] = k
            sop[1, k + 1] = knext
            sop[1, knext] = k + 1

            opp = edges.get((f[jnext], f[j]))

            if opp is not None:
                ix, jx = opp
                kx = 2 * (offset[ix] + jx)
                sop[2, k] = kx + 1
                sop[2, k + 1] = kx

    return {
         'sigma': sop,
         'chamberface': ch2face,
         'chamberindex': ch2index
    }


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
                for f in mesh.faceIndices()
                if all(v in vertexSet for v in f)
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


def orientedEdgesInOrientedBreadthFirstOrder(
    seed: int, mesh: Mesh[Vertex]
) -> Iterator[
    tuple[tuple[int, int], tuple[int, int]]
]:
    vertexOrder = { seed: 0 }
    count = 1
    queue = [seed]

    while len(queue) > 0:
        v = queue.pop(0)

        neighbors = mesh.vertexNeighbors(v)

        for w in neighbors:
            if vertexOrder.get(w) is None:
                vertexOrder[w] = count
                count += 1
                queue.append(w)

        offset = np.argmin([vertexOrder[w] for w in neighbors])

        for w in (neighbors[offset:] + neighbors[:offset]):
            yield (v, w), (vertexOrder[v], vertexOrder[w])


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
    current = start

    while True:
        result.append(current)
        current = advance(current)
        if current is None or current == start:
            return result


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

    print("Oriented breadth-first signature at seed 0:")
    sig = [
        v
        for _, e in orientedEdgesInOrientedBreadthFirstOrder(0, mesh)
        for v in list(e)
    ]
    print("  %s" % ' '.join(map(str, sig)))

    print("Found %s poles." % len(poleVertexIndices(mesh)))

    for i in range(1):
        mesh = subdivideSmoothly(
            mesh, lambda x: x, lambda x: False, lambda _, p: p
        )

    seeds = [
        v for v in mesh.faceIndices()[0]
        if coarseningTypes(v, mesh) is not None
    ]
    print("Coarsening seeds: %s" % seeds)

    components = connectedComponents(mesh)
    print("Split into %s connected components" % len(components))

    mesh = components[0]

    #'''
    ps.init()

    ps.register_surface_mesh(
        "mesh", np.array(mesh.vertices), mesh.faceIndices()
    )

    ps.show()
    #'''
