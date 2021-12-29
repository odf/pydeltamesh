class Complex(object):
    def __init__(self, faces):
        neighbors = {}
        facesAtVert = {}
        edgeIndex = {}
        vertsAtEdge = []
        facesAtEdge = []
        edgesAtFace = []

        for i, f in enumerate(faces):
            edgesAtFace.append([])

            for v, w in _cyclicPairs(f):
                neighbors.setdefault(v, set()).add(w)
                neighbors.setdefault(w, set()).add(v)
                facesAtVert.setdefault(v, set()).add(i)

                if edgeIndex.get((v, w)) is None:
                    edgeIndex[v, w] = edgeIndex[w, v] = len(facesAtEdge)
                    vertsAtEdge.append([v, w])
                    facesAtEdge.append(set())

                facesAtEdge[edgeIndex[v, w]].add(i)
                edgesAtFace[i].append(edgeIndex[v, w])

        self._faces = faces
        self._nrFaces = len(faces)
        self._nrEdges = len(facesAtEdge)

        self._vertexNeighbors = dict(
            (v, list(s)) for v, s in neighbors.items()
        )
        self._vertexFaces = dict(
            (v, list(s)) for v, s in facesAtVert.items()
        )
        self._edgeVertices = vertsAtEdge
        self._edgeFaces = [list(s) for s in facesAtEdge]
        self._faceEdges = edgesAtFace

    @property
    def nrFaces(self):
        return self._nrFaces

    @property
    def nrEdges(self):
        return self._nrEdges

    @property
    def faces(self):
        return self._faces

    def vertexNeighbors(self, v):
        return self._vertexNeighbors[v]

    def vertexFaces(self, v):
        return self._vertexFaces[v]

    def edgeVertices(self, i):
        return self._edgeVertices[i]

    def edgeFaces(self, i):
        return self._edgeFaces[i]

    def faceEdges(self, i):
        return self._faceEdges[i]


def subdivideMesh(mesh):
    from ..io.obj import Face, Mesh

    verticesOut, faceVerticesOut = subdivide(
        mesh.vertices, [f.vertices for f in mesh.faces]
    )
    texVertsOut, faceTexVertsOut = subdivide(
        mesh.texverts, [f.texverts for f in mesh.faces]
    )

    mapping = []
    for i, f in enumerate(mesh.faces):
        for _ in range(len(f.vertices)):
            mapping.append(i)

    facesOut = []
    for i in range(len(faceVerticesOut)):
        f = mesh.faces[mapping[i]]
        facesOut.append(Face(
            vertices=faceVerticesOut[i],
            texverts=faceTexVertsOut[i],
            normals=[],
            object=f.object,
            group=f.group,
            material=f.material,
            smoothinggroup=f.smoothinggroup
        ))

    return Mesh(
        vertices=verticesOut,
        normals=[],
        texverts=texVertsOut,
        faces=facesOut,
        materials=mesh.materials
    )


def subdivide(vertices, faces):
    import numpy as np

    cx = Complex(faces)

    nrVerts = len(vertices)
    subdFaces = subdivideTopology(cx, nrVerts)

    nrSubdVerts = nrVerts + cx.nrFaces + cx.nrEdges
    subdVerts = np.zeros((nrSubdVerts, vertices.shape[1]))

    for i, f in enumerate(cx.faces):
        subdVerts[i + nrVerts] = centroid(vertices[f])

    boundaryNeighbors = {}

    for i in range(cx.nrEdges):
        u, v = cx.edgeVertices(i)
        vs = [vertices[u], vertices[v]]
        if len(cx.edgeFaces(i)) == 2:
            vs.extend(subdVerts[k + nrVerts] for k in cx.edgeFaces(i))
        else:
            boundaryNeighbors.setdefault(u, set()).add(v)
            boundaryNeighbors.setdefault(v, set()).add(u)

        subdVerts[i + nrVerts + cx.nrFaces] = centroid(vs)

    for v in range(len(vertices)):
        if boundaryNeighbors.get(v) is None:
            m = len(cx.vertexNeighbors(v))
            p = vertices[v]
            r = centroid(vertices[cx.vertexNeighbors(v)])
            f = centroid(subdVerts[[k + nrVerts for k in cx.vertexFaces(v)]])
            subdVerts[v] = (f + r + (m - 2) * p) / m
        else:
            p = vertices[v]
            r = centroid(vertices[list(boundaryNeighbors[v])])
            subdVerts[v] = (3 * p + r) / 4

    return subdVerts, subdFaces


def subdivideTopology(cx, offset):
    subdFaces = []

    for i, f in enumerate(cx.faces):
        kf = i + offset
        ke = [k + offset + cx.nrFaces for k in cx.faceEdges(i)]
        for j in range(len(f)):
            subdFaces.append([f[j], ke[j], kf, ke[j - 1]])

    return subdFaces


def _cyclicPairs(items):
    return zip(items, items[1:] + items[:1])


def centroid(ps):
    import numpy as np
    m = len(ps)
    return np.dot([1.0/m] * m, ps)


if __name__ == "__main__":
    import sys
    from ..io import obj

    with open(sys.argv[1]) as fp:
        mesh = obj.load(fp)

    meshOut = subdivideMesh(mesh)

    with open("x-subd-out.obj", "w") as fp:
        obj.save(fp, meshOut)
