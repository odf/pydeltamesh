class Complex(object):
    def __init__(self, faces):
        nv = 1 + max(max(f) for f in faces)

        neighbors = [[] for _ in range(nv)]
        facesAtVert = [[] for _ in range(nv)]
        for i, f in enumerate(faces):
            for v, w in _cyclicPairs(f):
                if w not in neighbors[v]:
                    neighbors[v].append(w)
                if v not in neighbors[w]:
                    neighbors[w].append(v)
                if i not in facesAtVert[v]:
                    facesAtVert[v].append(i)

        edgeIndex = {}
        vertsAtEdge = []
        for f in faces:
            for v, w in _cyclicPairs(f):
                if edgeIndex.get((v, w)) is None:
                    edgeIndex[v, w] = edgeIndex[w, v] = len(vertsAtEdge)
                    vertsAtEdge.append([v, w])

        edgesAtFace = []
        facesAtEdge = [[] for _ in range(len(vertsAtEdge))]

        for i, f in enumerate(faces):
            edgesAtFace.append([])

            for v, w in _cyclicPairs(f):
                k = edgeIndex[v, w]
                edgesAtFace[i].append(k)
                if i not in facesAtEdge[k]:
                    facesAtEdge[k].append(i)

        self._faces = faces

        self._nrVertices = nv
        self._nrFaces = len(faces)
        self._nrEdges = len(facesAtEdge)

        self._vertexNeighbors = neighbors
        self._vertexFaces = facesAtVert
        self._edgeVertices = vertsAtEdge
        self._edgeFaces = facesAtEdge
        self._faceEdges = edgesAtFace

    @property
    def nrFaces(self):
        return self._nrFaces

    @property
    def nrEdges(self):
        return self._nrEdges

    @property
    def nrVertices(self):
        return self._nrVertices

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
    from pydeltamesh.io.obj import Face, Mesh

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
            normals=None,
            object=f.object,
            group=f.group,
            material=f.material,
            smoothinggroup=f.smoothinggroup
        ))

    return Mesh(
        vertices=verticesOut,
        normals=None,
        texverts=texVertsOut,
        faces=facesOut,
        materials=mesh.materials
    )


def subdivide(vertices, faces):
    cx = Complex(faces)
    subdFaces = subdivideTopology(cx)
    subdVerts = interpolatePerVertexData(vertices, cx)

    return subdVerts, subdFaces


def subdivideTopology(cx):
    subdFaces = []

    for i, f in enumerate(cx.faces):
        kf = i + cx.nrVertices
        ke = [k + cx.nrVertices + cx.nrFaces for k in cx.faceEdges(i)]
        for j in range(len(f)):
            subdFaces.append([f[j], ke[j], kf, ke[j - 1]])

    return subdFaces


def interpolatePerVertexData(vertexData, cx):
    import numpy as np

    nrSubdVerts = cx.nrVertices + cx.nrFaces + cx.nrEdges
    subdData = np.zeros((nrSubdVerts, vertexData.shape[1]))

    for i, f in enumerate(cx.faces):
        subdData[i + cx.nrVertices] = _centroid(vertexData, f)

    boundaryNeighbors = {}

    for i in range(cx.nrEdges):
        u, v = cx.edgeVertices(i)
        c = (vertexData[u] + vertexData[v]) / 2
        if len(cx.edgeFaces(i)) == 2:
            c += _centroid(
                subdData, (k + cx.nrVertices for k in cx.edgeFaces(i))
            )
            c /= 2
        else:
            boundaryNeighbors.setdefault(u, set()).add(v)
            boundaryNeighbors.setdefault(v, set()).add(u)

        subdData[i + cx.nrVertices + cx.nrFaces] = c

    for v in range(len(vertexData)):
        if boundaryNeighbors.get(v) is None:
            m = len(cx.vertexNeighbors(v))
            p = vertexData[v]
            r = _centroid(vertexData, cx.vertexNeighbors(v))
            f = _centroid(
                subdData, (k + cx.nrVertices for k in cx.vertexFaces(v))
            )
            subdData[v] = (f + r + (m - 2) * p) / m
        else:
            p = vertexData[v]
            r = _centroid(vertexData, boundaryNeighbors[v])
            subdData[v] = (3 * p + r) / 4

    return subdData


def _cyclicPairs(items):
    return zip(items, items[1:] + items[:1])


def _centroid(data, indices):
    count = 0
    sum = None

    for i in indices:
        sum = sum + data[i] if sum is not None else data[i]
        count += 1

    return sum / count


if __name__ == "__main__":
    import sys
    from os.path import dirname

    sys.path.append(dirname(dirname(dirname(__file__))))

    from pydeltamesh.io import obj

    with open(sys.argv[1]) as fp:
        mesh = obj.load(fp)

    meshOut = subdivideMesh(mesh)

    with open("x-subd-out.obj", "w") as fp:
        obj.save(fp, meshOut)
