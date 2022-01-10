import numpy as _np


class Complex(object):
    def __init__(self, faces):
        nv = 1 + max(max(f) for f in faces)

        isAllQuads = True
        neighbors = [[] for _ in range(nv)]
        facesAtVert = [[] for _ in range(nv)]
        edgesAtFace = [[] for _ in range(len(faces))]
        edgeIndex = {}
        vertsAtEdge = []
        facesAtEdge = []

        for i, f in enumerate(faces):
            if len(f) != 4:
                isAllQuads = False

            for v, w in _cyclicPairs(f):
                if w not in neighbors[v]:
                    neighbors[v].append(w)
                if v not in neighbors[w]:
                    neighbors[w].append(v)
                if i not in facesAtVert[v]:
                    facesAtVert[v].append(i)

                if edgeIndex.get((v, w)) is None:
                    edgeIndex[v, w] = edgeIndex[w, v] = len(vertsAtEdge)
                    vertsAtEdge.append([v, w])
                    facesAtEdge.append([])

                k = edgeIndex[v, w]
                edgesAtFace[i].append(k)
                if i not in facesAtEdge[k]:
                    facesAtEdge[k].append(i)

        boundaryNeighbors = {}
        for i in range(len(facesAtEdge)):
            if len(facesAtEdge[i]) != 2:
                u, v = vertsAtEdge[i]

                if not v in boundaryNeighbors.setdefault(u, []):
                    boundaryNeighbors[u].append(v)
                if not u in boundaryNeighbors.setdefault(v, []):
                    boundaryNeighbors[v].append(u)

        self._faces = faces
        self._isAllQuads = isAllQuads

        self._nrVertices = nv
        self._nrFaces = len(faces)
        self._nrEdges = len(facesAtEdge)

        self._vertexNeighbors = neighbors
        self._boundaryNeighbors = boundaryNeighbors
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

    @property
    def isAllQuads(self):
        return self._isAllQuads

    def vertexNeighbors(self, v):
        return self._vertexNeighbors[v]

    def boundaryNeighbors(self, v):
        return self._boundaryNeighbors.get(v)

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

    skipUVs = len(mesh.texverts) == 0

    verticesOut, faceVerticesOut = subdivide(
        mesh.vertices, [f.vertices for f in mesh.faces]
    )

    if skipUVs:
        texVertsOut = []
    else:
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
            texverts=([] if skipUVs else faceTexVertsOut[i]),
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
        ke = cx.faceEdges(i)
        t = cx.nrVertices + cx.nrFaces

        if len(f) == 4:
            subdFaces.extend([
                [f[0], ke[0] + t, kf, ke[3] + t],
                [ke[0] + t, f[1], ke[1] + t, kf],
                [kf, ke[1] + t, f[2], ke[2] + t],
                [ke[3] + t, kf, ke[2] + t, f[3]]
            ])
        else:
            for j in range(len(f)):
                subdFaces.append([f[j], ke[j] + t, kf, ke[j - 1] + t])

    return subdFaces


def interpolatePerVertexData(vertexData, cx):
    fPoints = _faceCenters(vertexData, cx)
    vPoints = _adjustedVertices(vertexData, fPoints, cx)
    ePoints = _edgePoints(vertexData, fPoints, cx)

    return _np.vstack([vPoints, fPoints, ePoints])


def adjustVertexData(vertexData, cx):
    fPoints = _faceCenters(vertexData, cx)
    return _adjustedVertices(vertexData, fPoints, cx)


def _faceCenters(vertices, cx):
    if cx.isAllQuads:
        vs = vertices
        fs = _np.array(cx.faces)

        return (vs[fs[:, 0]] + vs[fs[:, 1]] + vs[fs[:, 2]] + vs[fs[:, 3]]) / 4
    else:
        centers = _np.zeros((len(cx.faces), vertices.shape[1]))

        facesByDegree = {}
        for i in range(len(cx.faces)):
            facesByDegree.setdefault(len(cx.faces[i]), []).append(i)

        for d in facesByDegree:
            idcs = facesByDegree[d]
            vs = [cx.faces[i] for i in idcs]
            centers[idcs] = _np.sum(vertices[vs], axis=1) / d

        return centers


def _edgePoints(vertexData, fPoints, cx):
    vs = [cx.edgeVertices(i) for i in range(cx.nrEdges)]
    output = _np.sum(vertexData[vs], axis=1) / 2.0

    interior = [i for i in range(cx.nrEdges) if len(cx.edgeFaces(i)) == 2]
    ws = [cx.edgeFaces(i) for i in interior]
    output[interior] += _np.sum(fPoints[ws], axis=1) / 2.0
    output[interior] /= 2.0

    return output


def _adjustedVertices(vertexData, fPoints, cx):
    output = vertexData.copy()

    unchanged = []
    boundary = []
    interiorByDegree = {}

    for v in range(cx.nrVertices):
        m = len(cx.vertexNeighbors(v))

        if m > 0 and cx.boundaryNeighbors(v) is None:
            interiorByDegree.setdefault(m, []).append(v)
        elif m < 3:
            unchanged.append(v)
        else:
            boundary.append(v)

    ps = vertexData[[cx.boundaryNeighbors(v) for v in boundary]]
    output[boundary] = (
        0.75 * vertexData[boundary] + 0.125 * _np.sum(ps, axis=1)
    )

    for d in interiorByDegree:
        idcs = interiorByDegree[d]

        p = vertexData[idcs]

        vs = _np.array([cx.vertexNeighbors(v) for v in idcs])
        r = _np.sum(vertexData[vs], axis=1) / (d**2)

        ws = _np.array([cx.vertexFaces(v) for v in idcs])
        f = _np.sum(fPoints[ws], axis=1) / (d**2)

        output[idcs] = f + r + (d - 2) / d * p

    return output


def _cyclicPairs(items):
    return zip(items, items[1:] + items[:1])


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
