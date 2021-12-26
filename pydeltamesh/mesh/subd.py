def subdivideMesh(mesh):
    from ..io.obj import Face, Mesh

    verticesOut, faceVerticesOut = subdivideTopology(
        mesh.vertices, [f.vertices for f in mesh.faces]
    )
    texVertsOut, faceTexVertsOut = subdivideTopology(
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


def subdivideTopology(vertices, faces):
    import numpy as np

    facePointOffset = len(vertices)
    edgePointOffset = facePointOffset + len(faces)

    neighbors = [set() for _ in range(len(vertices))]
    facesAtVert = [[] for _ in range(len(vertices))]
    edgeIndex = {}
    facesAtEdge = []

    for i, f in enumerate(faces):
        for v, w in _cyclicPairs(f):
            neighbors[v].add(w)
            neighbors[w].add(v)
            facesAtVert[v].append(i)

            if edgeIndex.get((v, w)) is None:
                edgeIndex[v, w] = edgeIndex[w, v] = len(facesAtEdge)
                facesAtEdge.append([])
            facesAtEdge[edgeIndex[v, w]].append(i)

    subdFaces = []

    for i, f in enumerate(faces):
        kf = i + facePointOffset
        ke = [
            edgeIndex[v, w] + edgePointOffset
            for v, w in _cyclicPairs(f)
        ]
        for j in range(len(f)):
            subdFaces.append([f[j], ke[j], kf, ke[j - 1]])

    nrSubdVerts = edgePointOffset + len(facesAtEdge)
    subdVerts = np.zeros((nrSubdVerts, vertices.shape[1]))
    subdVerts[: len(vertices)] = vertices

    for i, f in enumerate(faces):
        subdVerts[i + facePointOffset] = centroid(vertices[f])

    boundaryNeighbors = {}

    for (u, v), i in edgeIndex.items():
        if u < v:
            vs = [vertices[u], vertices[v]]
            if len(facesAtEdge[i]) == 2:
                vs.extend(
                    subdVerts[k + facePointOffset] for k in facesAtEdge[i]
                )
            else:
                boundaryNeighbors.setdefault(u, set()).add(v)
                boundaryNeighbors.setdefault(v, set()).add(u)

            subdVerts[i + edgePointOffset] = centroid(vs)

    for v in range(len(vertices)):
        if boundaryNeighbors.get(v) is None:
            m = len(neighbors[v])
            p = vertices[v]
            r = centroid([vertices[w] for w in neighbors[v]])
            f = centroid(
                [subdVerts[k + facePointOffset] for k in facesAtVert[v]]
            )
            subdVerts[v] = (f + r + (m - 2) * p) / m
        else:
            p = vertices[v]
            r = centroid([vertices[w] for w in boundaryNeighbors[v]])
            subdVerts[v] = (3 * p + r) / 4

    return subdVerts, subdFaces


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
