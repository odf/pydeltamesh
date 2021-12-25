def subdivideMesh(mesh):
    from obj import Face, Mesh

    verticesOut, faceVerticesOut = _subdTopology(
        mesh.vertices, [f.vertices for f in mesh.faces]
    )
    texVertsOut, faceTexVertsOut = _subdTopology(
        mesh.texverts, [f.texverts for f in mesh.faces]
    )
    normalsOut, faceNormalsOut = _subdTopology(
        mesh.normals, [f.normals for f in mesh.faces]
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
            normals=faceNormalsOut[i],
            object=f.object,
            group=f.group,
            material=f.material,
            smoothinggroup=f.smoothinggroup
        ))

    return Mesh(
        vertices=verticesOut,
        normals=normalsOut,
        texverts=texVertsOut,
        faces=facesOut,
        materials=mesh.materials
    )


def _subdTopology(vertices, faces):
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
        subdVerts[i + facePointOffset] = centroid([vertices[v] for v in f])

    for (u, v), i in edgeIndex.items():
        vs = [subdVerts[k + facePointOffset] for k in facesAtEdge[i]]
        vs.extend([vertices[u], vertices[v]])
        subdVerts[i + edgePointOffset] = centroid(vs)

    for v in range(len(vertices)):
        m = len(neighbors[v])
        p = vertices[v]
        r = centroid([vertices[w] for w in neighbors[v]])
        f = centroid([subdVerts[k + facePointOffset] for k in facesAtVert[v]])

        subdVerts[v] = (f + r + (m - 2) * p) / m

    return subdVerts, subdFaces


def _cyclicPairs(items):
    return zip(items, items[1:] + items[:1])


def centroid(ps):
    import numpy as np
    return np.sum(ps, axis=0) / len(ps)


if __name__ == "__main__":
    import sys
    import obj

    with open(sys.argv[1]) as fp:
        mesh = obj.load(fp)

    meshOut = subdivideMesh(mesh)

    with open("x-subd-out.obj", "w") as fp:
        obj.save(fp, meshOut)
