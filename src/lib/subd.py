def subdivideMesh(mesh):
    from obj import Face

    faceVerticesOut = _subdTopology(
        mesh.vertices, [f.vertices for f in mesh.faces]
    )
    faceTexVertsOut = _subdTopology(
        mesh.texverts, [f.texverts for f in mesh.faces]
    )
    faceNormalsOut = _subdTopology(
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

    print(facesOut)


def _subdTopology(vertices, faces):
    facePointOffset = len(vertices)
    edgePointOffset = facePointOffset + len(faces)

    nextEdgeIndex = 0
    edgeIndex = {}

    for f in faces:
        for v, w in _cyclicPairs(f):
            if edgeIndex.get((v, w)) is None:
                edgeIndex[v, w] = edgeIndex[w, v] = nextEdgeIndex
                nextEdgeIndex += 1

    subdFaces = []

    for i, f in enumerate(faces):
        kf = i + facePointOffset
        ke = [
            edgeIndex[v, w] + edgePointOffset
            for v, w in _cyclicPairs(f)
        ]

        subdFaces.append([f[0], ke[0], kf, ke[-1]])
        subdFaces.append([ke[0], f[1], ke[1], kf])

        for j in range(1, len(f) - 2):
            subdFaces.append([kf, ke[j], f[j + 1], ke[j + 1]])

        subdFaces.append([ke[-1], kf, ke[-2], f[-1]])

    return subdFaces


def _cyclicPairs(items):
    return zip(items, items[1:] + items[:1])


if __name__ == "__main__":
    import sys
    import obj

    with open(sys.argv[1]) as fp:
        mesh = obj.load(fp)

    subdivideMesh(mesh)
