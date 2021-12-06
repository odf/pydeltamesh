def loadAndProcessMesh(path):
    import obj
    from match import topology

    with open(path) as fp:
        data = obj.load(fp, path)

    topo = topology(
        [ f["vertices"] for f in data["faces"] ],
        data["vertices"]
    )

    return topo, data


def display(*meshes):
    import numpy as np
    import polyscope as ps # type: ignore

    ps.init()

    for i in range(len(meshes)):
        mesh = meshes[i]
        ps.register_surface_mesh(
            "mesh_%02d" % i,
            np.array(mesh["vertices"]),
            [ [ v - 1 for v in f["vertices"] ] for f in mesh["faces"] ]
        )

    ps.show()


if __name__ == '__main__':
    import sys

    import obj
    from match import match

    topoBase, dataBase = loadAndProcessMesh(sys.argv[1])

    symcounts = [
        [ [ len(vs) for vs in inst ] for inst in val ]
        for val in topoBase.values()
    ]
    print("Counts for base: %s" % " ".join(map(str, symcounts)))
    print()

    if len(sys.argv) > 2:
        topoMorph, dataMorph = loadAndProcessMesh(sys.argv[2])

        symcounts = [
            [ [ len(vs) for vs in inst ] for inst in val ]
            for val in topoMorph.values()
        ]
        print("Counts for morph: %s" % " ".join(map(str, symcounts)))
        print()

    mapping = match(topoBase, topoMorph)
    print("Mapping (of %d vertices):\n%s" % (len(mapping), mapping))

    dataOut = dataBase.copy()
    dataOut["vertices"] = dataOut["vertices"].copy()

    vertsIn = dataBase["vertices"]
    vertsOut = dataOut["vertices"]
    vertsMorph = dataMorph["vertices"]

    for v, w in mapping:
        vertsOut[v - 1] = vertsMorph[w - 1]

    '''
    display(
        { "vertices": vertsIn - [0.1, 0.0, 0.0], "faces": dataBase["faces"] },
        { "vertices": vertsOut, "faces": dataBase["faces"] }
    )
    #'''

    with open("test.obj", "w") as fp:
        obj.save(fp, dataOut, "test.mtl")
