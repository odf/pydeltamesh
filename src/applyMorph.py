def run():
    import os.path

    import obj
    from match import match

    args = parseArguments()

    topoBase, dataBase = loadAndProcessMesh(args.basepath)
    if args.verbose:
        print("Counts for base: %s" % symmetryCounts(topoBase))

    topoMorph, dataMorph = loadAndProcessMesh(args.morphpath)
    if args.verbose:
        print("Counts for morph: %s" % symmetryCounts(topoMorph))

    mapping = match(topoBase, topoMorph)
    if args.verbose:
        print("Mapping (of %d vertices):\n%s" % (len(mapping), mapping))

    dataOut = dataBase.copy()
    dataOut["vertices"] = dataOut["vertices"].copy()

    vertsOut = dataOut["vertices"]
    vertsMorph = dataMorph["vertices"]

    for v, w in mapping:
        vertsOut[v - 1] = vertsMorph[w - 1]

    if args.show:
        vertsIn = dataBase["vertices"]
        display(
            { "vertices": vertsIn - [0.1, 0.0, 0.0], "faces": dataBase["faces"] },
            { "vertices": vertsOut, "faces": dataBase["faces"] }
        )

    outname, outext = os.path.splitext(args.outpath)
    if outext != ".obj":
        outname += outext

    with open(outname + ".obj", "w") as fp:
        obj.save(fp, dataOut, outname + ".mtl")


def parseArguments():
    import argparse

    parser = argparse.ArgumentParser(description='Apply a morph to a mesh.')
    parser.add_argument(
        'basepath',
        type=str,
        help='original mesh in OBJ format'
    )
    parser.add_argument(
        'morphpath',
        type=str,
        help='morphed mesh in OBJ format'
    )
    parser.add_argument(
        'outpath',
        type=str,
        help='path for the OBJ-formatted output data'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        default=False,
        help='print some information about the meshes and matching'
    )
    parser.add_argument(
        '-s', '--show',
        action='store_true',
        default=False,
        help='display the input and output via PolyScope'
    )

    return parser.parse_args()


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


def symmetryCounts(topo):
    return [
        [ [ len(vs) for vs in inst ] for inst in val ]
        for val in topo.values()
    ]


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
    run()
