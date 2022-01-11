def run():
    import os.path

    from pydeltamesh.fileio import obj
    from pydeltamesh.mesh.match import match
    from pydeltamesh.util.optimize import minimumWeightAssignment

    args = parseArguments()

    topoBase, dataBase = loadAndProcessMesh(args.basepath)
    if args.verbose:
        print("Counts for base: %s" % symmetryCounts(topoBase))

    topoMorph, dataMorph = loadAndProcessMesh(args.morphpath)
    if args.verbose:
        print("Counts for morph: %s" % symmetryCounts(topoMorph))

    mapping = match(
        topoBase, topoMorph, minimumWeightAssignment, verbose=args.verbose
    )
    if args.verbose:
        nrMapped = len(list(mapping))
        nrMoved = len([u for u, v in mapping if u != v])
        print("Mapped %d vertices, moved %d" % (nrMapped, nrMoved))

    vertsOut = dataBase.vertices.copy()
    vertsMorph = dataMorph.vertices

    for v, w in mapping:
        vertsOut[v] = vertsMorph[w]

    dataOut = dataBase._replace(vertices=vertsOut)

    if args.show:
        shifted = dataBase.vertices - [0.1, 0.0, 0.0]
        display(dataBase._replace(vertices=shifted), dataOut)

    outname, outext = os.path.splitext(args.outpath)
    if outext != ".obj":
        outname += outext

    with open(outname + ".obj", "w") as fp:
        obj.save(fp, dataOut, outname + ".mtl", writeNormals=False)


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
    from pydeltamesh.fileio import obj
    from pydeltamesh.mesh.match import topology

    with open(path) as fp:
        data = obj.load(fp, path)

    topo = topology([ f.vertices for f in data.faces ], data.vertices)

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
            np.array(mesh.vertices),
            [ f.vertices for f in mesh.faces ]
        )

    ps.show()


if __name__ == '__main__':
    import sys
    from os.path import abspath, dirname

    sys.path.append(dirname(dirname(abspath(__file__))))

    run()
