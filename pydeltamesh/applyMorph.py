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


def run():
    args = parseArguments()

    dataBase = loadMesh(args.basepath)
    dataMorph = loadMesh(args.morphpath)

    if args.verbose:
        log = lambda s: sys.stdout.write(s + '\n')
    else:
        log = None

    dataOut = mapVertices(dataBase, dataMorph, log=log)

    if args.show:
        shifted = dataBase.vertices - [0.1, 0.0, 0.0]
        display(dataBase._replace(vertices=shifted), dataOut)

    saveMesh(args.outpath, dataOut)


def loadMesh(path):
    from pydeltamesh.fileio import obj

    with open(path) as fp:
        mesh = obj.load(fp, path)

    return mesh


def saveMesh(path, mesh):
    import os.path
    from pydeltamesh.fileio import obj

    outname, outext = os.path.splitext(path)
    if outext != ".obj":
        outname += outext

    objpath = outname + '.obj'
    mtlpath = outname + '.mtl'

    with open(objpath, "w") as fp:
        obj.save(fp, mesh, mtlpath, writeNormals=False)


def mapVertices(dataBase, dataMorph, log=None):
    from pydeltamesh.mesh.match import match, topology
    from pydeltamesh.util.optimize import minimumWeightAssignment

    if log:
        log("Analyzing base mesh...")
    topoBase = topology(
        [ f.vertices for f in dataBase.faces ], dataBase.vertices
    )
    if log:
        n = sum(len(t) for t in topoBase.values())
        log("Base mesh has %d loose parts." % n)

    if log:
        log("Analyzing morphed mesh...")
    topoMorph = topology(
        [ f.vertices for f in dataMorph.faces ], dataMorph.vertices
    )
    if log:
        n = sum(len(t) for t in topoMorph.values())
        log("Morphed mesh has %d loose parts." % n)

    if log:
        log("Matching the base and morphed mesh by loose parts...")
    mapping = match(topoBase, topoMorph, minimumWeightAssignment, log=log)

    vertsOut = dataBase.vertices.copy()
    vertsMorph = dataMorph.vertices

    for v, w in mapping:
        vertsOut[v] = vertsMorph[w]

    return dataBase._replace(vertices=vertsOut)


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
