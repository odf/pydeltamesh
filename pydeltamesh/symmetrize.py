def parseArguments():
    import argparse

    parser = argparse.ArgumentParser(description='Symmetrize a mesh along x.')
    parser.add_argument(
        'inpath',
        type=str,
        help='original mesh in OBJ format'
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

    return parser.parse_args()


def run():
    args = parseArguments()

    dataIn = loadMesh(args.inpath)

    if args.verbose:
        log = lambda s: sys.stdout.write(s + '\n')
    else:
        log = None

    dataOut = symmetrize(dataIn, log=log)

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


def symmetrize(dataIn, log=None):
    from copy import deepcopy
    from pydeltamesh.mesh.match import match, topology
    from pydeltamesh.util.optimize import minimumWeightAssignment

    if log:
        log("Analyzing base mesh...")
    topoBase = topology(
        [ f.vertices for f in dataIn.faces ], dataIn.vertices
    )
    if log:
        n = sum(len(t) for t in topoBase.values())
        log("Base mesh has %d loose parts." % n)

    dataFlipped = deepcopy(dataIn)
    dataFlipped.vertices[:, 0] *= -1
    for f in dataFlipped.faces:
        f.vertices.reverse()

    if log:
        log("Analyzing flippe mesh...")
    topoFlipped = topology(
        [ f.vertices for f in dataFlipped.faces ], dataFlipped.vertices
    )
    if log:
        n = sum(len(t) for t in topoFlipped.values())
        log("Flipped mesh has %d loose parts." % n)

    if log:
        log("Matching the base and flipped mesh by loose parts...")
    mapping = match(topoBase, topoFlipped, minimumWeightAssignment, log=log)

    vertsOut = dataIn.vertices.copy()
    vertsFlipped = dataFlipped.vertices

    for v, w in mapping:
        vertsOut[v] = (vertsOut[v] + vertsFlipped[w]) / 2.0

    return dataIn._replace(vertices=vertsOut)


if __name__ == '__main__':
    import sys
    from os.path import abspath, dirname

    sys.path.append(dirname(dirname(abspath(__file__))))

    run()
