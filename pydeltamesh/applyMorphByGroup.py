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

    return parser.parse_args()


def run():
    from pydeltamesh.mesh.match import match
    from pydeltamesh.util.optimize import minimumWeightAssignment

    args = parseArguments()

    verbose = args.verbose
    basepath = args.basepath
    morphpath = args.morphpath
    outpath = args.outpath

    base = loadMesh(basepath)
    baseParts = processedByGroup(base, verbose)

    morph = loadMesh(morphpath)
    morphParts = processedByGroup(morph, verbose)

    vertsOut = base.vertices.copy()
    vertsMorph = morph.vertices

    log = (lambda s: sys.stdout.write(s + '\n')) if args.verbose else None

    for actor in baseParts:
        if actor in morphParts:
            mapping = match(
                baseParts[actor], morphParts[actor], minimumWeightAssignment, log=log
            )
            for v, w in mapping:
                vertsOut[v] = vertsMorph[w]

    saveMesh(outpath, base._replace(vertices=vertsOut))


def processedByGroup(data, verbose=False):
    import re

    from pydeltamesh.mesh.match import topology

    if verbose:
        print("Splitting mesh by groups...")

    groupFaces = {}
    for f in data.faces:
        groupFaces.setdefault(f.group, []).append(f.vertices)

    if verbose:
        print("Split mesh into %d groups." % len(groupFaces))

    topologies = {}

    if verbose:
        print("Analysing actor topologies...")

    for group in groupFaces:
        actor = re.sub(':.*', '', group)

        topo = topology(groupFaces[group], data.vertices)
        topologies[actor] = topo

    if verbose:
        print("Analysed all actor topologies.")

    return topologies


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


if __name__ == '__main__':
    import sys
    from os.path import abspath, dirname

    sys.path.append(dirname(dirname(abspath(__file__))))

    run()
