def parseArguments():
    import argparse

    parser = argparse.ArgumentParser(description='Convert mesh UVs to UDIM.')
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
    import sys

    args = parseArguments()

    dataIn = loadMesh(args.inpath)

    if args.verbose:
        log = lambda s: sys.stdout.write(s + '\n')
    else:
        log = None

    dataOut = toUDIM(dataIn, log=log)

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


def toUDIM(dataIn, log=None):
    shifts = {
        "skin_HEAD": (0,0),
        "lacrimals": (0,0),
        "lips": (0,0),

        "skin_BODY": (1,0),

        "skin_LEGS": (2,0),

        "skin_ARMS": (3,0),
        "nailsFingers": (3,0),
        "nailsToes": (3,0),

        "lashes": (4,0),

        "mouthInner": (5,0),
        "teeth": (5,0),
        "tongue": (5,0),

        "irisLeft": (6,0),
        "pupilLeft": (6,0),
        "scleraLeft": (6,0),

        "irisRight": (7,0),
        "pupilRight": (7,0),
        "scleraRight": (7,0),

        "corneaLeft": (8,0),

        "corneaRight": (9,0),
    }

    uvShift = {}

    for f in dataIn.faces:
        s = shifts[f.material]

        for vt in f.texverts:
            if uvShift.get(vt) not in [None, s]:
                raise RuntimeError("texture vertex %d has conflicting shifts")
            uvShift[vt] = s

    tvOut = []
    for i in range(len(dataIn.texverts)):
        tvOut.append(dataIn.texverts[i] % 1.0 + uvShift[i])

    return dataIn._replace(texverts=tvOut)


if __name__ == '__main__':
    import sys
    from os.path import abspath, dirname

    sys.path.append(dirname(dirname(abspath(__file__))))

    run()
