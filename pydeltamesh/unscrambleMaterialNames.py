def parseArguments():
    import argparse

    parser = argparse.ArgumentParser(
        description='Split material names into group and material names.'
    )
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

    return parser.parse_args()


def run():
    args = parseArguments()
    dataIn = loadMesh(args.inpath)
    dataOut = unscramble(dataIn)
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


def unscramble(dataIn):
    facesOut = []
    for f in dataIn.faces:
        groupOut, matOut = f.material.split(":")
        facesOut.append(f._replace(group=groupOut, material=matOut))

    matsOut = {k: v for k, v in dataIn.materials.items() if not ':' in k}

    return dataIn._replace(faces=facesOut, materials=matsOut)


if __name__ == '__main__':
    import sys
    from os.path import abspath, dirname

    sys.path.append(dirname(dirname(abspath(__file__))))

    run()
