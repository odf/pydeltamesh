import numpy as np

from collections import namedtuple


Mesh = namedtuple(
    "Mesh",
    [ "vertices", "normals", "texverts", "faces", "materials"]
)


def parseArguments():
    import argparse

    parser = argparse.ArgumentParser(
        description='Create a texture transfer object.'
    )
    parser.add_argument(
        'target',
        type=str,
        help='mesh with target UVsin OBJ format'
    )
    parser.add_argument(
        'source',
        type=str,
        help='mesh with source UVs in OBJ format'
    )
    parser.add_argument(
        'output',
        type=str,
        help='path for the OBJ-formatted output data'
    )
    parser.add_argument(
        "--target_translations",
        type=str,
        help='text file specifying material name translation for target'
    )
    parser.add_argument(
        "--source_translations",
        type=str,
        help='text file specifying material name translation for source'
    )

    return parser.parse_args()


def run():
    args = parseArguments()
    dataTarget = loadMesh(args.target)
    dataSource = loadMesh(args.source)
    lookupTarget = loadTranslations(args.target_translations)
    lookupSource = loadTranslations(args.source_translations)

    dataOutput = compose(dataTarget, dataSource, lookupTarget, lookupSource)
    saveMesh(args.output, dataOutput)


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


def loadTranslations(filepath):
    with open(filepath) as fp:
        text = fp.read()

    translation = {}
    for block in text.split("\n\n"):
        lines = block.strip().split("\n")
        dst = lines[0].strip()

        for src in lines[1:]:
            translation[src.strip()] = dst

    return translation


def compose(dataTarget, dataSource, lookupTarget, lookupSource):
    faceLookup = {}
    for face in dataTarget.faces:
        key, f = canonicalFace(face)
        faceLookup[key] = f

    faces = []

    for face in dataSource.faces:
        key, sourceFace = canonicalFace(face)
        targetFace = faceLookup[key]

        group = lookupTarget.get(targetFace.material)
        material = lookupSource.get(sourceFace.material)

        if not (group is None or material is None):
            faces.append(sourceFace._replace(
                vertices=targetFace.texverts,
                normals=[],
                group=group,
                material=material
            ))

    return Mesh(
        vertices=addDim(dataTarget.texverts),
        normals=np.array([]),
        texverts=dataSource.texverts,
        faces=faces,
        materials=dataSource.materials
    )


def canonicalFace(face):
    key = tuple(face.vertices)
    best = 0

    for i in range(1, len(key)):
        k = rotate(key, i)
        if k < key:
            best = i

    k = rotate(key, best)
    f = face._replace(
        vertices=rotate(face.vertices, best),
        texverts=rotate(face.texverts, best),
        normals=rotate(face.normals, best),
    )

    return k, f


def rotate(a, i):
    return a[i:] + a[:i]


def addDim(a):
    return np.array([list(p) + [0] for p in a])


if __name__ == '__main__':
    import sys
    from os.path import abspath, dirname

    sys.path.append(dirname(dirname(abspath(__file__))))

    run()
