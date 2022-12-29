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
        "target_translations",
        type=str,
        help='text file specifying material name translation for target'
    )
    parser.add_argument(
        "source_translations",
        type=str,
        help='text file specifying material name translation for source'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default="transfer.obj",
        help='path for the OBJ-formatted output data'
    )

    return parser.parse_args()


def run():
    args = parseArguments()
    dataTarget = loadMesh(args.target)
    dataSource = loadMesh(args.source)
    targetSpec = loadTranslations(args.target_translations)
    sourceSpec = loadTranslations(args.source_translations)

    dataOutput = compose(dataTarget, dataSource, targetSpec, sourceSpec)
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

    tiles = []
    translation = {}

    for block in text.split("\n\n"):
        lines = block.strip().split("\n")
        dst = lines[0].strip()
        tiles.append(dst)

        for src in lines[1:]:
            translation[src.strip()] = dst

    return tiles, translation


def compose(dataTarget, dataSource, targetSpec, sourceSpec):
    faceLookup = {}
    for face in dataTarget.faces:
        key, f = canonicalFace(face)
        faceLookup[key] = f

    tilesTarget, lookupTarget = targetSpec
    _, lookupSource = sourceSpec

    shifts = {}
    for i, k in enumerate(tilesTarget):
        shifts[k] = i

    faces = []
    objVertices = []
    vertexLookup = {}

    for face in dataSource.faces:
        key, sourceFace = canonicalFace(face)
        targetFace = faceLookup[key]

        group = lookupTarget.get(targetFace.material)
        material = lookupSource.get(sourceFace.material)

        if not (group is None or material is None):
            faceVertices = []
            for k in targetFace.texverts:
                if not (k, group) in vertexLookup:
                    vertexLookup[k, group] = len(objVertices)
                    x, y = dataTarget.texverts[k]
                    x = (x % 1) + shifts[group] - 0.5
                    y = (y % 1) - 0.5
                    objVertices.append((x, y, 0.0))
                faceVertices.append(vertexLookup[k, group])

            faces.append(sourceFace._replace(
                vertices=faceVertices,
                normals=[],
                group=group,
                material=material
            ))

    return Mesh(
        vertices=np.array(objVertices),
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
