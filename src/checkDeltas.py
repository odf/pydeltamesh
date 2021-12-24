import numpy as np


def run():
    import sys

    baseMesh = loadMesh(sys.argv[1])
    baseVerts = baseMesh.vertices
    baseNormals = baseMesh.normals
    baseTexVerts = baseMesh.texverts

    vertsN = loadMesh(sys.argv[2]).vertices
    vertsU = loadMesh(sys.argv[3]).vertices
    vertsV = loadMesh(sys.argv[4]).vertices

    dN = normalized(vertsN - baseVerts)
    dU = normalized(vertsU - baseVerts)
    dV = normalized(vertsV - baseVerts)

    badNormals = [
        i for i in range(len(dN)) if dot(baseNormals[i], dN[i]) < 0.99
    ]
    if len(badNormals):
        print("Bad normals:")
        for i in badNormals:
            print("%5d: %s" % (i, dot(baseNormals[i], dN[i])))
        print()
    else:
        print("No bad normals.")


def dot(u, v):
    return np.sum(u * v, axis=-1)


def norm(v):
    return np.sqrt(dot(v, v))


def normalized(v):
    lv = norm(v)
    return v / np.where(lv, lv, 1.0).reshape(-1, 1)


def loadMesh(path):
    from lib import obj

    with open(path) as fp:
        data = obj.load(fp, path)

    return data


if __name__ == "__main__":
    run()
