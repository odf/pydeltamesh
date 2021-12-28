def run():
    from uuid import uuid4

    from .io.pmd import Deltas, MorphTarget, write_pmd

    args = parseArguments()

    base = loadMesh(args.basepath, args.verbose)
    morph = loadMesh(args.morphpath, args.verbose)

    morphDeltas = extractDeltas(base, morph, args.verbose)

    name = args.morphname or "morph"
    actor = "BODY"
    uuid = str(uuid4())

    numbDeltas = len(base.vertices)

    if args.subdivisionLevel == 0:
        deltas = morphDeltas
        subd_deltas = {}
    else:
        deltas = Deltas(numbDeltas, [], [])
        subd_deltas = {}
        for level in range(1, args.subdivisionLevel):
            subd_deltas[level] = Deltas(0, [], [])
        subd_deltas[args.subdivisionLevel] = morphDeltas

    target = MorphTarget(name, actor, uuid, deltas, subd_deltas)

    if args.verbose:
        print("Writing the PMD...")

    with open("%s.pmd" % name, "wb") as fp:
        write_pmd(fp, [target])

    if args.verbose:
        print("Wrote the PMD.")

    writeInjectionPoseFile(actor, name, uuid, numbDeltas, args.verbose)


def loadMesh(path, verbose=False):
    from .io import obj

    if verbose:
        print("Loading mesh from %s..." % path)

    with open(path) as fp:
        data = obj.load(fp, path)

    if verbose:
        print("Loaded mesh with %d vertices and %d faces." % (
            len(data.vertices), len(data.faces)
        ))

    return data


def processMesh(data, verbose=False):
    from .mesh.match import topology

    if verbose:
        print("Analysing the topology...")

    topo = topology([ f.vertices for f in data.faces ], data.vertices)

    if verbose:
        print("Analysed %d connected mesh parts." % len(topo))

    return topo


def extractDeltas(base, morph, verbose=False):
    import numpy as np
    from .io.pmd import Deltas

    if verbose:
        print("Computing deltas...")

    vertexLocations = {}
    for i, f in enumerate(base.faces):
        for j, v in enumerate(f.vertices):
            vertexLocations.setdefault(v, []).append((i, j))

    deltaIndices = []
    deltaVectors = []

    for i in range(len(base.vertices)):
        d = morph.vertices[i] - base.vertices[i]

        if norm(d) > 1e-5:
            n, u, v = deltaDirectionsAtVertex(i, base, vertexLocations)

            deltaIndices.append(i)
            deltaVectors.append([np.dot(d, n), np.dot(d, u), np.dot(d, v)])

    if verbose:
        print("Computed %d deltas." % len(deltaIndices))

    return Deltas(len(base.vertices), deltaIndices, deltaVectors)


def deltaDirectionsAtVertex(v, base, vertexLocations):
    import numpy as np

    uDir = np.zeros(3)
    vDir = np.zeros(3)

    for i, j in vertexLocations[v]:
        vs = base.faces[i].vertices
        a = base.vertices[vs[j]]
        b = base.vertices[vs[(j + 1) % len(vs)]]
        c = base.vertices[vs[j - 1]]

        tvs = base.faces[i].texverts
        p = base.texverts[tvs[j]]
        q = base.texverts[tvs[(j + 1) % len(tvs)]]
        r = base.texverts[tvs[j - 1]]

        if abs(cross2d(p, q) + cross2d(q, r) + cross2d(r, p)) > 1e-12:
            for k, dir in [(0, vDir), (1, uDir)]:
                if abs(q[k] - r[k]) > 1e-8:
                    t = (p[k] - q[k]) / (r[k] - q[k])

                    if 0 <= t <= 1:
                        d = (1 - t) * b + t * c - a
                        if (1 - t) * q[k - 1] + t * r[k - 1] < p[k - 1]:
                            d = -d
                        dir[:] += normalized(d)

    n = normalized(base.normals[v])
    u = normalized(uDir - np.dot(n, uDir) * n)
    v = normalized(vDir - np.dot(n, vDir) * n - np.dot(u, vDir) * u)

    return n, u, v


def norm(v):
    return sum(x * x for x in v)**0.5


def normalized(v):
    n = norm(v)
    return v / n if n > 0 else v


def cross2d(v, w):
    return v[0] * w[1] - v[1] * w[0]


def writeInjectionPoseFile(actor, name, uuid, numbDeltas, verbose=False):
    from .io.poserFile import PoserFile

    templateText = '''{

version
        {
        number 12
        build 619
        }
injectPMDFileMorphs -
createFullBodyMorph -

actor -
        {
        channels
                {
                targetGeom -
                        {
                        name -
                        initValue 0
                        hidden 0
                        enabled 1
                        forceLimits 1
                        min 0
                        max 1
                        trackingScale 0.004
                        masterSynched 1
                        interpStyleLocked 0
                        uuid -
                        numbDeltas -
                        useBinaryMorph 1
                        blendType 0
                        }
                }
        }
}
'''

    morphPath = ':Runtime:libraries:Pose:%s.pmd' % name

    source = PoserFile(templateText.splitlines())
    root = source.root

    next(root.select('injectPMDFileMorphs')).rest = morphPath
    next(root.select('createFullBodyMorph')).rest = name
    next(root.select('actor')).rest = actor + ":1"

    targetNode = next(root.select('actor', 'channels', 'targetGeom'))
    targetNode.rest = name
    next(targetNode.select('name')).rest = name
    next(targetNode.select('uuid')).rest = uuid
    next(targetNode.select('numbDeltas')).rest = str(numbDeltas)

    if verbose:
        print("Writing the PZ2...")

    with open("%s.pz2" % name, "w") as fp:
        source.writeTo(fp)

    if verbose:
        print("Wrote the PZ2.")


def parseArguments():
    import argparse

    parser = argparse.ArgumentParser(description='Apply a morph to a mesh.')
    parser.add_argument(
        'basepath',
        type=str,
        help='the subdivided base mesh in OBJ format'
    )
    parser.add_argument(
        'morphpath',
        type=str,
        help='the subdivided and morphed mesh in OBJ format'
    )
    parser.add_argument(
        '-d', '--subdivisionLevel',
        type=int,
        default=1,
        help='the subdivision level of both the base and morphed mesh'
    )
    parser.add_argument(
        '-n', '--morphname',
        type=str,
        help='the name of the morph to create (defaults to the filename)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        default=False,
        help='print some information while working'
    )

    return parser.parse_args()


if __name__ == '__main__':
    run()
