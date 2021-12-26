def run():
    from uuid import uuid4

    from .io.pmd import Deltas, MorphTarget, write_pmd
 
    args = parseArguments()

    base = loadMesh(args.basepath, args.verbose)
    morph = loadMesh(args.morphpath, args.verbose)

    baseVerts = base.vertices
    baseNormals = base.normals
    morphVerts = morph.vertices

    morphDeltas = extractDeltas(
        baseVerts, morphVerts, baseNormals, args.verbose
    )

    name = args.morphname or "morph"
    actor = "BODY"
    uuid = str(uuid4())

    numbDeltas = len(baseVerts)

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


def extractDeltas(verticesBase, verticesMorph, normalsBase, verbose=False):
    import numpy as np
    from .io.pmd import Deltas

    if verbose:
        print("Computing deltas...")

    def norm(v):
        return np.sqrt(np.dot(v, v))

    deltaIndices = []
    deltaVectors = []

    for i in range(len(verticesBase)):
        d = verticesMorph[i] - verticesBase[i]

        if norm(d) > 1e-5:
            u = normalsBase[i]
            u /= norm(u)

            deltaIndices.append(i)
            deltaVectors.append([np.dot(d, u), 0.0, 0.0])

    if verbose:
        print("Computed %d deltas." % len(deltaIndices))

    return Deltas(len(verticesBase), deltaIndices, deltaVectors)


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
