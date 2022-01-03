def parseArguments():
    import argparse

    parser = argparse.ArgumentParser(description='Apply a morph to a mesh.')
    parser.add_argument(
        'basepath',
        type=str,
        help='an unwelded full-figure mesh in OBJ format'
    )
    parser.add_argument(
        'weldedpath',
        type=str,
        help='a welded version of the base mesh in OBJ format'
    )
    parser.add_argument(
        'morphpath',
        type=str,
        help=''.join([
            'a morphed, welded or unwelded, possibly incomplete,',
            ' version of the base mesh in OBJ format'
        ])
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


def run(basepath, weldedpath, morphpath, name, verbose):
    import os.path
    from pydeltamesh.mesh.subd import subdivideMesh
    from pydeltamesh.io.pmd import write_pmd

    if name is None:
        name = os.path.splitext(os.path.split(morphpath)[1])[0]

    base = loadMesh(basepath, verbose)
    welded = removeDisconnectedVertices(loadMesh(weldedpath, verbose))
    morph = removeDisconnectedVertices(loadMesh(morphpath, verbose))

    morphedVertsSubd0 = morph.vertices[: len(welded.vertices)]
    morphSubd0 = welded._replace(vertices=morphedVertsSubd0)

    targets = makeMorphTargets(name, base, morphSubd0, verbose)

    if verbose:
        print("Subdividing base mesh with baked down morph...")

    subdBase = morphSubd0
    subdLevel = 0
    while len(subdBase.faces) < len(morph.faces):
        subdBase = subdivideMesh(subdBase)
        subdLevel += 1

    if verbose:
        print("Subdivided %d times." % subdLevel)

    subdTarget = makeSubdTarget(name, subdBase, morph, subdLevel, verbose)
    targets.append(subdTarget)

    with open("%s.pmd" % name, "wb") as fp:
        write_pmd(fp, targets)

    with open("%s.pz2" % name, "w") as fp:
        writeInjectionPoseFile(fp, name, targets)


def loadMesh(path, verbose=False):
    from pydeltamesh.io import obj

    if verbose:
        print("Loading mesh from %s..." % path)

    with open(path) as fp:
        data = obj.load(fp, path)

    if verbose:
        print("Loaded mesh with %d vertices and %d faces." % (
            len(data.vertices), len(data.faces)
        ))

    return data


def removeDisconnectedVertices(mesh):
    used = set()
    for f in mesh.faces:
        for v in f.vertices:
            used.add(v)
    used = sorted(used)

    mapping = [-1] * len(mesh.vertices)
    for i, v in enumerate(used):
        mapping[v] = i

    vertsOut = mesh.vertices[used]

    facesOut = []
    for f in mesh.faces:
        vs = [ mapping[v] for v in f.vertices ]
        facesOut.append(f._replace(vertices=vs))

    return mesh._replace(vertices=vertsOut, faces=facesOut)


def makeMorphTargets(name, baseMesh, morphedMesh, verbose=False):
    from uuid import uuid4
    from pydeltamesh.io.pmd import MorphTarget

    baseParts = processedByGroup(baseMesh, verbose)
    morphParts = processedByGroup(morphedMesh, verbose)

    targets = []

    for actor in baseParts:
        if actor in morphParts:
            deltas = findDeltas(baseParts[actor], morphParts[actor])

            if deltas:
                if verbose:
                    n = len(deltas.indices)
                    print("Found %d deltas for %s." % (n, actor))

                key = str(uuid4())
                targets.append(MorphTarget(name, actor, key, deltas, {}))

    return targets


def makeSubdTarget(name, base, morph, subdLevel, verbose=False):
    from uuid import uuid4
    from pydeltamesh.io.pmd import Deltas, MorphTarget

    actor = "BODY"
    key = str(uuid4())
    baseDeltas = Deltas(len(base.vertices), [], [])

    subdDeltas = {}

    if subdLevel > 0:
        for level in range(1, subdLevel):
            subdDeltas[level] = Deltas(0, [], [])
        subdDeltas[subdLevel] = findSubdDeltas(base, morph, args.verbose)

    return MorphTarget(name, actor, key, baseDeltas, subdDeltas)


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


def findDeltas(base, morph):
    from pydeltamesh.io.pmd import Deltas
    from pydeltamesh.mesh.match import match
    from pydeltamesh.util.optimize import minimumWeightAssignment

    mapping = match(base, morph, minimumWeightAssignment)

    idcs = []
    vecs = []
    for u, v in sorted(mapping):
        d = morph.vertexPosition(v) - base.vertexPosition(u)
        if norm(d) > 1e-5:
            idcs.append(u)
            vecs.append(d)

    if len(idcs):
        offset = base.verticesUsed[0]
        num = len(base.verticesUsed)

        return Deltas(num, [i - offset for i in idcs], vecs)
    else:
        return None


def findSubdDeltas(base, morph, verbose=False):
    from pydeltamesh.io.pmd import Deltas

    if verbose:
        print("Computing deltas...")

    nv = 1 + max(max(f.vertices) for f in base.faces)

    neighbors = [[] for _ in range(nv)]
    for f in base.faces:
        for i in range(len(f.vertices)):
            v = f.vertices[i - 1]
            w = f.vertices[i]
            if w not in neighbors[v]:
                neighbors[v].append(w)
            if v not in neighbors[w]:
                neighbors[w].append(v)

    deltaIndices = []
    deltaVectors = []

    for v in range(len(base.vertices)):
        d = morph.vertices[v] - base.vertices[v]

        if norm(d) > 1e-5:
            p = base.vertices[v]
            qs = base.vertices[neighbors[v]]
            n = normalized(vertexNormal(p, qs))
            deltaIndices.append(v)
            deltaVectors.append([dot(d, n), 0.0, 0.0])

    if verbose:
        print("Computed %d subd deltas." % len(deltaIndices))

    return Deltas(len(base.vertices), deltaIndices, deltaVectors)


def vertexNormal(v, ws):
    ds = [w - v for w in ws]

    ns = []
    for i in range(len(ds)):
        for j in range(i + 1, len(ds)):
            ns.append(cross(ds[i], ds[j]))

    ns.sort(key=norm, reverse=True)

    normal = [0.0, 0.0, 0.0]
    for i in range(min(len(ds), len(ns))):
        normal[0] += ns[i][0]
        normal[1] += ns[i][1]
        normal[2] += ns[i][2]

    return normalized(normal)


def dot(v, w):
    return sum(x * y for x, y in zip(v, w))


def cross(v, w):
    return [
        v[1] * w[2] - v[2] * w[1],
        v[2] * w[0] - v[0] * w[2],
        v[0] * w[1] - v[1] * w[0]
    ]


def norm(v):
    return sum(x * x for x in v)**0.5


def normalized(v):
    n = norm(v)
    return [x / n for x in v] if n > 0 else v


def writeInjectionPoseFile(fp, name, targets):
    from pydeltamesh.io.poserFile import PoserFile

    morphPath = ':Runtime:libraries:Pose:%s.pmd' % name

    source = PoserFile(fileTemplate.splitlines())
    root = source.root

    next(root.select('injectPMDFileMorphs')).rest = morphPath
    next(root.select('createFullBodyMorph')).rest = name

    for target in targets:
        actor = PoserFile(actorTemplate.splitlines()).root
        next(actor.select('actor')).rest = target.actor + ":1"

        targetNode = next(actor.select('actor', 'channels', 'targetGeom'))
        targetNode.rest = name
        groupsNode = next(actor.select('actor', 'channels', 'groups'))

        if target.actor == "BODY":
            next(groupsNode.select('groupNode', 'parmNode')).rest = target.name

            valueOpNode = next(targetNode.select('valueOpDeltaAdd'))
            for _ in range(5):
                valueOpNode.nextSibling.unlink()
            valueOpNode.unlink()
        else:
            groupsNode.unlink()
            next(targetNode.select('@name')).text = target.name

        next(targetNode.select('name')).rest = target.name
        next(targetNode.select('uuid')).rest = target.uuid

        next(targetNode.select('numbDeltas')).rest = str(
            target.deltas.numb_deltas
        )

        next(root.select('}')).prependSibling(actor)

    source.writeTo(fp)


fileTemplate = '''{
version
    {
    number 12
    build 619
    }
injectPMDFileMorphs -
createFullBodyMorph -
}
'''


actorTemplate = '''
actor BODY:1
    {
    channels
        {
		groups
			{
			groupNode Morph
				{
				parmNode @name
				}
			}
        targetGeom -
            {
            name @name
            initValue 0
            hidden 0
            enabled 1
            forceLimits 1
            min -100000
            max 100000
            trackingScale 0.004
            masterSynched 1
            interpStyleLocked 0
            valueOpDeltaAdd
                Figure 1
                BODY:1
                @name
                strength 1.000000
                deltaAddDelta 1.000000
            uuid @uuid
            numbDeltas 0
            useBinaryMorph 1
            blendType 0
            }
        }
    }
'''


if __name__ == '__main__':
    import sys
    from os.path import dirname

    sys.path.append(dirname(dirname(__file__)))

    args = parseArguments()

    basepath = args.basepath
    weldedpath = args.weldedpath
    morphpath = args.morphpath
    morphname = args.morphname
    verbose = args.verbose

    run(basepath, weldedpath, morphpath, morphname, verbose)
