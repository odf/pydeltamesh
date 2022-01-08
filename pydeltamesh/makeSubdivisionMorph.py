import numpy as _np

from collections import namedtuple as _namedtuple


Face = _namedtuple("Face", [ "vertices", "group" ])

Mesh = _namedtuple("Mesh", [ "vertices", "faces" ])


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
    from pydeltamesh.io.pmd import write_pmd
    from pydeltamesh.mesh.subd import Complex, subdivideTopology

    if name is None:
        name = os.path.splitext(os.path.split(morphpath)[1])[0]

    base = loadMesh(basepath, verbose)
    weldedRaw = loadMesh(weldedpath, verbose)
    morph = loadMesh(morphpath, verbose)

    welded = expandNumbering(weldedRaw, usedVertices(morph))

    faces = [f.vertices for f in welded.faces]
    complexes = []
    while len(faces) < len(morph.faces):
        cx = Complex(faces)
        complexes.append(cx)
        faces = subdivideTopology(cx)

    vertsMorphed = bakeDownMorph(
        welded.vertices, morph.vertices, complexes, verbose
    )
    weldedMorphed = welded._replace(vertices=vertsMorphed)
    targets = makeMorphTargets(name, base, weldedMorphed, verbose)

    targets.append(makeSubdTarget(
        name, weldedMorphed, morph, complexes, verbose
    ))

    with open("%s.pmd" % name, "wb") as fp:
        write_pmd(fp, targets)

    with open("%s.pz2" % name, "w") as fp:
        writeInjectionPoseFile(fp, name, targets)


def loadMesh(path, verbose=False):
    if verbose:
        print("Loading mesh from %s..." % path)

    with open(path) as fp:
        vertices = []
        faces = []
        group = None

        for line in fp:
            fields = line.split()

            if len(fields) == 0:
                pass
            elif fields[0] == 'f':
                ns = [ int(s[: s.find('/')]) for s in fields[1:] ]
                vs = [ n - 1 if n > 0 else len(vertices) + n for n in ns]
                faces.append(Face(vs, group))
            elif fields[0] == 'g':
                group = fields[1]
            elif fields[0] == 'v':
                vertices.append([float(s) for s in fields[1:]])

    if verbose:
        print("Loaded mesh with %d vertices and %d faces." % (
            len(vertices), len(faces)
        ))

    return Mesh(_np.array(vertices), faces)


def usedVertices(mesh):
    used = set()
    for f in mesh.faces:
        for v in f.vertices:
            used.add(v)
    return sorted(used)


def expandNumbering(mesh, used):
    nv = len(mesh.vertices)
    vertsOut = _np.zeros((used[nv - 1] + 1, 3))
    for i in range(nv):
        vertsOut[used[i]] = mesh.vertices[i]

    facesOut = []
    for f in mesh.faces:
        facesOut.append(f._replace(vertices=[used[i] for i in f.vertices]))

    return mesh._replace(vertices=vertsOut, faces=facesOut)


def bakeDownMorph(baseVerts, morphVerts, complexes, verbose=False):
    from pydeltamesh.mesh.subd import (
        adjustVertexData, interpolatePerVertexData
    )

    if len(complexes) == 0:
        return morphVerts

    if verbose:
        print("Subdividing for baking...")

    n = len(baseVerts)

    verts = baseVerts
    for i in range(len(complexes) - 1):
        verts = interpolatePerVertexData(verts, complexes[i])
    verts = adjustVertexData(verts, complexes[-1])

    if verbose:
        print("Subdivided %d times." % len(complexes))

    return baseVerts + morphVerts[: n] - verts[: n]


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


def makeSubdTarget(name, baseSubd0, morph, complexes, verbose=False):
    from uuid import uuid4
    from pydeltamesh.mesh.subd import (
        interpolatePerVertexData, subdivideTopology
    )
    from pydeltamesh.io.pmd import Deltas, MorphTarget

    actor = "BODY"
    key = str(uuid4())
    baseDeltas = Deltas(len(morph.vertices), [], [])
    subdDeltas = {}

    subdLevel = len(complexes)
    verts = baseSubd0.vertices

    for level in range(1, subdLevel + 1):
        if verbose:
            print("Subdividing morph for level %d..." % level)

        verts = interpolatePerVertexData(verts, complexes[level - 1])
        faces = subdivideTopology(complexes[level - 1])

        if verbose:
            print("Finding deltas for level %d..." % level)

        morphedVerts = bakeDownMorph(
            verts, morph.vertices, complexes[level:], verbose
        )

        deltas, displacements = findSubdDeltas(verts, morphedVerts, faces)
        subdDeltas[level] = deltas

        if verbose:
            print("Found %d deltas." % len(deltas.indices))

        if level < subdLevel:
            verts[deltas.indices] += displacements

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


def findSubdDeltas(baseVertices, morphedVertices, faces):
    from pydeltamesh.io.pmd import Deltas

    before, after = angleNeighbors(faces)

    diffs = morphedVertices[: len(baseVertices)] - baseVertices
    norms = _np.sqrt(_np.sum(diffs * diffs, axis=1))
    deltaIndices = _np.where(norms > 1e-5)[0]

    normals = [
        vertexNormal(
            baseVertices[v],
            baseVertices[after[v]],
            baseVertices[before[v]]
        )
        for v in deltaIndices
    ]

    diffsAlongNormals = _np.sum(diffs[deltaIndices] * normals, axis=1)
    deltaVectors = diffsAlongNormals[:, None] * [1, 0, 0]
    displacements = diffsAlongNormals[:, None] * normals

    deltas = Deltas(len(baseVertices), deltaIndices, deltaVectors)

    return deltas, displacements


def angleNeighbors(faces):
    nv = 1 + max(max(f) for f in faces)

    before = [[] for _ in range(nv)]
    after = [[] for _ in range(nv)]
    for f in faces:
        for i in range(len(f)):
            u = f[i - 2]
            v = f[i - 1]
            w = f[i]
            before[v].append(u)
            after[v].append(w)
    return before,after


def vertexNormal(p, qs, rs):
    normal = [0.0, 0.0, 0.0]

    for i in range(len(qs)):
        n = cross(qs[i] - p, rs[i] - p)
        normal[0] += n[0]
        normal[1] += n[1]
        normal[2] += n[2]

    return normalized(normal)


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
