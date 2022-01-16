import numpy as _np

from collections import namedtuple as _namedtuple


Mesh = _namedtuple("Mesh", [ "vertices", "faces", "faceGroups" ])


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


def run(unweldedBasePath, weldedBasePath, morphPath, name, log=None):
    import os.path
    from pydeltamesh.fileio.pmd import write_pmd

    if name is None:
        name = os.path.splitext(os.path.split(morphPath)[1])[0]

    unwelded = loadMesh(unweldedBasePath, log=log)
    weldedRaw = loadMesh(weldedBasePath, log=log)
    morph = loadMesh(morphPath, log=log)

    used = usedVertices(morph)
    welded = expandNumbering(weldedRaw, sorted(used))

    targets = makeTargets(name, unwelded, welded, morph, used, log=log)

    with open("%s.pmd" % name, "wb") as fp:
        write_pmd(fp, targets)

    with open("%s.pz2" % name, "w") as fp:
        writeInjectionPoseFile(fp, name, targets)


def makeTargets(
    name, unwelded, welded, morph, used, postTransform=False, log=None
):
    from pydeltamesh.mesh.subd import Complex, subdivideTopology

    faces = welded.faces
    complexes = []

    if len(faces) < len(morph.faces):
        cx = Complex(faces)
        complexes.append(cx)

        while 4 * len(cx.faces) < len(morph.faces):
            faces = subdivideTopology(cx)
            cx = Complex(faces)
            complexes.append(cx)

    vertsMorphed = bakeDownMorph(
        welded.vertices, morph.vertices, complexes, log=log
    )

    if postTransform:
        displacements = vertsMorphed - welded.vertices
    else:
        displacements = vertsMorphed - unwelded.vertices

    targets = makeBaseTargets(name, unwelded, displacements, used, log=log)

    weldedMorphed = welded._replace(vertices=vertsMorphed)
    targets.append(makeSubdTarget(
        name, weldedMorphed, morph, complexes, log=log
    ))

    return targets


def loadMesh(path, log=None):
    if log:
        log("Loading mesh from %s..." % path)

    with open(path) as fp:
        vertices = []
        faces = []
        faceGroups = []
        group = None

        for line in fp:
            fields = line.split()

            if len(fields) == 0:
                pass
            elif fields[0] == 'f':
                if '/' in line:
                    ns = [ int(s[: s.find('/')]) for s in fields[1:] ]
                else:
                    ns = map(int, fields[1:])
                vs = [ n - 1 if n > 0 else len(vertices) + n for n in ns]
                faces.append(vs)
                faceGroups.append(group)
            elif fields[0] == 'g':
                group = fields[1]
            elif fields[0] == 'v':
                vertices.append([float(s) for s in fields[1:]])

    if log:
        log("Loaded mesh with %d vertices and %d faces." % (
            len(vertices), len(faces)
        ))

    return Mesh(_np.array(vertices), faces, faceGroups)


def saveMesh(path, mesh):
    lines = []

    for v in mesh.vertices:
        lines.append("v %.8f %.8f %.8f\n" % tuple(v))

    group = None

    for i, f in enumerate(mesh.faces):
        if mesh.faceGroups[i] != group:
            group = mesh.faceGroups[i]
            lines.append("g %s\n" % group)

        lines.append("f")
        for v in f:
            lines.append(" %s//" % (v + 1))
        lines.append("\n")

    with open(path, "w") as fp:
        fp.write("".join(lines))


def usedVertices(mesh):
    used = set()
    for f in mesh.faces:
        used.update(f)
    return used


def expandNumbering(mesh, used):
    nv = len(mesh.vertices)
    if len(used) < nv:
        used += range(used[-1] + 1, nv + used[-1] + 1 - len(used))

    vertsOut = _np.zeros((used[nv - 1] + 1, 3))
    for i in range(nv):
        vertsOut[used[i]] = mesh.vertices[i]

    facesOut = []
    for f in mesh.faces:
        facesOut.append([used[i] for i in f])

    return mesh._replace(vertices=vertsOut, faces=facesOut)


def compressNumbering(mesh, used):
    nv = len(used)
    vertsOut = _np.zeros((nv, 3))
    for i in range(nv):
        vertsOut[i] = mesh.vertices[used[i]]

    mapping = _np.full(len(mesh.vertices), -1)
    for i in range(nv):
        mapping[used[i]] = i

    facesOut = []
    for f in mesh.faces:
        facesOut.append([mapping[i] for i in f])

    return mesh._replace(vertices=vertsOut, faces=facesOut)


def bakeDownMorph(baseVerts, morphVerts, complexes, log=None):
    from pydeltamesh.mesh.subd import (
        adjustVertexData, interpolatePerVertexData
    )

    if len(complexes) == 0:
        return morphVerts

    if log:
        log("Subdividing for baking...")

    n = len(baseVerts)

    verts = baseVerts
    for i in range(len(complexes) - 1):
        verts = interpolatePerVertexData(verts, complexes[i])
    verts = adjustVertexData(verts, complexes[-1])

    if log:
        log("Subdivided %d times." % len(complexes))

    return baseVerts + morphVerts[: n] - verts[: n]


def makeBaseTargets(name, baseMesh, displacements, used, log=None):
    from uuid import uuid4
    from pydeltamesh.fileio.pmd import Deltas, MorphTarget

    if log:
        log("Finding deltas for level 0")

    vertsByGroup = {}
    for i in range(len(baseMesh.faces)):
        f = baseMesh.faces[i]
        g = baseMesh.faceGroups[i]
        vertsByGroup.setdefault(g, set()).update(f)

    targets = []
    nrDeltas = 0

    for actor in vertsByGroup:
        verts = sorted(vertsByGroup[actor])
        idcs = []
        vecs = []

        for v in verts:
            if v in used:
                d = displacements[v]
                if norm(d) > 1e-5:
                    idcs.append(v)
                    vecs.append(d)

        if len(idcs):
            nrDeltas += len(idcs)
            offset = verts[0]
            num = len(verts)
            deltas = Deltas(num, [i - offset for i in idcs], vecs)
            key = str(uuid4())
            targets.append(MorphTarget(name, actor, key, deltas, {}))

    if log:
        log("Found a total of %d deltas for %d actors." % (
            nrDeltas, len(targets)
        ))

    return targets


def makeSubdTarget(name, baseSubd0, morph, complexes, log=None):
    from uuid import uuid4
    from pydeltamesh.mesh.subd import (
        interpolatePerVertexData, subdivideTopology
    )
    from pydeltamesh.fileio.pmd import Deltas, MorphTarget

    actor = "BODY"
    key = str(uuid4())
    baseDeltas = Deltas(len(morph.vertices), [], [])
    subdDeltas = {}

    subdLevel = len(complexes)
    verts = baseSubd0.vertices

    for level in range(1, subdLevel + 1):
        if log:
            log("Subdividing morph for level %d..." % level)

        verts = interpolatePerVertexData(verts, complexes[level - 1])
        faces = subdivideTopology(complexes[level - 1])

        morphedVerts = bakeDownMorph(
            verts, morph.vertices, complexes[level:], log=log
        )

        if log:
            log("Computing vertex normals...")
        normals = vertexNormals(verts, faces)

        if log:
            log("Finding deltas for level %d..." % level)

        deltas, displacements = findSubdDeltas(verts, morphedVerts, normals)
        subdDeltas[level] = deltas

        if log:
            log("Found %d deltas." % len(deltas.indices))

        if level < subdLevel:
            verts[deltas.indices] += displacements

    return MorphTarget(name, actor, key, baseDeltas, subdDeltas)


def findSubdDeltas(baseVertices, morphedVertices, normals):
    from pydeltamesh.fileio.pmd import Deltas

    diffs = morphedVertices[: len(baseVertices)] - baseVertices
    dists = _np.sqrt(_np.sum(diffs * diffs, axis=1))
    idcs = _np.where(dists > 1e-5)[0]
    diffsAlongNormals = _np.sum(diffs[idcs] * normals[idcs], axis=1)
    deltaVectors = diffsAlongNormals[:, None] * [1, 0, 0]
    displacements = diffsAlongNormals[:, None] * normals[idcs]

    deltas = Deltas(len(baseVertices), idcs, deltaVectors)

    return deltas, displacements


def vertexNormals(vertices, faces):
    vs = _np.array(vertices)
    fs = _np.array(faces)
    ns = _np.zeros_like(vs)

    for i in range(fs.shape[1]):
        ps = fs[:, i - 2]
        qs = fs[:, i - 1]
        rs = fs[:, i]

        ns[qs] += _np.cross(vs[qs] - vs[ps], vs[rs] - vs[qs])

    ds = _np.sqrt(_np.sum(ns * ns, axis=1)).reshape(-1, 1)
    return ns / ds


def norm(v):
    return sum(x * x for x in v)**0.5


def writeInjectionPoseFile(
    fp, name, targets, pmdPath=None, postTransform=False
):
    from pydeltamesh.fileio.poserFile import PoserFile

    if pmdPath is None:
        pmdPath = ':Runtime:libraries:Pose:%s.pmd' % name

    source = PoserFile(fileTemplate.splitlines())
    root = source.root

    next(root.select('injectPMDFileMorphs')).rest = pmdPath
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

        if postTransform:
            next(targetNode.select('afterBend')).rest = '1'

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
            afterBend 0
            }
        }
    }
'''


if __name__ == '__main__':
    import sys
    from os.path import abspath, dirname

    sys.path.append(dirname(dirname(abspath(__file__))))

    args = parseArguments()

    basepath = args.basepath
    weldedpath = args.weldedpath
    morphpath = args.morphpath
    morphname = args.morphname

    log = print if args.verbose else None

    run(basepath, weldedpath, morphpath, morphname, log)
