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


def run(unweldedBasePath, weldedBasePath, morphPath, name, verbose):
    import os.path
    from pydeltamesh.fileio.pmd import write_pmd

    if name is None:
        name = os.path.splitext(os.path.split(morphPath)[1])[0]

    unwelded = loadMesh(unweldedBasePath, verbose)
    weldedRaw = loadMesh(weldedBasePath, verbose)
    morph = loadMesh(morphPath, verbose)

    used = usedVertices(morph)
    welded = expandNumbering(weldedRaw, sorted(used))

    targets = makeTargets(name, unwelded, welded, morph, used, verbose=True)

    with open("%s.pmd" % name, "wb") as fp:
        write_pmd(fp, targets)

    with open("%s.pz2" % name, "w") as fp:
        writeInjectionPoseFile(fp, name, targets)


def makeTargets(
    name, unwelded, welded, morph, used, skipSubd=False, verbose=False
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
        welded.vertices, morph.vertices, complexes, verbose
    )
    weldedMorphed = welded._replace(vertices=vertsMorphed)
    targets = makeBaseTargets(name, unwelded, weldedMorphed, used, verbose)

    if not skipSubd:
        targets.append(makeSubdTarget(
            name, weldedMorphed, morph, complexes, verbose
        ))

    return targets


def loadMesh(path, verbose=False):
    if verbose:
        print("Loading mesh from %s..." % path)

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
                ns = [ int(s[: s.find('/')]) for s in fields[1:] ]
                vs = [ n - 1 if n > 0 else len(vertices) + n for n in ns]
                faces.append(vs)
                faceGroups.append(group)
            elif fields[0] == 'g':
                group = fields[1]
            elif fields[0] == 'v':
                vertices.append([float(s) for s in fields[1:]])

    if verbose:
        print("Loaded mesh with %d vertices and %d faces." % (
            len(vertices), len(faces)
        ))

    return Mesh(_np.array(vertices), faces, faceGroups)


def usedVertices(mesh):
    used = set()
    for f in mesh.faces:
        used.update(f)
    return used


def expandNumbering(mesh, used):
    nv = len(mesh.vertices)
    vertsOut = _np.zeros((used[nv - 1] + 1, 3))
    for i in range(nv):
        vertsOut[used[i]] = mesh.vertices[i]

    facesOut = []
    for f in mesh.faces:
        facesOut.append([used[i] for i in f])

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


def makeBaseTargets(name, baseMesh, morphedMesh, used, verbose=False):
    from uuid import uuid4
    from pydeltamesh.fileio.pmd import Deltas, MorphTarget

    if verbose:
        print("Finding deltas for level 0")

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
                d = morphedMesh.vertices[v] - baseMesh.vertices[v]
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

    if verbose:
        print("Found a total of %d deltas for %d actors." % (
            nrDeltas, len(targets)
        ))

    return targets


def makeSubdTarget(name, baseSubd0, morph, complexes, verbose=False):
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
        if verbose:
            print("Subdividing morph for level %d..." % level)

        verts = interpolatePerVertexData(verts, complexes[level - 1])
        faces = subdivideTopology(complexes[level - 1])

        morphedVerts = bakeDownMorph(
            verts, morph.vertices, complexes[level:], verbose
        )

        if verbose:
            print("Computing vertex normals...")
        normals = vertexNormals(verts, faces)

        if verbose:
            print("Finding deltas for level %d..." % level)

        deltas, displacements = findSubdDeltas(verts, morphedVerts, normals)
        subdDeltas[level] = deltas

        if verbose:
            print("Found %d deltas." % len(deltas.indices))

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


def writeInjectionPoseFile(fp, name, targets, pmdPath=None):
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
    from os.path import abspath, dirname

    sys.path.append(dirname(dirname(abspath(__file__))))

    args = parseArguments()

    basepath = args.basepath
    weldedpath = args.weldedpath
    morphpath = args.morphpath
    morphname = args.morphname
    verbose = args.verbose

    run(basepath, weldedpath, morphpath, morphname, verbose)
