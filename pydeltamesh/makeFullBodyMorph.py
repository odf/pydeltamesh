def parseArguments():
    import argparse

    parser = argparse.ArgumentParser(description='Apply a morph to a mesh.')
    parser.add_argument(
        'basepath',
        type=str,
        help='an unwelded full-figure mesh in OBJ format'
    )
    parser.add_argument(
        'morphpath',
        type=str,
        help=''.join([
            'a morphed, welded or unwelded, possibly incomplete,',
            ' version of the base mesh'
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


def run(basepath, morphpath, morphname, verbose):
    from uuid import uuid4

    from pydeltamesh.io.pmd import MorphTarget, write_pmd

    name = morphname

    base = loadMesh(basepath, verbose)
    baseParts = processedByGroup(base, verbose)

    morph = loadMesh(morphpath, verbose)
    morphParts = processedByGroup(morph, verbose)

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


def writeInjectionPoseFile(fp, name, targets):
    from uuid import uuid4

    from pydeltamesh.io.poserFile import PoserFile

    morphPath = ':Runtime:libraries:Pose:%s.pmd' % name

    source = PoserFile(fileTemplate.splitlines())
    root = source.root

    next(root.select('injectPMDFileMorphs')).rest = morphPath
    next(root.select('createFullBodyMorph')).rest = name

    actor = PoserFile(actorTemplate.splitlines()).root

    targetNode = next(actor.select('actor', 'channels', 'targetGeom'))
    targetNode.rest = name
    next(targetNode.select('name')).rest = name
    next(targetNode.select('uuid')).rest = str(uuid4())

    valueOpNode = next(targetNode.select('valueOpDeltaAdd'))
    for _ in range(5):
        valueOpNode.nextSibling.unlink()
    valueOpNode.unlink()

    next(root.select('}')).prependSibling(actor)

    for target in targets:
        actor = PoserFile(actorTemplate.splitlines()).root
        next(actor.select('actor')).rest = target.actor + ":1"
        next(actor.select('actor', 'channels', 'groups')).unlink()

        targetNode = next(actor.select('actor', 'channels', 'targetGeom'))
        targetNode.rest = name
        next(targetNode.select('name')).rest = target.name
        next(targetNode.select('uuid')).rest = target.uuid
        next(targetNode.select('@name')).text = target.name
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
            min 0
            max 1
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


def norm(v):
    return sum(x * x for x in v)**0.5


if __name__ == '__main__':
    import sys
    from os.path import dirname

    sys.path.append(dirname(dirname(__file__)))

    args = parseArguments()

    basepath = args.basepath
    morphpath = args.morphpath
    morphname = args.morphname or "Morph" #TODO based default on morphpath
    verbose = args.verbose

    run(basepath, morphpath, morphname, verbose)
