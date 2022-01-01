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


def run(args):
    from .io.pmd import print_deltas

    base = loadMesh(args.basepath, args.verbose)
    baseParts = processedByGroup(base, args.verbose)

    morph = loadMesh(args.morphpath, args.verbose)
    morphParts = processedByGroup(morph, args.verbose)

    for actor in baseParts:
        if actor in morphParts:
            deltas = findDeltas(baseParts[actor], morphParts[actor])

            if deltas:
                if args.verbose:
                    n = len(deltas.indices)
                    print("Found %d deltas for %s." % (n, actor))
                    print_deltas(deltas)


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


def processedByGroup(data, verbose=False):
    import re

    from .mesh.match import topology

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
    from .io.pmd import Deltas
    from .mesh.match import match
    from .util.optimize import minimumWeightAssignment

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


def norm(v):
    return sum(x * x for x in v)**0.5


if __name__ == '__main__':
    run(parseArguments())
