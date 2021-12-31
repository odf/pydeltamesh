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
    from uuid import uuid4

    from .io.pmd import Deltas, MorphTarget, write_pmd

    base = loadMesh(args.basepath, args.verbose)
    baseParts = processedByGroup(base, args.verbose)
    morph = loadMesh(args.morphpath, args.verbose)
    morphParts = processedByGroup(morph, args.verbose)


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
    from .mesh.match import topology

    if verbose:
        print("Splitting mesh by groups...")

    groupFaces = {}
    for f in data.faces:
        groupFaces.setdefault(f.group, []).append(f.vertices)

    if verbose:
        print("Split mesh into %d groups." % len(groupFaces))

    topologies = {}

    for group in groupFaces:
        if verbose:
            print("Analysing %s topology..." % group)

        topo = topology(groupFaces[group], data.vertices)
        topologies[group] = topo

        if verbose:
            print("Analysed %d connected parts for %s." % (len(topo), group))

    return topologies


if __name__ == '__main__':
    run(parseArguments())
