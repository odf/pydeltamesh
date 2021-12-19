def run():
    from uuid import uuid4

    from pmd import Deltas, MorphTarget, write_pmd
 
    args = parseArguments()

    topoBase, base = loadAndProcessMesh(args.basepath, args.verbose)
    topoMorph, morph = loadAndProcessMesh(args.morphpath, args.verbose)

    baseVerts = base["vertices"]
    morphVerts = morph["vertices"]

    morphDeltas = extractDeltas(baseVerts, morphVerts, args.verbose)

    name = args.morphname or "morph"
    actor = "BODY"
    uuid = str(uuid4())

    if args.subdivisionLevel == 0:
        deltas = morphDeltas
        subd_deltas = {}
    else:
        deltas = Deltas(len(baseVerts), [], [])
        subd_deltas = { args.subdivisionLevel: morphDeltas }

    target = MorphTarget(name, actor, uuid, deltas, subd_deltas)

    if args.verbose:
        print("Writing the PMD...")

    with open("%s.pmd" % name, "wb") as fp:
        write_pmd(fp, [target])

    if args.verbose:
        print("Wrote the PMD.")


def loadAndProcessMesh(path, verbose=False):
    import obj
    from match import topology

    if verbose:
        print("Loading mesh from %s..." % path)

    with open(path) as fp:
        data = obj.load(fp, path)

    if verbose:
        print("Loaded mesh with %d vertices and %d faces." % (
            len(data["vertices"]), len(data["faces"])
        ))
        print("Analysing the topology...")

    topo = topology(
        [ f["vertices"] for f in data["faces"] ],
        data["vertices"]
    )

    if verbose:
        print("Analysed %d connected mesh parts." % len(topo))

    return topo, data


def extractDeltas(verticesBase, verticesMorph, verbose=False):
    from pmd import Deltas

    if verbose:
        print("Computing deltas...")

    deltaIndices = []
    deltaVectors = []

    for i in range(len(verticesBase)):
        d = verticesMorph[i] - verticesBase[i]
        if any(d > 1e-6):
            deltaIndices.append(i)
            deltaVectors.append(d)

    if verbose:
        print("Computed %d deltas." % len(deltaIndices))

    return Deltas(len(verticesBase), deltaIndices, deltaVectors)


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
