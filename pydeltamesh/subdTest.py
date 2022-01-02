def squeeze(mesh):
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


if __name__ == "__main__":
    import math
    import sys
    from os.path import dirname

    sys.path.append(dirname(dirname(__file__)))

    from pydeltamesh.io import obj
    from pydeltamesh.mesh import subd
    from pydeltamesh.mesh.match import match, topology
    from pydeltamesh.util.optimize import minimumWeightAssignment

    with open(sys.argv[1]) as fp:
        baseMesh = obj.load(fp, sys.argv[1])
        print(
            "Loaded base mesh with %d vertices and %d faces" % (
                len(baseMesh.vertices), len(baseMesh.faces)
            )
        )

    with open(sys.argv[2]) as fp:
        subdMesh = obj.load(fp, sys.argv[2])
        print(
            "Loaded subd mesh with %d vertices and %d faces" % (
                len(subdMesh.vertices), len(subdMesh.faces)
            )
        )

    subdLevel = int(math.log2(len(subdMesh.faces) / len(baseMesh.faces)) / 2)

    print("Subdividing the base mesh %d times..." % subdLevel)
    meshA = baseMesh
    for i in range(subdLevel):
        meshA = subd.subdivideMesh(meshA)
        print(
            "Subdivided mesh has %d vertices and %d faces" % (
                len(meshA.vertices), len(meshA.faces)
            )
        )

    print("Removing unconnected vertices from the subdivision mesh...")
    meshB = squeeze(subdMesh)
    nrRemaining = len(meshB.vertices)
    nrRemoved = len(subdMesh.vertices) - nrRemaining
    print("Removed %d vertices, %d remaining." % (nrRemoved, nrRemaining))

    print("Analyzing the topology for the subdivided base mesh...")
    topoA = topology([ f.vertices for f in meshA.faces ], meshA.vertices)

    print("Analyzing the topology for the cleaned up subdivision mesh...")
    topoB = topology([ f.vertices for f in meshB.faces ], meshB.vertices)

    print("Comparing the topologies...")
    mapping = match(topoA, topoB, minimumWeightAssignment)

    nrMapped = len(list(mapping))
    nrMoved = len([u for u, v in mapping if u != v])
    print("Mapped %d vertices, moved %d" % (nrMapped, nrMoved))
