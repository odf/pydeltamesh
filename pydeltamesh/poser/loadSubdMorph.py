def loadSubdMorph(name=None):
    import os.path
    import numpy as np
    import poser
    from pydeltamesh.makeSubdivisionMorph import (
        Mesh, loadMesh, makeTargets, usedVertices
    )

    def saveObj(path, mesh):
        lines = []

        for v in mesh.vertices:
            lines.append("v %.8f %.8f %.8f\n" % tuple(v))

        for f in mesh.faces:
            lines.append("f")
            for i in range(len(f)):
                lines.append(" %s//" % (f[i] + 1))
            lines.append("\n")

        with open(path, "w") as fp:
            fp.write("".join(lines))


    chooser = poser.DialogFileChooser(poser.kDialogFileChooserOpen)
    chooser.Show()
    path = chooser.Path()
    morph = loadMesh(path, verbose=True)

    if name is None:
        name = os.path.splitext(os.path.split(path)[1])[0]

    dir = os.path.dirname(path)

    scene = poser.Scene()
    figure = scene.CurrentFigure()

    geom, actors, actorVertexIdcs = figure.UnimeshInfo()

    verts = geom.Vertices()
    polys = geom.Polygons()
    sets = geom.Sets()

    welded = Mesh(
        vertices=np.array([[v.X(), v.Y(), v.Z()] for v in verts]),
        faces=[sets[p.Start(): p.Start() + p.NumVertices()] for p in polys],
        faceGroups=[p.Groups()[0] for p in polys]
    )

    used = usedVertices(welded)

    saveObj(os.path.join(dir, "x-welded.obj"), welded)

    vertices = np.zeros_like(welded.vertices)
    faces = []
    faceGroups = []

    for i, actor in enumerate(actors):
        geom = actor.Geometry()
        verts = geom.Vertices()
        polys = geom.Polygons()
        sets = geom.Sets()
        idcs = actorVertexIdcs[i]

        for k in range(len(verts)):
            v = verts[k]
            vertices[idcs[k]] = [v.X(), v.Y(), v.Z()]

        for p in polys:
            f = sets[p.Start(): p.Start() + p.NumVertices()]
            faces.append([idcs[k] for k in f])

        faceGroups.extend([p.Groups()[0] for p in polys])

    unwelded = Mesh(np.array(vertices), faces, faceGroups)

    saveObj(os.path.join(dir, "x-unwelded.obj"), unwelded)

    targets = makeTargets(name, unwelded, welded, morph, used, True)



if __name__ == '__main__':
    import sys
    from os.path import abspath, dirname

    libdir = dirname(dirname(dirname(abspath(__file__))))

    if not libdir in sys.path:
        sys.path.append(libdir)
