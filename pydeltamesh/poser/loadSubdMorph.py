def loadSubdMorph(name=None):
    import os.path
    import numpy as np
    import poser
    from pydeltamesh.fileio.pmd import write_pmd
    from pydeltamesh.makeSubdivisionMorph import (
        Mesh, loadMesh, makeTargets, usedVertices, writeInjectionPoseFile
    )

    chooser = poser.DialogFileChooser(poser.kDialogFileChooserOpen)
    chooser.Show()
    path = chooser.Path()
    morph = loadMesh(path, verbose=True)

    if name is None:
        name = os.path.splitext(os.path.split(path)[1])[0]

    dir = os.path.dirname(path)

    scene = poser.Scene()
    figure = scene.CurrentFigure()

    geom = figure.SubdivGeometry()

    verts = geom.Vertices()
    polys = geom.Polygons()
    sets = geom.Sets()

    welded = Mesh(
        vertices=np.array([[v.X(), v.Y(), v.Z()] for v in verts]),
        faces=[sets[p.Start(): p.Start() + p.NumVertices()] for p in polys],
        faceGroups=[p.Groups()[0] for p in polys]
    )

    used = usedVertices(welded)

    geom, actors, actorVertexIdcs = figure.UnimeshInfo()

    vertices = []
    faces = []
    faceGroups = []

    for actor in actors:
        geom = actor.Geometry()
        verts = geom.Vertices()
        polys = geom.Polygons()
        sets = geom.Sets()

        offset = len(vertices)

        vertices.extend([[v.X(), v.Y(), v.Z()] for v in verts])
        faces.extend([
            [sets[p.Start() + k] + offset for k in range(p.NumVertices())]
            for p in polys
        ])
        faceGroups.extend([p.Groups()[0] for p in polys])

    unwelded = Mesh(np.array(vertices), faces, faceGroups)

    targets = makeTargets(name, unwelded, welded, morph, used, verbose=True)

    with open(os.path.join(dir, "%s.pmd" % name), "wb") as fp:
        write_pmd(fp, targets)

    with open(os.path.join(dir, "%s.pz2" % name), "w") as fp:
        writeInjectionPoseFile(fp, name, targets)


if __name__ == '__main__':
    import sys
    from os.path import abspath, dirname

    libdir = dirname(dirname(dirname(abspath(__file__))))

    if not libdir in sys.path:
        sys.path.append(libdir)
