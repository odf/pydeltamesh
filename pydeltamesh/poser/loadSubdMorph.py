def loadSubdMorph(name=None):
    import os.path
    import numpy as np
    import poser
    from pydeltamesh.makeSubdivisionMorph import Mesh, loadMesh

    chooser = poser.DialogFileChooser(poser.kDialogFileChooserOpen)
    chooser.Show()
    path = chooser.Path()
    morph = loadMesh(path, verbose=True)

    if name is None:
        name = os.path.splitext(os.path.split(path)[1])[0]

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

    print("Figure mesh has %d vertices and %d faces" % (
        len(welded.vertices), len(welded.faces)
    ))



if __name__ == '__main__':
    import sys
    from os.path import abspath, dirname

    libdir = dirname(dirname(dirname(abspath(__file__))))

    if not libdir in sys.path:
        sys.path.append(libdir)
