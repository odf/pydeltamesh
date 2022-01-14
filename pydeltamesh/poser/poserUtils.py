def getUnweldedBaseMesh(figure):
    import numpy as np
    from pydeltamesh.makeSubdivisionMorph import Mesh

    geom, actors, _ = figure.UnimeshInfo()

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

    return Mesh(np.array(vertices), faces, faceGroups)


def getWeldedBaseMesh(figure):
    import numpy as np
    from pydeltamesh.makeSubdivisionMorph import Mesh

    nrSubdivPreview = figure.NumbSubdivLevels()
    nrSubdivRender = figure.NumbSubdivRenderLevels()

    figure.SetNumbSubdivLevels(0)
    figure.SetNumbSubdivRenderLevels(0)

    geom = figure.SubdivGeometry()

    figure.SetNumbSubdivLevels(nrSubdivPreview)
    figure.SetNumbSubdivRenderLevels(nrSubdivRender)

    verts = geom.Vertices()
    polys = geom.Polygons()
    sets = geom.Sets()

    return Mesh(
        vertices=np.array([[v.X(), v.Y(), v.Z()] for v in verts]),
        faces=[sets[p.Start(): p.Start() + p.NumVertices()] for p in polys],
        faceGroups=[p.Groups()[0] for p in polys]
    )
