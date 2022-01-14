def loadSubdMorph(name=None):
    import os.path
    import poser
    from pydeltamesh.fileio.pmd import write_pmd
    from pydeltamesh.poser import poserUtils
    from pydeltamesh.makeSubdivisionMorph import (
        loadMesh, makeTargets, usedVertices, writeInjectionPoseFile
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

    unwelded = poserUtils.getUnweldedBaseMesh(figure)
    welded = poserUtils.getWeldedBaseMesh(figure)
    used = usedVertices(welded)

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
