def loadSubdMorph(name, path, postTransform):
    import time
    import os

    import poser

    from pydeltamesh.fileio.pmd import write_pmd
    from pydeltamesh.poser import poserUtils
    from pydeltamesh import makeSubdivisionMorph as subd

    t0 = time.time()

    scene = poser.Scene()
    figure = scene.CurrentFigure()

    def log(s):
        print(s)
        scene.ProcessSomeEvents(5)

    morph = subd.loadMesh(path, log=log)

    if not name:
        name = os.path.splitext(os.path.split(path)[1])[0]

    unwelded = poserUtils.getUnweldedBaseMesh(figure)
    welded = poserUtils.getWeldedBaseMesh(figure)
    used = subd.usedVertices(welded)

    morph = subd.compressNumbering(morph, sorted(subd.usedVertices(morph)))
    morph = subd.expandNumbering(morph, sorted(used))

    targets = subd.makeTargets(
        name, unwelded, welded, morph, used,
        postTransform=postTransform, log=log
    )

    tempdir = poser.TempLocation()
    pmdPath = os.path.join(tempdir, "%s.pmd" % name)
    pz2Path = os.path.join(tempdir, "%s.pz2" % name)

    with open(pmdPath, "wb") as fp:
        write_pmd(fp, targets)

    with open(pz2Path, "w") as fp:
        subd.writeInjectionPoseFile(
            fp, name, targets, pmdPath=pmdPath, postTransform=postTransform
        )

    scene.LoadLibraryPose(pz2Path)
    os.remove(pmdPath)
    os.remove(pz2Path)

    print("Morph loaded in %s seconds." % (time.time() - t0))


if __name__ == '__main__':
    import sys
    from os.path import abspath, dirname
    import poser

    libdir = dirname(dirname(dirname(abspath(__file__))))

    if not libdir in sys.path:
        sys.path.append(libdir)

    textEntry = poser.DialogTextEntry(message="Morph Name")
    success = textEntry.Show()
    name = textEntry.Text() if success else None

    postTransform = poser.DialogSimple.YesNo("Make a post-transform morph?")

    chooser = poser.DialogFileChooser(
        poser.kDialogFileChooserOpen,
        message="Select an *.obj file"
    )
    if chooser.Show():
        path = chooser.Path()
        loadSubdMorph(name, path, postTransform)
