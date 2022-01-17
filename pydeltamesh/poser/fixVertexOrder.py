if __name__ == '__main__':
    import sys
    from os.path import abspath, basename, dirname, exists
    import poser

    libdir = dirname(dirname(dirname(abspath(__file__))))
    if not libdir in sys.path:
        sys.path.append(libdir)

    from pydeltamesh.applyMorph import loadMesh, saveMesh, mapVertices

    scene = poser.Scene()

    def log(s):
        print(s)
        scene.ProcessSomeEvents(5)

    chooser = poser.DialogFileChooser(
        poser.kDialogFileChooserOpen,
        message="Select the unmorphed (base) mesh (*.obj)"
    )

    done = False

    if chooser.Show():
        pathBase = chooser.Path()
        meshBase = loadMesh(pathBase)
        log("Loaded %s with %d vertices and %d faces." % (
            basename(pathBase),
            len(meshBase.vertices), len(meshBase.faces)
        ))

        chooser = poser.DialogFileChooser(
            poser.kDialogFileChooserOpen,
            message="Select the morphed mesh (*.obj)"
        )

        if chooser.Show():
            pathMorphed = chooser.Path()
            meshMorphed = loadMesh(pathMorphed)
            log("Loaded %s with %d vertices and %d faces." % (
                basename(pathMorphed),
                len(meshMorphed.vertices), len(meshMorphed.faces)
            ))

            meshOut = mapVertices(meshBase, meshMorphed, log=log)

            chooser = poser.DialogFileChooser(
                poser.kDialogFileChooserSave,
                message="File to save the fixed morph to (*.obj)"
            )

            if chooser.Show():
                pathOut = chooser.Path()
                if (
                    not exists(pathOut) or
                    poser.DialogSimple.YesNo("File exists. Overwrite?")
                ):
                    saveMesh(pathOut, meshOut)
                    done = True

    message = "Script completed!" if done else "Script cancelled!"
    poser.DialogSimple.MessageBox(message)
