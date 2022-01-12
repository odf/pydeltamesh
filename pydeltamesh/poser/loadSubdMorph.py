def loadSubdMorph(name=None):
    import os.path

    import poser

    from pydeltamesh.makeSubdivisionMorph import loadMesh

    chooser = poser.DialogFileChooser(poser.kDialogFileChooserOpen)
    chooser.Show()
    path = chooser.Path()

    morph = loadMesh(path, verbose=True)

    if name is None:
        name = os.path.splitext(os.path.split(path)[1])[0]


if __name__ == '__main__':
    import sys
    from os.path import abspath, dirname

    libdir = dirname(dirname(dirname(abspath(__file__))))

    if not libdir in sys.path:
        sys.path.append(libdir)
