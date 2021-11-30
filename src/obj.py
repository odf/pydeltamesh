import numpy as np


# -- Code for reading an .obj file

def load(fp, path=None):
    import os.path
    dir = None if path is None else os.path.dirname(os.path.abspath(path))

    vertices = []
    normals = []
    texverts = []
    faces = []
    materials = {}

    obj = None
    group = None
    material = None
    smoothinggroup = None

    for rawline in fp.readlines():
        line = rawline.strip()

        if len(line) == 0 or line[0] == '#':
            continue

        fields = line.split()
        cmd = fields[0]
        pars = fields[1:]

        if cmd == 'mtllib':
            mtlpath = pars[0]
            if os.path.dirname(mtlpath) == '' and dir is not None:
                mtlpath = os.path.join(dir, mtlpath)

            with open(mtlpath) as f:
                loadmaterials(f, materials)
        elif cmd == 'v':
            vertices.append([float(pars[0]), float(pars[1]), float(pars[2])])
        elif cmd == 'vt':
            texverts.append([float(pars[0]), float(pars[1])])
        elif cmd == 'vn':
            normals.append([float(pars[0]), float(pars[1]), float(pars[2])])
        elif cmd == 'o':
            obj = pars[0]
        elif cmd == 'g':
            group = pars[0]
        elif cmd == 'usemtl':
            material = pars[0]
        elif cmd == 's':
            smoothinggroup = pars[0]
        elif cmd == 'f':
            fv = []
            ft = []
            fn = []

            for i in range(len(pars)):
                v, vt, vn = map(int, (pars[i] + '/0/0').split('/')[:3])
                fv.append(v + (len(vertices) + 1 if v < 0 else 0))
                ft.append(vt + (len(texverts) + 1 if vt < 0 else 0))
                fn.append(vn + (len(normals) + 1 if vn < 0 else 0))

            faces.append({
                'vertices': fv,
                'texverts': ft,
                'normals': fn,
                'object': obj,
                'group': group,
                'material': material,
                'smoothinggroup': smoothinggroup
            })

    return {
        'vertices': np.array(vertices),
        'normals': np.array(normals),
        'texverts': np.array(texverts),
        'faces': faces,
        'materials': materials
    }


def loadmaterials(fp, target):
    material = None

    for rawline in fp.readlines():
        line = rawline.strip()

        if len(line) == 0 or line[0] == '#':
            continue

        fields = line.split()

        if fields[0] == 'newmtl':
            material = fields[1]
            target.setdefault(material, [])
        else:
            target[material].append(line)
