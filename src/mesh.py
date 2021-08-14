import numpy as np


def loadmesh(fp):
    vertices = []
    normals = []
    texverts = []
    faces = []

    obj = '_default'
    group = '_default'
    material = '_default'
    smoothinggroup = '_default'

    for rawline in fp.readlines():
        line = rawline.strip()

        if len(line) == 0 or line[0] == '#':
            continue

        fields = line.split()
        cmd = fields[0]
        pars = fields[1:]

        if cmd == 'mtllib':
            pass # TODO read and store material definitions from file
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
                fv.append(len(vertices) + 1 - v if v < 0 else v)
                ft.append(len(texverts) + 1 - vt if vt < 0 else vt)
                fn.append(len(normals) + 1 - vn if vn < 0 else vn)

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
        'faces': faces
    }


if __name__ == '__main__':
    import sys

    with open(sys.argv[1]) as fp:
        mesh = loadmesh(fp)
        print(mesh)
