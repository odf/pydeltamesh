import numpy as np


def loadmesh(fp):
    rawmesh = loadrawmesh(fp)

    return Mesh(
        rawmesh['vertices'],
        rawmesh['faces'],
        rawmesh['normals'],
        rawmesh['texverts'],
        rawmesh['materials']
    )


def loadrawmesh(fp):
    vertices = []
    normals = []
    texverts = []
    faces = []
    materials = {}

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
            # TODO if path is relative, start at .obj file path
            with open(pars[0]) as f:
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
        'faces': faces,
        'materials': materials
    }


def loadmaterials(fp, target):
    material = '_default'

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


class Mesh(object):
    def __init__(
        self, vertices, faces, normals=None, texverts=None, materials=None
    ):
        self._vertices = vertices
        self._normals = normals
        self._texverts = texverts
        self._materials = materials

        self._faceobjects = [f['object'] for f in faces]
        self._facegroups = [f['group'] for f in faces]
        self._facematerials = [f['material'] for f in faces]
        self._facesmoothinggroups = [f['smoothinggroup'] for f in faces]

        rawchambers = makechambers([f['vertices'] for f in faces])
        chamberface = rawchambers['chamberface']
        chamberindex = rawchambers['chamberindex']

        self._sigma = rawchambers['sigma']
        self._nrchambers = nc = self._sigma.shape[1]
        self._chambervertex = np.full(nc, -1, dtype=np.int32)
        self._chambertexvert = np.full(nc, -1, dtype=np.int32)
        self._chambernormal = np.full(nc, -1, dtype=np.int32)

        for i in range(self._nrchambers):
            f = faces[chamberface[i]]
            k = chamberindex[i]
            self._chambervertex[i] = f['vertices'][k]
            self._chambertexvert[i] = f['texverts'][k]
            self._chambernormal[i] = f['normals'][k]

    def write(self, fp, basename=None):
        if basename is not None and len(self._materials) > 0:
            with open('%s.mtl' % basename, 'w') as f:
                for key in self._materials:
                    f.write('newmtl %s\n' % key)
                    f.write('\n'.join(self._materials[key]))
                    f.write('\n')

        for x, y, z in self._vertices:
            fp.write('v %.8f %.8f %.8f\n' % (x, y, z))
        for x, y, z in self._normals:
            fp.write('vn %.8f %.8f %.8f\n' % (x, y, z))
        for x, y in self._texverts:
            fp.write('vt %.8f %.8f\n' % (x, y))
        for key in sorted(self._materials):
            fp.write('usemtl %s\n' % key)

        parts = {}
        for i in range(len(self._faceobjects)):
            key = (
                self._faceobjects[i],
                self._facegroups[i],
                self._facematerials[i],
                self._facesmoothinggroups[i]
            )
            parts.setdefault(key, []).append(i)


def makechambers(faces):
    edges = {}
    offset = [0]

    for i in range(len(faces)):
        f = faces[i]
        n = len(f)
        offset.append(offset[-1] + n)

        for j in range(n):
            u, v = f[j], f[(j + 1) % n]

            if edges.get((u, v)) is None:
                edges[u, v] = (i, j)
            else:
                raise RuntimeError(
                    'duplicate directed edge %s -> %s' % (u, v)
                )

    nc = 2 * len(edges)

    sop = np.full((3, nc), -1, dtype=np.int32)
    ch2face = np.full(nc, -1, dtype=np.int32)
    ch2index = np.full(nc, -1, dtype=np.int32)

    for i in range(len(faces)):
        f = faces[i]
        n = len(f)

        for j in range(n):
            jnext = (j + 1) % n
            k = 2 * (offset[i] + j)
            knext = 2 * (offset[i] + jnext)

            ch2face[k] = i
            ch2face[k + 1] = i
            ch2index[k] = j
            ch2index[k + 1] = jnext

            sop[0, k] = k + 1
            sop[0, k + 1] = k
            sop[1, k + 1] = knext
            sop[1, knext] = k + 1

            opp = edges.get((f[jnext], f[j]))

            if opp is not None:
                ix, jx = opp
                kx = 2 * (offset[ix] + jx)
                sop[2, k] = kx + 1
                sop[2, k + 1] = kx

    return {
         'sigma': sop,
         'chamberface': ch2face,
         'chamberindex': ch2index
    }


if __name__ == '__main__':
    import sys

    with open(sys.argv[1]) as fp:
        mesh = loadmesh(fp)

    for key in mesh.__dict__:
        print(key)
        print(mesh.__dict__[key])
        print()

    with open('x.obj', 'w') as fp:
        mesh.write(fp, 'x')
