import numpy as np


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
                fv.append(v + (len(vertices) if v < 0 else -1))
                ft.append(vt + (len(texverts) if vt < 0 else -1))
                fn.append(vn + (len(normals) if vn < 0 else -1))

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


def save(fp, mesh, mtlpath = None, writeNormals=True):
    if mtlpath is not None:
        with open(mtlpath, "w") as f:
            savematerials(f, mesh["materials"])
        fp.write("mtllib %s\n" % mtlpath)

    for v in mesh["vertices"]:
        fp.write("v %.8f %.8f %.8f\n" % tuple(v))
    if writeNormals:
        for v in mesh["normals"]:
            fp.write("vn %.8f %.8f %.8f\n" % tuple(v))
    for v in mesh["texverts"]:
        fp.write("vt %.8f %.8f\n" % tuple(v))

    for mat in mesh["materials"]:
        fp.write("usemtl %s\n" % mat)

    last_object = last_group = last_material = last_smoothinggroup = None

    for f in mesh["faces"]:
        if f["object"] != last_object:
            last_object = f["object"]
            fp.write("o %s\n" % last_object)
        if f["group"] != last_group:
            last_group = f["group"]
            fp.write("g %s\n" % last_group)
        if f["material"] != last_material:
            last_material = f["material"]
            fp.write("usemtl %s\n" % last_material)
        if f["smoothinggroup"] != last_smoothinggroup:
            last_smoothinggroup = f["smoothinggroup"]
            fp.write("s %s\n" % last_smoothinggroup)

        v = f["vertices"]
        vn = f["normals"]
        vt = f["texverts"]

        fp.write("f")
        for i in range(max(len(v), len(vn), len(vt))):
            fp.write(" %s/%s/%s" % (
                v[i] + 1 if i < len(v) else "",
                vt[i] + 1 if i < len(vt) else "",
                vn[i] + 1 if writeNormals and i < len(vn) else "",
            ))
        fp.write("\n")


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


def savematerials(fp, materials):
    for mat in materials:
        fp.write("newmtl %s\n" % mat)

        for line in materials[mat]:
            fp.write("%s\n" % line)
