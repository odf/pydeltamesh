import numpy as np

from collections import namedtuple


Face = namedtuple(
    "Face",
    [
        "vertices",
        "texverts",
        "normals",
        "object",
        "group",
        "material",
        "smoothinggroup"
    ]
)

Mesh = namedtuple(
    "Mesh",
    [ "vertices", "normals", "texverts", "faces", "materials"]
)


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
                idcs = pars[i].split('/')
                v = int(idcs[0]) if len(idcs) > 0 and idcs[0] else 0
                vt = int(idcs[1]) if len(idcs) > 1 and idcs[1] else 0
                vn = int(idcs[2]) if len(idcs) > 2 and idcs[2] else 0
                fv.append(v + (len(vertices) if v < 0 else -1))
                ft.append(vt + (len(texverts) if vt < 0 else -1))
                fn.append(vn + (len(normals) if vn < 0 else -1))

            faces.append(Face(
                vertices=fv,
                texverts=ft,
                normals=fn,
                object=obj,
                group=group,
                material=material,
                smoothinggroup=smoothinggroup
            ))

    return Mesh(
        vertices=np.array(vertices),
        normals=np.array(normals),
        texverts=np.array(texverts),
        faces=faces,
        materials=materials
    )


def save(fp, mesh, mtlpath = None, writeNormals=True):
    lines = []

    if mtlpath is not None:
        with open(mtlpath, "w") as f:
            savematerials(f, mesh.materials)
        lines.append("mtllib %s\n" % mtlpath)

    for v in mesh.vertices:
        lines.append("v %.8f %.8f %.8f\n" % tuple(v))

    for v in mesh.texverts:
        lines.append("vt %.8f %.8f\n" % tuple(v))

    if writeNormals and mesh.normals:
        for v in mesh.normals:
            lines.append("vn %.8f %.8f %.8f\n" % tuple(v))

    for mat in mesh.materials:
        lines.append("usemtl %s\n" % mat)

    last_object = last_group = last_material = last_smoothinggroup = None

    for f in mesh.faces:
        if f.object != last_object:
            last_object = f.object
            lines.append("o %s\n" % last_object)
        if f.group != last_group:
            last_group = f.group
            lines.append("g %s\n" % last_group)
        if f.material != last_material:
            last_material = f.material
            lines.append("usemtl %s\n" % last_material)
        if f.smoothinggroup != last_smoothinggroup:
            last_smoothinggroup = f.smoothinggroup
            lines.append("s %s\n" % last_smoothinggroup)

        v = f.vertices
        vn = f.normals
        vt = f.texverts

        lines.append("f")
        for i in range(len(v)):
            lines.append(" %s/%s/%s" % (
                v[i] + 1 if v else "",
                vt[i] + 1 if vt else "",
                vn[i] + 1 if writeNormals and vn else "",
            ))
        lines.append("\n")

    fp.write("".join(lines))


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
