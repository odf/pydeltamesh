import struct

from collections import namedtuple


Deltas = namedtuple(
    "Deltas",
    ["numb_deltas", "indices", "vectors"]
)

MorphTarget = namedtuple(
    "MorphTarget",
    ["name", "actor", "uuid", "deltas", "subd_deltas"]
)


def read_pmd(fp):
    fp.seek(0)

    magic = fp.read(4).decode()
    if magic != "PZMD":
        raise IOError("Not a Poser PMD file")

    fp.seek(16)
    nrTargets = read_uint(fp)

    targets = []
    dataStarts = []
    uuidToIndex = {}

    for i in range(nrTargets):
        name = read_str(fp)
        actor = read_str(fp)
        numb_deltas = read_uint(fp)
        dataStarts.append(read_uint(fp))

        targets.append(
            MorphTarget(
                name=name,
                actor=actor,
                uuid=None,
                deltas=Deltas(numb_deltas, [], []),
                subd_deltas={}
            )
        )

    if min(dataStarts) > fp.tell():
        for i in range(nrTargets):
            uuid = read_str(fp)
            uuidToIndex[uuid] = i
            targets[i] = targets[i]._replace(uuid = uuid)

    for i in range(nrTargets):
        fp.seek(dataStarts[i])
        targets[i] = targets[i]._replace(
            deltas = read_deltas(fp, targets[i].deltas)
        )

    while True:
        pos = fp.tell()
        dummy = fp.read(8)
        if len(dummy) < 8:
            break
        else:
            fp.seek(pos + 4)

        size = read_uint(fp)
        nextPropertyPos = fp.tell() + size - 4
        propertyName = read_str(fp)

        if propertyName == "SDLEVELS":
            subdivDeltas = {}
            nrSubdTargets = read_uint(fp)

            for i in range(nrSubdTargets):
                uuid = read_str(fp)
                targetIndex = uuidToIndex[uuid]

                nrLevels = read_uint(fp)
                subdivDeltas[targetIndex] = {}

                for j in range(nrLevels):
                    level = read_uint(fp)
                    numb_deltas = read_uint(fp)
                    pos = read_uint(fp)
                    if pos > 0:
                        subdivDeltas[targetIndex][level] = numb_deltas, pos

            for i in subdivDeltas:
                for j in subdivDeltas[i]:
                    numb_deltas, pos = subdivDeltas[i][j]
                    template = Deltas(numb_deltas, [], [])
                    fp.seek(pos)
                    subdivDeltas[i][j] = read_deltas(fp, template)

                targets[i].subd_deltas.update(subdivDeltas[i])

        fp.seek(nextPropertyPos)

    return targets


def write_pmd(fp, targets):
    fp.seek(0)
    fp.write("PZMD".encode())
    write_uint(fp, 3)
    write_uint(fp, 0)
    write_uint(fp, 0)

    write_uint(fp, len(targets))

    dataStartPointers = []

    for target in targets:
        write_str(fp, target.name)
        write_str(fp, target.actor)
        write_uint(fp, target.deltas.numb_deltas)
        dataStartPointers.append(fp.tell())
        write_uint(fp, 0)

    if all(t.uuid for t in targets):
        for target in targets:
            write_str(fp, target.uuid)

    for i, target in enumerate(targets):
        write_uint_at(fp, dataStartPointers[i], fp.tell())
        write_deltas(fp, target.deltas)

    if any(t.subd_deltas for t in targets):
        write_uint(fp, 1)
        blockStart = fp.tell()
        write_uint(fp, 0)
        write_str(fp, "SDLEVELS")

        subdTargets = [t for t in targets if t.subd_deltas]
        write_uint(fp, len(subdTargets))

        dataStartPointers = {}

        for target in subdTargets:
            write_str(fp, target.uuid)
            write_uint(fp, len(target.subd_deltas))

            for level, deltas in target.subd_deltas.items():
                write_uint(fp, level)
                write_uint(fp, deltas.numb_deltas)
                if deltas.numb_deltas > 0:
                    dataStartPointers[target.uuid, level] = fp.tell()
                write_uint(fp, 0)

        write_uint(fp, 0)

        for target in subdTargets:
            for level, deltas in target.subd_deltas.items():
                if deltas.numb_deltas > 0:
                    pos = dataStartPointers[target.uuid, level]
                    write_uint_at(fp, pos, fp.tell())
                    write_deltas(fp, deltas)

        write_uint_at(fp, blockStart, fp.tell() - blockStart)


def read_deltas(fp, template=None):
    nrIndices = read_uint(fp)

    indices = []
    vectors = []
    for i in range(nrIndices):
        indices.append(read_uint(fp))
        vectors.append(read_vector3d(fp))

    if template:
        return template._replace(indices=indices, vectors=vectors)
    else:
        return Deltas(0, indices, vectors)


def write_deltas(fp, deltas):
    nrIndices = len(deltas.indices)
    write_uint(fp, nrIndices)

    for i in range(nrIndices):
        write_uint(fp, deltas.indices[i])
        write_vector3d(fp, deltas.vectors[i])


def print_deltas(deltas):
    for i in range(len(deltas.indices)):
        k = deltas.indices[i]
        x, y, z = deltas.vectors[i]
        print("%7d    %g  %g  %g" % (k, x, y, z))


def read_uint(fp):
    return struct.unpack("!I", fp.read(4))[0]


def read_str(fp):
    strlen = struct.unpack("!b", fp.read(1))[0]
    return fp.read(strlen).decode()


def read_vector3d(fp):
    return struct.unpack("!fff", fp.read(12))


def write_uint(fp, value):
    fp.write(struct.pack("!I", value))


def write_uint_at(fp, pos, value):
    oldpos = fp.tell()
    fp.seek(pos)
    write_uint(fp, value)
    fp.seek(oldpos)


def write_str(fp, s):
    assert(len(s) < 256)
    fp.write(struct.pack("!b", len(s)))
    fp.write(s.encode())


def write_vector3d(fp, value):
    fp.write(struct.pack("!fff", *value))


if __name__ == "__main__":
    import sys

    with open(sys.argv[1], "rb") as fp:
        targets = read_pmd(fp)

    for target in targets:
        print(target)
        print()

    with open("testOutput.pmd", "wb") as fp:
        write_pmd(fp, targets)
