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
        propertyName = read_str(fp)
        nextPropertyPos = fp.tell() + size - 4

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


def print_deltas(deltas):
    for i in range(len(deltas.indices)):
        k = deltas.indices[i]
        x, y, z = deltas.vectors[i]
        print("%7d    %g  %g  %g" % (k, x, y, z))


def read_uint(fp):
    return struct.unpack("!I", fp.read(4))[0]


def read_str(fp):
    strlen = ord(fp.read(1))
    return fp.read(strlen).decode()


def read_vector3d(fp):
    return struct.unpack("!fff", fp.read(12))


if __name__ == "__main__":
    import sys

    with open(sys.argv[1], "rb") as fp:
        targets = read_pmd(fp)

        for target in targets:
            print(target)
            print()
