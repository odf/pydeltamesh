import codecs
import struct

from collections import namedtuple


Deltas = namedtuple("Deltas", ["indices", "vectors"])


def read_pmd(fp):
    fp.seek(0)

    magic = codecs.decode(fp.read(4))
    if magic != "PZMD":
        raise IOError("Not a Poser PMD file")

    fp.seek(16)
    nrTargets = read_uint(fp)

    morphnames = []
    actornames = []
    nrDeltas = []
    dataStarts = []
    uuids = []
    uuidToIndex = {}

    for i in range(nrTargets):
        morphnames.append(read_str(fp))
        actornames.append(read_str(fp))
        nrDeltas.append(read_uint(fp))
        dataStarts.append(read_uint(fp))

    if min(dataStarts) > fp.tell():
        for i in range(nrTargets):
            uuid = read_str(fp)
            uuidToIndex[uuid] = i
            uuids.append(uuid)

    deltas = []

    for pos in dataStarts:
        fp.seek(pos)
        deltas.append(read_deltas(fp))

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
            subdivDeltas = [None] * len(morphnames)
            nrTargets = read_uint(fp)

            for i in range(nrTargets):
                uuid = read_str(fp)
                targetIndex = uuidToIndex[uuid]

                nrLevels = read_uint(fp)
                subdivDeltas[targetIndex] = [None] * (nrLevels + 1)

                for j in range(nrLevels):
                    level = read_uint(fp)
                    nrDeltas = read_uint(fp)
                    pos = read_uint(fp)
                    if pos > 0:
                        subdivDeltas[targetIndex][level] = pos

            for i in range(len(subdivDeltas)):
                if subdivDeltas[i] is not None:
                    for j in range(len(subdivDeltas[i])):
                        if subdivDeltas[i][j] is not None:
                            fp.seek(subdivDeltas[i][j])
                            subdivDeltas[i][j] = read_deltas(fp)

        fp.seek(nextPropertyPos)


def read_deltas(fp):
    nrIndices = read_uint(fp)

    indices = []
    vectors = []
    for i in range(nrIndices):
        indices.append(read_uint(fp))
        vectors.append(read_vector3d(fp))

    return Deltas(indices, vectors)


def print_deltas(deltas):
    for i in range(len(deltas.indices)):
        k = deltas.indices[i]
        x, y, z = deltas.vectors[i]
        print("%7d    %g  %g  %g" % (k, x, y, z))


def read_uint(fp):
    return struct.unpack("!I", fp.read(4))[0]


def read_str(fp):
    strlen = ord(fp.read(1))
    return codecs.decode(fp.read(strlen))


def read_vector3d(fp):
    return struct.unpack("!fff", fp.read(12))


if __name__ == "__main__":
    import sys

    with open(sys.argv[1], "rb") as fp:
        read_pmd(fp)
