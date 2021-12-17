import codecs
import struct


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

    deltaIndices = []
    deltaVectors = []

    for pos in dataStarts:
        fp.seek(pos)
        indices, vectors = read_deltas(fp)
        deltaIndices.append(indices)
        deltaVectors.append(vectors)

    while True:
        dummy = fp.read(4)
        if len(dummy) < 4:
            break

        size = read_uint(fp)
        propertyName = read_str(fp)
        nextPropertyPos = fp.tell() + size - 4

        if propertyName == "SDLEVELS":
            nrTargets = read_uint(fp)

            for i in range(nrTargets):
                uuid = read_str(fp)
                nrLevels = read_uint(fp)
                print("target %s has %d subd levels" % (uuid, nrLevels))

                for j in range(nrLevels):
                    level = read_uint(fp)
                    nrDeltas = read_uint(fp)
                    pos = read_uint(fp)

        fp.seek(nextPropertyPos)


def read_deltas(fp):
    nrIndices = read_uint(fp)

    indices = []
    vectors = []
    for i in range(nrIndices):
        indices.append(read_uint(fp))
        vectors.append(read_vector3d(fp))

    return indices, vectors


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
