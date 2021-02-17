import fastann.target.release.fastann as fastann
import random


def make_indices(dimensions):
    indices = []
    indices.append(fastann.BruteForceIndex())
    indices.append(fastann.BPForestIndex(dimensions, 6, -1))
    indices.append(fastann.HnswIndex(10, 10000000, 200,
                                300, 20, "angular", 200, False))
    indices.append(fastann.PQIndex(10, 10, 4,
                              100,  "angular"))

    return indices


def make_test_data(dimensions, indices):
    for i in range(1000):
        v = [random.gauss(0, 1) for z in range(dimensions)]
        for idx in indices:
            idx.add(v, "{}".format(i))


def main():
    dimensions = 10
    indices = make_indices(dimensions)
    make_test_data(dimensions, indices)

    for idx in indices:
        idx.construct("angular")

    for idx in indices:
        v = [random.gauss(0, 1) for z in range(dimensions)]
        for item in idx.search_k(v, 10):
            print("{} {} {} {} ".format(idx.name(),
                                        item[0].idx, item[0].vectors, item[1]))


if __name__ == "__main__":
    main()
