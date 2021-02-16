import fastann.target.release.fastann as fa
import random


def make_indices(dimensions):
    indices = []
    indices.append(fa.BruteForceIndex())
    indices.append(fa.BPForestIndex(dimensions, 6, -1))
    indices.append(fa.HnswIndex(10,10000000,200,300,20,"angular",200,False))

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
