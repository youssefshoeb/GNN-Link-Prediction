from dataset import generator
import math
import numpy as np
import csv

directories = {"train": './data/raw/gnnet-ch21-dataset-train',
               "val": './data/raw/gnnet-ch21-dataset-validation',
               "test": './data/raw/gnnet-ch21-dataset-test-with-labels'}


class Welford(object):
    """ Implements Welford's algorithm for computing a running mean, std, and standard error
    """

    def __init__(self, lst=None):
        self.k = 0
        self.M = 0
        self.S = 0

        self.__call__(lst)

    def update(self, x):
        if x is None or isinstance(x, np.str_):
            return
        self.k += 1
        newM = self.M + (x - self.M) * 1. / self.k
        newS = self.S + (x - self.M) * (x - newM)
        self.M, self.S = newM, newS

    def consume(self, lst):
        lst = iter(lst)
        for x in lst:
            self.update(x)

    def __call__(self, x):
        if hasattr(x, "__iter__"):
            self.consume(x)
        else:
            self.update(x)

    @property
    def mean(self):
        return self.M

    @property
    def standard_error(self):
        if self.k == 0:
            return 0
        return self.std / math.sqrt(self.k)

    @property
    def std(self):
        if self.k == 1:
            return 0
        return math.sqrt(self.S / (self.k - 1))

    def __repr__(self):
        return "<Welford: {} +- {}>".format(self.mean, self.std)


class stats():
    def __init__(self, features):
        self.mean_dict = dict()
        path_mean_dict = dict((key, Welford()) for key, _ in features['paths'].items())
        link_mean_dict = dict((key, Welford()) for key, _ in features['links'].items())
        self.mean_dict.update(path_mean_dict)
        self.mean_dict.update(link_mean_dict)
        self.max_dict = dict((key, 0) for key, _ in self.mean_dict.items())
        self.min_dict = dict((key, np.inf) for key, _ in self.mean_dict.items())

    def process(self, features):
        for key, _ in features["paths"].items():
            self.mean_dict[key]((features['paths'][key]))

            if not isinstance(features['paths'][key][0], np.str_):
                if self.max_dict[key] < max(features['paths'][key]):
                    self.max_dict[key] = max(features['paths'][key])

                if self.min_dict[key] > min(features['paths'][key]):
                    self.min_dict[key] = min(features['paths'][key])

        for key, _ in features["links"].items():
            self.mean_dict[key]((features['links'][key]))

            if not isinstance(features['links'][key][0], np.str_):
                if self.max_dict[key] < max(features['links'][key]):
                    self.max_dict[key] = max(features['links'][key])

                if self.min_dict[key] > min(features['links'][key]):
                    self.min_dict[key] = min(features['links'][key])

    def save(self, filename):
        header = ["Feature", "Max", "Min", "Mean", "STD", "SE"]

        print(f'Saving...{filename}')
        with open(filename, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(header)

            # write the data
            for k, v in self.mean_dict.items():
                writer.writerow([k, self.max_dict[k], self.min_dict[k], v.mean, v.std, v.standard_error])

    def process_dataset(self, filename):
        for features, _ in generator(filename):
            self.process(features)


def process_dataset(filename, stats):
    for features, _ in generator(filename):
        for stat in stats:
            stat.process(features)


if __name__ == '__main__':
    # initialize stats
    for features, _ in generator(directories['train']):
        all_stats = stats(features)
        val_stats = stats(features)
        test_stats = stats(features)
        break

    # Process training set stats
    process_dataset(directories['train'], [all_stats])

    # Save training_set stats
    all_stats.save('training_set_stats.csv')

    # Process validation_set stats
    process_dataset(directories['val'], [all_stats, val_stats])

    # Save validation_set stats
    val_stats.save('validation_set_stats.csv')

    # Process test_set stats
    process_dataset(directories['test'], [all_stats, test_stats])

    # Save validation_set stats
    test_stats.save('test_set_stats.csv')

    # Save dataset stats
    all_stats.save('dataset_stats.csv')
