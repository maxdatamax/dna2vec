import random
import string
# import resource
import logbook
import arrow
import numpy as np
import os
import sys
import time

def split_Xy(df, y_colname='label'):
    X = df.drop([y_colname], axis=1)
    y = df[y_colname]
    return (X, y)

def shuffle_tuple(tup, rng):
    lst = list(tup)
    rng.shuffle(lst)
    return tuple(lst)

def shuffle_dataframe(df, rng):
    """
    this does NOT do in-place shuffling
    """
    return df.reindex(rng.permutation(df.index))

def random_str(N):
    return ''.join(random.SystemRandom().choice(string.ascii_lowercase + string.ascii_uppercase + string.digits) for _ in range(N))

def memory_usage():
    # return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1E6
    return 1234567

def estimate_bytes(filenames):
    return sum([os.stat(f).st_size for f in filenames])

def get_output_fileroot(dirpath, name, postfix):
    return '{}/{}-{}-{}-{}'.format(
        dirpath,
        name,
        arrow.utcnow().format('YYYYMMDD-HHmm'),
        postfix,
        random_str(3))

class Tee(object):
    def __init__(self, fptr):
        self.file = fptr

    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, exception_type, exception_value, traceback):
        sys.stdout = self.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

class Benchmark():
    def __init__(self):
        self.time_wall_start = time.time()
        self.time_cpu_start = time.clock()
        self.logger = logbook.Logger(self.__class__.__name__)

    def diff_time_wall_secs(self):
        return (time.time() - self.time_wall_start)

    def print_time(self, label=''):
        self.logger.info("%s wall=%.3fm cpu=%.3fm" % (
            label,
            self.diff_time_wall_secs() / 60.0,
            (time.clock() - self.time_cpu_start) / 60.0,
        ))