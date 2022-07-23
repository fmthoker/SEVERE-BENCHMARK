import csv
import torch
import math
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.lists = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        for _ in range(0, n):
            self.lists.append(val)

    def std(self):
        std_sum = 0.0
        for i in range(0, len(self.lists)):
            std_sum = std_sum + pow(self.lists[i]-self.avg, 2)
        std_sum = pow(std_sum / self.count, 0.5)
        return std_sum


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'a+')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'a+') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value





