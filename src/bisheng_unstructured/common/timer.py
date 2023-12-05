import time


class Timer(object):
    def __init__(self):
        self.tic()
        self.elapses = []

    def tic(self):
        self.tstart = time.time()

    def toc(self, reset=True, memorize=True):
        elapse = round(time.time() - self.tstart, 3)
        if memorize:
            self.elapses.append(elapse)

        if reset:
            self.tic()

    def get(self):
        n = round(sum(self.elapses), 3)
        elapses = self.elapses + [
            n,
        ]
        return elapses
