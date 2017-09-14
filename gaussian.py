import numpy as np
# from matplotlib import pyplot as plt

# sumstd = 0
# sumlpstd = 0
# iteration = 1
# for i in range(iteration):
#     lg = np.random.normal(size=(2000000,))
#     # print(lg.var())
#     sumstd+=lg.var()
#
#     lp = lg.copy()
#     for j in range(1,len(lp)):
#         lp[j] = lp[j-1]*0.7 + lp[j]*0.3
#
#     for j in range(1,len(lp)):
#         lp[j] = lp[j-1]*0.7 + lp[j]*0.3
#
#     # print(lp.var())
#     sumlpstd += lp.var()

# plt.plot(lp)
# plt.show()

# print('meanofvar',sumstd/iteration)
# print('meanoflpvar',sumlpstd/iteration)
# print('extra scaling factor', sumstd/sumlpstd)

# second-order low-passed gaussian noise source. stddev is close to one
class lowpassgaussian:
    def __init__(self):
        self.first = 0
        self.second = 0

        for i in range(100):
            self.sample()

    def sample(self):
        update_factor = .4
        scaling_factor = 2.74
        stay_factor = 1-update_factor
        g = np.random.normal()
        self.first = self.first*stay_factor+g*update_factor
        self.second = self.second*stay_factor+self.first*update_factor

        return self.second * scaling_factor

# samples are still evenly drawn from [0,1], but higher low-freq component.
class lowpassuniform:
    def __init__(self):
        self.first = 0

        for i in range(2):
            self.sample()

    def sample(self):
        u = np.random.uniform()
        if np.random.uniform()<0.5: # for 20% of the time
            self.first = u
        else:
            pass
        return self.first

if __name__ == '__main__':
    lpg = lowpassgaussian()
    for p in range(10):
        gs = np.random.normal(size=100000)
        lpgs = np.array([lpg.sample() for i in range(100000)])

        print(gs.std(),lpgs.std(),gs.var(),lpgs.var())
