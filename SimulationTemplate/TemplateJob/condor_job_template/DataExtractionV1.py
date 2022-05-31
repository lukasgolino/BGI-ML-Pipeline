import sys
import pandas as pd
import numpy as np
import pickle

actualdata = pd.read_csv(sys.argv[1])
actualdata = actualdata[['initial x', 'final x']]

#If data exists:
real_profile = actualdata['initial x']
real_stddev = real_profile.std()
real_profile_hist, bins = np.histogram(real_profile, bins=1024, range=(-0.028, 0.028))

distorted_profile_5k = actualdata['final x'].sample(n = 5000)
distorted_stddev_5k = distorted_profile_5k.std()
distorted_profile_hist_5k, bins2_5k = np.histogram(distorted_profile_5k, bins=1024, range=(-0.028, 0.028))

distorted_profile_10k = actualdata['final x'].sample(n = 10000)
distorted_stddev_10k = distorted_profile_10k.std()
distorted_profile_hist_10k, bins2_10k = np.histogram(distorted_profile_10k, bins=1024, range=(-0.028, 0.028))

distorted_profile_25k = actualdata['final x'].sample(n = 25000)
distorted_stddev_25k = distorted_profile_25k.std()
distorted_profile_hist_25k, bins2_25k = np.histogram(distorted_profile_25k, bins=1024, range=(-0.028, 0.028))

distorted_profile_50k = actualdata['final x'].sample(n = 50000)
distorted_stddev_50k = distorted_profile_50k.std()
distorted_profile_hist_50k, bins2_50k = np.histogram(distorted_profile_50k, bins=1024, range=(-0.028, 0.028))

distorted_profile_100k = actualdata['final x'].sample(n = 100000)
distorted_stddev_100k = distorted_profile_100k.std()
distorted_profile_hist_100k, bins2_100k = np.histogram(distorted_profile_100k, bins=1024, range=(-0.028, 0.028))

distorted_profile_200k = actualdata['final x'].sample(n = 200000)
distorted_stddev_200k = distorted_profile_200k.std()
distorted_profile_hist_200k, bins2_200k = np.histogram(distorted_profile_200k, bins=1024, range=(-0.028, 0.028))

distorted_profile_500k = actualdata['final x'].sample(n = 500000)
distorted_stddev_500k = distorted_profile_500k.std()
distorted_profile_hist_500k, bins2_500k = np.histogram(distorted_profile_500k, bins=1024, range=(-0.028, 0.028))

distorted_profile_1mil = actualdata['final x'].sample(n = 1000000)
distorted_stddev_1mil = distorted_profile_1mil.std()
distorted_profile_hist_1mil, bins2_1mil = np.histogram(distorted_profile_1mil, bins=1024, range=(-0.028, 0.028))

distorted_profile_max = actualdata['final x']
distorted_stddev_max = distorted_profile_max.std()
distorted_profile_hist_max, bins2_max = np.histogram(distorted_profile_max, bins=1024, range=(-0.028, 0.028))

with open(sys.argv[1]+'.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([real_stddev, bins, real_profile_hist, bins2_max, distorted_profile_hist_max,
                 distorted_profile_hist_5k, distorted_profile_hist_10k, distorted_profile_hist_25k,
                 distorted_profile_hist_50k, distorted_profile_hist_100k, distorted_profile_hist_200k,
                 distorted_profile_hist_500k, distorted_profile_hist_1mil], f)