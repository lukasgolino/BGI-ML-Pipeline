import sys
import pandas as pd
import numpy as np
import pickle

actualdata = pd.read_csv(sys.argv[1])
actualdata = actualdata[['initial x', 'final x']]


#If data exists:
real_profile = actualdata['initial x']
real_stddev = real_profile.std()
real_profile_hist, bins = np.histogram(real_profile, bins=400, range=(-0.01, 0.01))


distorted_profile = actualdata['final x']
distorted_stddev = distorted_profile.std()
distorted_profile_hist, bins2 = np.histogram(distorted_profile, bins=400, range=(-0.01, 0.01))

with open(sys.argv[1], 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([real_stddev, distorted_stddev, bins, real_profile_hist, bins2, distorted_profile_hist], f)