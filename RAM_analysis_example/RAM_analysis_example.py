import numpy as np
from RamAnalysis import BestDistribution, AnalysisPlot


# Item name

comp = "Item_x"

# Item time to repair [h]

ttr = np.array([ 1. , 0.2, 0.7, 0. , 0.6, 0.5, 1. , 1.5, 1., 1.])

# Item time to failure [h]

ttf = np.array([ 247.,  462.,  724.,  329.,  112.,  467.,  126.,  325.,  790., 1281.])


# RAM Analysis

analysis = AnalysisPlot(ttf, ttr, comp)

#Reliability

reliability_dist = BestDistribution(ttf)

dist, par = reliability_dist.get_best_distribution()

analysis.reliability(dist, par)
 

#Maintainability

maintanability = BestDistribution(ttr)

dist, par = maintanability.get_best_distribution()

analysis.maintainability(dist, par)


#Availability

analysis.availability()
