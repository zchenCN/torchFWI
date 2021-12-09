"""
Cross validation to select hyperparameters

@data: 2021-12-09
@author: chazen
"""

import numpy as np

###########################################################
#                       Settings                          
###########################################################
sources_x = 
receivers_x = 



###########################################################
#                       Train test split                          
###########################################################
ns = len(sources_x)
index = np.random.shuffle(np.arange(ns))
split = int(0.7 * ns)
train_x = sources_x[:split]
test_x = sources_x[split:]