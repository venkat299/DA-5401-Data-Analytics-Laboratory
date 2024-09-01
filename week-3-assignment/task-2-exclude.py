# Task 2 [10 points]
# -----------------
# Perform EDA on the dataset to understand the predictor features and how are they influencing each
# other. Also, study how each individual predictor influence the output variable. You may use
# correlation study to estimate the influence. Add necessary visualization and its representive
# interpretations to substantiate your inferences. The outcome of this step is figure out the requires
# features and their respective transformation.

import pandas as pd
import numpy as np
from autoviz.AutoViz_Class import AutoViz_Class

import common

# load data
data = common.load_assignment_data()
data = data.astype('float')

# create numpy objects for X,y
x = np.array(data[['x1','x2','x3','x4','x5']])
y = np.expand_dims(data['y'], 1)

# Step 1 : Lets analyse each variable individually

# check the statistic of data
print(data.describe())
#                x1          x2          x3          x4          x5             y
# count  101.000000  101.000000  101.000000  101.000000  101.000000    101.000000
# mean     7.548713   23.755050  111.371386   98.133762   34.000000  10244.460297
# std      0.380115  292.850177   55.812213    4.942089   30.557704   1022.766123
# min      6.680000 -466.860000    9.800000   86.830000    0.000000   8062.540000
# 25%      7.250000 -208.890000   66.380000   94.240000    6.760000   9469.940000
# 50%      7.530000   38.950000  104.180000   97.900000   25.000000  10187.660000
# 75%      7.800000  262.920000  168.160000  101.420000   57.760000  10866.630000
# max      8.370000  546.880000  195.810000  108.850000  100.000000  12631.050000


# Create visualization charts
AV = AutoViz_Class()
target_variable = "y"
df_av = AV.AutoViz(
    "",
    depVar=target_variable,
    dfte=data,
    chart_format ='png',
    header=0,
    verbose=2,
    save_plot_dir='./')

# generated violin plot

# generated distribution plot for all variables

# generated heat map

# generated pair scatter plot

# generated scatter plot(for predictor variable in y axis)

# Inference on variables from violin plot and distribution plots:
# x1 is symmetric(very less skew-0.1), normal(from qq-plot) 
# x2 is symmetric(skew=0) and but has fat tails ie has more values on extreme when compared to  normal(from qq-plot)
# x3 appears to be bimodal ( from violin plot) and left tailed(from qq-plot)
# x4 appears symmetric(violin plot) and normal(from qq-plot)
# x5 is right skewed(skew=0.6) and has fat tails(from qq-plot)
# y is symmetric(very less skew-0.3), normal(from qq-plot) 

# Step 2 : pairwise correlation

# let's add the output column to the mix and evaluate the correlation coefficient.
all_data = np.concatenate((x, y), axis=1)
corr_all = np.corrcoef(all_data.T)

print(corr_all)
# result 
# [[ 1.          0.70227559 -0.22369731  0.99997833 -0.00144945  0.9982136 ]
#  [ 0.70227559  1.         -0.03260292  0.70319638  0.00261306  0.71670271]
#  [-0.22369731 -0.03260292  1.         -0.22480781  0.08132155 -0.22026513]
#  [ 0.99997833  0.70319638 -0.22480781  1.         -0.00177718  0.99826603]
#  [-0.00144945  0.00261306  0.08132155 -0.00177718  1.          0.04660974]
#  [ 0.9982136   0.71670271 -0.22026513  0.99826603  0.04660974  1.        ]]


print(corr_all > 0.7)
# result 
# [[ True  True False  True False  True]
#  [ True  True False  True False  True]
#  [False False  True False False False]
#  [ True  True False  True False  True]
#  [False False False False  True False]
#  [ True  True False  True False  True]]

# Inference from scatter plot:
# feature x1 and feature x4 are heavily correlated (linear)
# It may be worth removing one of those features from our system.
# feature x2 and feature x5 has a  quadratic relation
# feature x1 and feature x2 has a positive trend  
# feature x4 and feature x2 has a  positive trend



# conclusion
# feature x4 can be dropped from model as it has linear relation with x1 
# feature x5 can also be dropped as it has quadratic relation with x2