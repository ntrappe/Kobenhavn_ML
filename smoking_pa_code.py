# 
# Author: Nicole Trappe
# Institution: UCPH
# Date: February 9, 2020
#
# HOMEWORK 1: SMOKING
#

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from scipy import stats
from scipy.stats import ttest_ind, norm

# Constants
AGE_IDX = 0
FEV1_IDX = 1
HEIGHT_IDX = 2
GENDER_IDX = 3
SMOKE_STAT_IDX = 4
WEIGHT_IDX = 5

#-----------------------------------------------------------------------------#
# EXERCISE 1:  Read the data from the file smoking.txt, and divide the dataset
# into two groups con- sisting of smokers and non-smokers. Write a script which 
# computes the average lung function, measured in FEV1, among the smokers and 
# among the non-smokers.
#-----------------------------------------------------------------------------#
print('==========EXERCISE 1==========')
smoking_data = np.loadtxt("smoking.txt")

# Create empty smoker and non-smoker arrays
smk_arr = np.empty((0,6), int)
non_smk_arr = np.empty((0,6), int)
mean_smk = 0
mean_non = 0

# If we have a smoker, add to smoker array and its sum; otherwise, do for 
# non-smoker
for row in smoking_data:
        if(row[SMOKE_STAT_IDX] == 1):
                smk_arr = np.append(smk_arr, [row], axis=0)
                mean_smk += row[FEV1_IDX]
        else:
                non_smk_arr = np.append(non_smk_arr, [row], axis=0)
                mean_non += row[FEV1_IDX]

# Divide both sums by the number of data points
smk_num_row, smk_num_cols = np.shape(smk_arr)
mean_smk /= smk_num_row
non_num_row, non_num_cols = np.shape(non_smk_arr)
mean_non /= non_num_row

print('Smoker FEV1 mean', mean_smk)
print('Non-smoker FEV1 mean', mean_non)

# Calculate the median for the smoker and non-smoker FEV1 and Age
print('Smoker FEV1 median', np.median(smk_arr[:,FEV1_IDX]))
print('Non-smoker FEV1 median', np.median(non_smk_arr[:,FEV1_IDX]))
print('Smoker Age median', np.median(smk_arr[:,AGE_IDX]))
print('Non-smoker Age median', np.median(non_smk_arr[:,AGE_IDX]))


#-----------------------------------------------------------------------------#
# EXERCISE 2: Make a box plot of the FEV1 in the two groups.
#-----------------------------------------------------------------------------#
print('\n==========EXERCISE 2==========')
df = pd.DataFrame(smoking_data, columns=['Age','FEV1','Height','Gender','Status','Weight'])
ax1 = sns.boxplot(x='Status', y='FEV1', data=df, palette='magma')
ax1.set(title='Boxplot of FEV1 Levels in Non-smokers and Smokers', xlabel='Status: 0-Non-smoker, 1-Smoker', ylabel='Lung Function (FEV1)');

# Variances of the lung function (FEV1) of both distributions
print('Smoker FEV1 variance', np.var(smk_arr[:,1]))
print('Non-smoker FEV1 variance', np.var(non_smk_arr[:,1]))


#-----------------------------------------------------------------------------#
# EXERCISE 3: Write a script that performs a two-sided t-test whose null 
# hypothesis is that the two populations have the same mean. Use a 
# significance level of alpha=0.05, and return a binary response indicating 
# acceptance or rejection of the null hypothesis. You should try do implement
# it by yourself - though not the CDF of the t-distribution, use scipy. If 
# you cant, you may use scipys stats.ttest_ind. Make a box plot of the FEV1 
# in the two groups.
#-----------------------------------------------------------------------------#
print('\n==========EXERCISE 3==========')
# Manually completed
sample1, sample2 = smk_arr[:,FEV1_IDX], non_smk_arr[:,FEV1_IDX]
alpha = 0.05
# Standard deviations of both samples   
std_smk, std_non = np.std(sample1), np.std(sample2)
# Standard error of both samples  
err_smk, err_non = stats.sem(sample1), stats.sem(sample2)
# Standard error on difference between the samples    
sed = np.sqrt(err_smk**2 + err_non**2)
# T-statistic (distribution density)
t_stat = (mean_smk - mean_non)/sed
# Degrees of freedom (both samples combined - 2)
degf = np.size(sample1,0) + np.size(sample2,0) - 2
# P-val; 1 - upper-tail then * 2 bc 2-tailed
pval = norm.cdf(t_stat, degf) * 2
# value on distribution to be greater/less than
t_critical = round(stats.t.ppf(alpha/2, degf), 3)

# NOTE: My manually coded p-value is such a small number that it keeps getting 
# rounded to 0 so I had to use the built in ttest_ind because I can't see how 
# to allow for more significant digits. My test statistic was the same.

# Using build in independent samples t test
# equal_var False bc unequal pop. variances & nan_policy to omit bc diff. sizes
t_stat, pval = stats.ttest_ind(a=smk_arr[:,FEV1_IDX], b=non_smk_arr[:,FEV1_IDX], equal_var=False, nan_policy='omit')

if pval < alpha:
    print('Reject null hypothesis')
else:
    print('Fail to reject null hypothesis')

print('t-critical:', t_critical)
print('t-statistic:', t_stat)
print('degrees of freedom:', degf)
print('p-value:', pval)

#-----------------------------------------------------------------------------#
# EXERCISE 4: Compute the correlation between age and FEV1. Make a 2D plot of 
# age versus FEV1 where non smokers appear in one color and smokers appear in 
# another.
#-----------------------------------------------------------------------------#
print('\n==========EXERCISE 4==========')
corr = np.corrcoef(smoking_data[:,AGE_IDX], smoking_data[:,FEV1_IDX])
print('correlation matrix', corr)
print('correlation coefficient', corr[0,1])

ax2 = sns.catplot(x='Age', y='FEV1', hue='Status', jitter=False, aspect=1.5, data=df, palette='magma');
ax2.set(title='Age vs FEV1 for Non-smokers and Smokers', xlabel='Age', ylabel='Lung Function (FEV1)');

#-----------------------------------------------------------------------------#
# EXERCISE 5: Create a histogram over the age of subjects in each of the two 
# groups, smokers and non-smokers.
#-----------------------------------------------------------------------------#
print('\n==========EXERCISE 5==========')
ax3 = sns.FacetGrid(df)
ax3 = sns.distplot(a=smk_arr[:,0], color='tomato', axlabel='Age');
ax3 = sns.distplot(a=non_smk_arr[:,0], color='purple');
ax3.set(title='Histograms of Age for Non-smokers and Smokers')
print('Non-smokers in purple and smokers in red')
