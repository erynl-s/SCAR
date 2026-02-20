'''
@author: eryn

Code to implement the causal graph for the data generation process.

define causal variable distributions and SCMs for which the volumes will be subject to. 
Variable distributions and SCMs can be defined freely by the used to suit their needs. 
Mulitple "deformation" or effect variables can be defined to cause effect on mutliple regions in a single image.
ISV distrbutions are also defined here. These define the amount of global defomation will be inflicted on the template image to simulate unique subjects.

'''

## Imports
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from sklearn.model_selection import train_test_split
from numpy.random import Generator, PCG64
import matplotlib.pyplot as plt
import seaborn as sns
from utils import merge_shuffle_data
from pathlib import Path
import sklearn.preprocessing

# name of experiment
EXP = ''

# define ISV variables
ISV_MEAN = 0
ISV_SD = 1
ISV_LOWER_BOUND = -1.25
ISV_UPPER_BOUND = 1.25
SAMPLES = 1000 #number of samples to generate

# define variable for causal variable distributions
SEED_A = 31
SEED_ISV = 7
SEED_NOISE = 20
SEED_B = 10
SEED_GENE = 15

A_dst_type = 'normal' #distribution type for a

A_MEAN = 75
A_SD = 10
A_UPPER_BOUND = 95
A_LOWER_BOUND = 55
ALPHA_A = 1 #weighting of a in SCM
ALPHA_B = 1 #weighting of b in SCM



main_dir = ''
Path(main_dir + EXP + '/effect_distributions').mkdir(parents=True, exist_ok=True)
save_dir = main_dir + EXP + '/effect_distributions/'

#function to generate sampling distributions
def get_subjectVariability_distribution(seed, num, lower_bound, upper_bound):
    '''
    get dataframe with subject effects
    
    -----inputs------
    -seed: seed for deterministic random number generation
    -num: number of samples to generate
    -lower_bound: lower bound of truncated gaussian distribution for subject sampling
    -upper_bound: upper bound of truncated gaussian distribution for subject sampling
    -----outputs-----
    -isv_df: subject effect distribution dataframe
    '''
    #define subject effect distribution
    numpy_randomGen = Generator(PCG64(seed))
    truncnorm.random_state=numpy_randomGen
    isv_dst_bounds = [lower_bound, upper_bound]
    
    # create a distrbution with uniform distribution
    isv_dst_raw = numpy_randomGen.uniform(isv_dst_bounds[0], isv_dst_bounds[1], num) #get samples from subject effect distribution
    return isv_dst_raw


def get_a_distribution(seed, dst_type, a_mu, a_sd, num, a_lower_bound, a_upper_bound):
    '''
    get dataframe with A (causal variable) distributions 
    
    -----inputs------
    -seed: seed for deterministic random number generation
    -mu: mean value of subject effect sampling distribution if not
    -sd: standard deviation of subject effect sampling distribution if normal
    -num: number of samples to generate
    -lower_bound: lower bound of truncated gaussian distribution for subject sampling
    -upper_bound: upper bound of truncated gaussian distribution for subject sampling
    -----outputs-----
    -a_dst_raw: age distribution
    '''
    
    numpy_randomGen = Generator(PCG64(seed))
    truncnorm.random_state=numpy_randomGen

    if dst_type == 'normal':
        age_dst_mean = a_mu
        age_dst_sd = a_sd
        age_dst_bounds = [a_lower_bound, a_upper_bound]
        age_dst = truncnorm.rvs((age_dst_bounds[0]-age_dst_mean)/age_dst_sd, (age_dst_bounds[1]-age_dst_mean)/age_dst_sd, loc=age_dst_mean, scale=age_dst_sd, size=num)
    elif dst_type == 'uniform':
        age_dst_mean = (a_lower_bound + a_upper_bound) / 2
        age_dst_sd = (a_upper_bound - a_lower_bound) / 4
        age_dst_bounds = [a_lower_bound, a_upper_bound]
        age_dst = numpy_randomGen.uniform(a_lower_bound, a_upper_bound, num)
    else:
        raise ValueError("dst_type must be 'normal' or 'uniform'")

    return age_dst
#####################################################################
# Define and generate distributions for ISV and any causal variables
#####################################################################
# ISV needed to generate different subjects
isv_dst = get_subjectVariability_distribution(seed=SEED_ISV, num=SAMPLES,  lower_bound=-1.25, upper_bound=1.25)

# e.g., A distribution
a_dst = get_a_distribution(seed=SEED_A, dst_type = A_dst_type, a_mu=75, a_sd=10, num=SAMPLES,  a_lower_bound=55, a_upper_bound=95)

#e.g., B distribution
numpy_randomGen = Generator(PCG64(SEED_B))
theta = [0.3,0.5,0.2]
b_dst = numpy_randomGen.choice([0, 1, 2], size=SAMPLES, p=theta) 

# Note: these distributions can be defined freely to the users needs

#Normalise variables if necessary
a_scaled = (a_dst - A_LOWER_BOUND) / (A_UPPER_BOUND - A_LOWER_BOUND) # normalized to between 0 and 1

#########################################################################
# Define SCMs where the "deformation" variable will be the scale of which the original volume will be multiplied by
#######################################################################
# include noise variable in SCM
numpy_randomGen = Generator(PCG64(SEED_NOISE))
noise = numpy_randomGen.random(SAMPLES) 
#e.g., SCM Note: can be any valid SCM and can define multiple 'defomation variable to apply to different regions

deformation =  a_scaled *ALPHA_A + (2/3)*b_dst + (noise/10) #edu_scaled*ALPHA_E + (noise/10) #ABETA+ edu_scaled * ALPHA_E + (noise/10)

deformation = sklearn.preprocessing.minmax_scale(deformation, feature_range=(0.8, 1.2))
# deformation_2 = sklearn.preprocessing.minmax_scale(deformation_2, feature_range=(0.8, 1.2))

s = {'a_dst': a_dst,
     'b_dst': b_dst,
     'isv_dst': isv_dst,
     'effect1_dst': deformation,
    #  'effect2_dst': deformation_2 
    }
df = pd.DataFrame(data=s)


# save dataframe to csv
df.to_csv(save_dir + '/train_dst.csv')


