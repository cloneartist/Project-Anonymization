import pandas as pd
import numpy as np
import math
import random
import streamlit as st
# Load the tweet dataset into memory
tweets = pd.read_csv('covid_data.csv',nrows=20)
st.write(tweets)
# Identify sensitive attributes
sensitive_attrs = ['Likes Received',  'Retweets Received']

# Choose quasi-identifiers (QIs)
qis = ['User Followers',  'User Following']

# Define L-diversity threshold
l_diversity_threshold = 2

# Define privacy parameters for differential privacy
epsilon = 0.1
delta = 1e-5

# Group the dataset into partitions based on QI values
groups = tweets.groupby(qis)

# Define Laplace noise function
def add_laplace_noise(x, epsilon):
    sensitivity = 1.0 / epsilon
    noise = np.random.laplace(loc=0.0, scale=sensitivity, size=1)
    return x + noise[0]

# Loop over partitions and apply L-diversity anonymization with differential privacy
for name, group in groups:
    # Calculate diversity of sensitive attributes within partition
    diversity = 0
    for attr in sensitive_attrs:
        counts = group[attr].value_counts()
        p = counts / counts.sum()
        diversity += -p.dot(np.log2(p))

    # If diversity does not meet threshold, apply anonymization with differential privacy
    if diversity < l_diversity_threshold:
        print("hi")
        for attr in sensitive_attrs:
            sensitivity=1
            # Apply Laplace noise to sensitive attribute with differential privacy
            group[attr] = group[attr].apply(lambda x: add_laplace_noise(x, epsilon))
            # Apply post-processing step to ensure differential privacy
            group[attr] = group[attr].apply(lambda x: x + np.random.normal(scale=math.sqrt(2 * math.log(1.25/delta)) * sensitivity / epsilon))

# Save anonymized dataset to a new file
tweets.to_csv('anonymized_tweets.csv', index=False)
st.write(tweets)