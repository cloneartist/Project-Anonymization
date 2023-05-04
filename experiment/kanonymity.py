import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import OneHotEncoder

def entropy(s):
    p, ln = pd.Series(s).value_counts(normalize=True), np.log2
    return -p.dot(ln(p))


def quasi_identifiers_fn(df, threshold=0.8):
    QI = []
    for col in df.columns:
        E = entropy(df[col])
        if E > threshold:
            QI.append(col)
    return QI


def calculate_entropy(column):
    """
    Calculate the entropy of a column of data.
    """
    value_counts = column.value_counts()
    proportions = value_counts / len(column)
    entropy = -(proportions * np.log2(proportions)).sum()
    return entropy


def calculate_proportion_unique(data, quasi_identifier):
    """
    Calculate the proportion of records with a unique quasi-identifier.
    """
    grouped = data.groupby(quasi_identifier)
    unique_counts = grouped.size().value_counts()
    total = unique_counts.sum()
    proportion_unique = unique_counts[1] / total
    return proportion_unique


def generalize(value, column_name):
    if column_name == 'age':
        # Generalize the age value to the nearest 5-year interval
        return round(value/5.0)*5
    elif column_name == 'zipcode':
        # Generalize the zipcode to the first 3 digits, to replace the exact
        # location with a higher-level geographic region
        return str(value)[:3]
    else:
        # If the column is not age or zipcode, return the original value
        return value


# Example usage
df = pd.read_csv('data.csv')
st.write(df)
quasi_identifiers = quasi_identifiers_fn(df)
print(quasi_identifiers)


# Assume the dataset is stored in a pandas DataFrame object 'df'
# list of quasi-identifier column names
quasi_identifiers = ['age', 'gender', 'race']

# Group the dataset by quasi-identifier values
grouped_data = df.groupby(quasi_identifiers).size().reset_index(name='count')

# Assume the dataset is stored in a pandas DataFrame object 'df'
k = 2# desired k-anonymity level

# Identify small groups that violate the k-anonymity requirement
small_groups = grouped_data[grouped_data['count'] < k]

# Anonymize small groups using generalization
for i, row in small_groups.iterrows():
    group_values = row[quasi_identifiers]
    generalized_values = []
    for qi in quasi_identifiers:
        # Generalize the quasi-identifier value to a higher level of abstraction
        # For example, round the age to the nearest 5-year interval, or replace the
        # exact zipcode with the name of the city or county
        generalized_value = generalize(group_values[qi], qi)
        generalized_values.append(generalized_value)
    # Replace the original quasi-identifier values with the generalized values
    df.loc[(df[quasi_identifiers] == group_values).all(
        axis=1), quasi_identifiers] = generalized_values

df.to_csv("anonymized.csv", index=False)

st.write(df)


# example data
data = pd.read_csv("data.csv")
anonymized_data = pd.read_csv("anonymized.csv")
# calculate the entropy of the sensitive attribute before and after anonymization
sensitive = "education"
original_entropy = calculate_entropy(data[sensitive])
anonymized_entropy = calculate_entropy(anonymized_data[sensitive])

# calculate the proportion of records with a unique quasi-identifier before and after anonymization
original_proportion = calculate_proportion_unique(data, quasi_identifiers)
anonymized_proportion = calculate_proportion_unique(
    anonymized_data, quasi_identifiers)

# create a bar chart showing the results
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(["Original", "Anonymized"], [original_entropy,
       anonymized_entropy], color=["blue", "orange"])
ax.set_ylabel("Entropy")
ax.set_title("Entropy of Sensitive Attribute Before and After Anonymization")

# show the results in Streamlit
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(["Original", "Anonymized"], [original_proportion,
       anonymized_proportion], color=["blue", "orange"])
ax.set_ylabel("Proportion")
ax.set_title(
    "Proportion of Records with Unique Quasi-Identifier Before and After Anonymization")

# show the results in Streamlit
st.pyplot(fig)


import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from sklearn.metrics import pairwise_distances

# Load the original dataset and the anonymized dataset
original_data = pd.read_csv('data.csv')
anonymized_data = pd.read_csv('anonymized.csv')
print(len(original_data),len(anonymized_data))
# Specify the quasi-identifier and sensitive attributes
qi_attributes = quasi_identifiers
sensitive_attribute = 'occupation'

# Calculate the information loss
original_data_modes = original_data.groupby(qi_attributes)[sensitive_attribute].apply(lambda x: x.mode().iloc[0]).reset_index()
anonymized_data_modes = anonymized_data.groupby(qi_attributes)[sensitive_attribute].apply(lambda x: x.mode().iloc[0]).reset_index()
num_rows = min(len(original_data_modes), len(anonymized_data_modes))
original_data_modes = original_data_modes.sample(num_rows, random_state=42).reset_index()
anonymized_data_modes = anonymized_data_modes.sample(num_rows, random_state=42).reset_index()
iloss = np.sum(original_data_modes != anonymized_data_modes) / (num_rows)

# Measure the degree of re-identification risk
encoder = OneHotEncoder(sparse=False)
encoded_data = encoder.fit_transform(anonymized_data[qi_attributes])
distances = pairwise_distances(encoded_data, metric='hamming')
risks = np.sum(distances < 1/(2*k), axis=1) / len(qi_attributes)


# Define a function to create a scatter plot of the re-identification risk
def create_scatter_plot(risks):
    data = pd.DataFrame({
        'Risk': risks,
        'Group': range(len(risks)),
    })
    chart = alt.Chart(data).mark_circle().encode(
        x='Group',
        y='Risk',
        size=alt.Size('Risk', scale=alt.Scale(range=[50, 500])),
        color=alt.Color('Risk', scale=alt.Scale(scheme='redyellowgreen')),
    ).configure_axis(
        labelFontSize=20,
        titleFontSize=20,
    ).configure_text(
        fontSize=20,
    )
    return chart

# Display the evaluation results
st.write('# Anonymization Quality Evaluation')
st.write('## Information Loss')
st.write(iloss.drop('index'))
st.write('## Re-identification Risk')
st.write(create_scatter_plot(risks))

