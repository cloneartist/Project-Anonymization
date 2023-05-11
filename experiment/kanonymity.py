import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import pairwise_distances


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
        # Generalize the zipcode to the first 3 digits, to replace the exact location with a higher-level geographic region
        return str(value)[:3]
    else:
        # If the column is not age or zipcode, return the original value
        return value



def ola_anonymity():
    # Load the dataset
    data = pd.read_csv('data.csv')

    # Get user input for the quasi-identifier attributes and the sensitive attribute
    qi_attributes = st.multiselect('Select the quasi-identifier attributes:', options=data.columns)
    sensitive_attribute = st.selectbox('Select the sensitive attribute:', options=data.columns)

    # Get user input for the desired k-anonymity level
    k = st.slider('Select the desired k-anonymity level:', min_value=1, max_value=len(data)//2, step=1)

    # Get user input for the sensitivity weights for each attribute
    sensitivity_weights = {}
    for attribute in data.columns:
        if attribute in qi_attributes or attribute == sensitive_attribute:
            sensitivity_weights[attribute] = st.slider(f'Select the sensitivity weight for {attribute}:', min_value=1, max_value=10, step=1)
        else:
            sensitivity_weights[attribute] = 1

    # Apply the weights to each attribute
    attribute_weights = {}
    for attribute in data.columns:
        if attribute in qi_attributes or attribute == sensitive_attribute:
            attribute_weights[attribute] = sensitivity_weights[attribute]
        else:
            attribute_weights[attribute] = 1

    # Normalize the weights
    total_weight = sum(attribute_weights.values())
    normalized_weights = {attribute: weight/total_weight for attribute, weight in attribute_weights.items()}

    # Get user input for the suppression and generalization rules for each attribute
    suppression_rules = {}
    generalization_rules = {}
    for attribute in data.columns:
        if attribute in qi_attributes:
            suppression_rules[attribute] = st.selectbox(f'Select the suppression rule for {attribute}:', options=['keep', 'remove'])
            generalization_rules[attribute] = st.selectbox(f'Select the generalization rule for {attribute}:', options=['keep', 'round', 'truncate'])
        elif attribute == sensitive_attribute:
            suppression_rules[attribute] = st.selectbox(f'Select the suppression rule for {attribute}:', options=['keep', 'remove'])
            generalization_rules[attribute] = st.selectbox(f'Select the generalization rule for {attribute}:', options=['keep', 'round', 'truncate'])
        else:
            suppression_rules[attribute] = 'keep'
            generalization_rules[attribute] = 'keep'


    if st.button("Submit"):
    # Group the data by the quasi-identifier attributes
        groups = data.groupby(qi_attributes)


        # Apply the anonymization rules to each group
        anonymized_data = []
        for _, group in groups:
            # Determine the k-anonymity level of the group
            group_size = len(group)
            if group_size < k:
                # Apply more aggressive anonymization rules to smaller groups
                suppression_rule = 'remove'
                generalization_rule = 'truncate'
            else:
                suppression_rule = suppression_rules[sensitive_attribute]
                generalization_rule = generalization_rules[sensitive_attribute]
            
            # Apply the anonymization rules to the sensitive attribute
            if suppression_rule == 'remove':
                group[sensitive_attribute] = 'Unknown'
            else:
                if generalization_rule == 'keep':
                    group[sensitive_attribute] = group[sensitive_attribute]
                elif generalization_rule == 'round':
                    group[sensitive_attribute] = round(group[sensitive_attribute], -3)
                elif generalization_rule == 'truncate':
                    group[sensitive_attribute] = group[sensitive_attribute] // 1000 * 1000
                    
            # Apply the generalization rules to the quasi-identifier attributes
            for attribute in qi_attributes:
                if generalization_rules[attribute] == 'keep':
                    group[attribute] = group[attribute]
                elif generalization_rules[attribute] == 'round':
                    group[attribute] = round(group[attribute], -1)
                elif generalization_rules[attribute] == 'truncate':
                    group[attribute] = group[attribute] // 10 * 10
            
            # Add the anonymized group to the output dataset
            anonymized_data.append(group)

        # Combine the anonymized groups into a single dataframe
        anonymized_data = pd.concat(anonymized_data)

        # Output the anonymized dataset
        anonymized_data.to_csv('anonymized_data_ola.csv', index=False)
        st.write(anonymized_data)    
def k_anonymity_algo():
    
    
    dataf = pd.read_csv(uploaded_file,nrows=250)
    df=dataf.copy()
    print(list(df.columns))

    print(df.dtypes)
    # Display column headers
    st.write("## Select Quasi Identifiers")

    # Get column names
    columns = df.columns.tolist()

    # Create checkboxes for column selection
    generalize_cols = st.multiselect("Select columns", columns)

    # df = pd.read_csv('data.csv')
    st.write(df)
    quasi_identifiers = quasi_identifiers_fn(df)
    print(len(quasi_identifiers), len(df.columns))
    k_level = st.slider("Select k-anonymity level", 2, 10, 5)
    st.write("Selected k-anonymity level:", k_level)

    # Assume the dataset is stored in a pandas DataFrame object 'df'

    # quasi_identifiers = ['age', 'gender', 'race']
    quasi_identifiers = generalize_cols

    # k = 2# desired k-anonymity level
    k = k_level
    
    sensitive_attr_list=[col for col in df.columns if col not in quasi_identifiers]
    sensitive_attr = st.selectbox("Select a sensitive attribute", sensitive_attr_list)

    # st.write(ola_anonymity(quasi_identifiers,sensitive_attr))
    if st.button("Submit"):
       
        # Group the dataset by quasi-identifier values

        grouped_data = df.groupby(
            quasi_identifiers).size().reset_index(name='count')

        # Identify small groups that violate the k-anonymity requirement
        small_groups = grouped_data[grouped_data['count'] < k]

        # Anonymize small groups using generalization
        for i, row in small_groups.iterrows():
            group_values = row[quasi_identifiers]
            generalized_values = []
            for qi in quasi_identifiers:
                # Generalize the quasi-identifier value to a higher level of abstraction
                generalized_value = generalize(group_values[qi], qi)
                generalized_values.append(generalized_value)
            # Replace the original quasi-identifier values with the generalized values
            df.loc[(df[quasi_identifiers] == group_values).all(
                axis=1), quasi_identifiers] = generalized_values

        df.to_csv("anonymized.csv", index=False)

        st.write(df)

        # data = pd.read_csv(uploaded_file)
        data=dataf.copy()
        anonymized_data = pd.read_csv("anonymized.csv")

        # calculate the "entropy" of the sensitive attribute before and after anonymization
        # sensitive = "education"
        sensitive= sensitive_attr
        original_entropy = calculate_entropy(data[sensitive])
        anonymized_entropy = calculate_entropy(anonymized_data[sensitive])
        # create a bar chart showing the results
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(["Original", "Anonymized"], [original_entropy,
            anonymized_entropy], color=["blue", "orange"])
        ax.set_ylabel("Entropy")
        ax.set_title("Entropy of Sensitive Attribute Before and After Anonymization")

        # show the results in Streamlit
        st.pyplot(fig)
        # calculate the proportion of records with a unique quasi-identifier before and after anonymization

        original_proportion = calculate_proportion_unique(
            data, quasi_identifiers)
        anonymized_proportion = calculate_proportion_unique(
            anonymized_data, quasi_identifiers)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(["Original", "Anonymized"], [original_proportion,
                                            anonymized_proportion], color=["blue", "orange"])
        ax.set_ylabel("Proportion")
        ax.set_title(
            "Proportion of Records with Unique Quasi-Identifier Before and After Anonymization")

        # show the results in Streamlit
        st.pyplot(fig)

        # Load the original dataset and the anonymized dataset
        # original_data = pd.read_csv('data.csv')
        original_data=dataf.copy()
        anonymized_data = pd.read_csv('anonymized.csv')
        print(len(original_data), len(anonymized_data))
        # Specify the quasi-identifier and sensitive attributes
        qi_attributes = quasi_identifiers
        # sensitive_attribute = 'occupation'
        sensitive_attribute=sensitive_attr
        # Calculate the information loss
        original_data_modes = original_data.groupby(qi_attributes)[
            sensitive_attribute].apply(lambda x: x.mode().iloc[0]).reset_index()
        anonymized_data_modes = anonymized_data.groupby(qi_attributes)[
            sensitive_attribute].apply(lambda x: x.mode().iloc[0]).reset_index()
        num_rows = min(len(original_data_modes), len(anonymized_data_modes))
        original_data_modes = original_data_modes.sample(
            num_rows, random_state=42).reset_index()
        anonymized_data_modes = anonymized_data_modes.sample(
            num_rows, random_state=42).reset_index()
        iloss = np.sum(original_data_modes !=
                       anonymized_data_modes) / (num_rows)

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
                color=alt.Color('Risk', scale=alt.Scale(
                    scheme='redyellowgreen')),
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

uploaded_file = st.file_uploader("Upload dataset", type="csv")
if uploaded_file is not None:
   # Read the CSV file into a Pandas dataframe
    algorithm = st.selectbox('Select the k-anonymization algorithm:', ('Basic', 'OLA'))
    if algorithm=="Basic":
        k_anonymity_algo()
    elif algorithm=="OLA":
        ola_anonymity()