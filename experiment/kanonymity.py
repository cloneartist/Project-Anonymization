import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import pairwise_distances
import hashlib
import time
import math


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
        return round(value / 5.0) * 5
    elif column_name == 'zipcode':
        # Generalize the zipcode to the first 3 digits, to replace the exact location with a higher-level geographic region
        return str(value)[:3]
    elif column_name == 'latitude':
        return round(value)
    elif column_name == 'longitude':
        return round(value)
    elif column_name == 'housing_median_age':
        # Generalize the housing median age to the nearest 10-year interval
        return round(value / 10.0) * 10
    elif column_name == 'mean_rooms':
        # Generalize the mean number of rooms to the nearest integer
        return round(value)
    elif column_name == 'mean_bedrooms':
        # Generalize the mean number of bedrooms to the nearest integer
        return round(value)
    elif column_name == 'population':
        # Generalize the population to the nearest thousand
        return round(value / 1000) * 1000
    elif column_name == 'households':
        # Generalize the number of households to the nearest thousand
        return round(value / 1000) * 1000
    elif column_name == 'median_income':
        # Generalize the median income to the nearest thousand
        return round(value / 1000) * 1000
    elif column_name == 'median_house_value':
        # Generalize the median house value to the nearest ten thousand
        return round(value / 10000) * 10000
    elif column_name == 'ocean_proximity':
        # No generalization for ocean proximity column, return the original value
        return value
    else:
        # If the column is not recognized, return the original value
        return value



def ola_anonymity():
    # Load the dataset
    data = pd.read_csv('data.csv')

    # Get user input for the quasi-identifier attributes and the sensitive attribute
    qi_attributes = st.multiselect(
        'Select the quasi-identifier attributes:', options=data.columns)
    sensitive_attribute = st.selectbox(
        'Select the sensitive attribute:', options=data.columns)

    # Get user input for the desired k-anonymity level
    k = st.slider('Select the desired k-anonymity level:',
                  min_value=1, max_value=len(data)//2, step=1)

    # Get user input for the sensitivity weights for each attribute
    sensitivity_weights = {}
    for attribute in data.columns:
        if attribute in qi_attributes or attribute == sensitive_attribute:
            sensitivity_weights[attribute] = st.slider(
                f'Select the sensitivity weight for {attribute}:', min_value=1, max_value=10, step=1)
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
    normalized_weights = {
        attribute: weight/total_weight for attribute, weight in attribute_weights.items()}

    # Get user input for the suppression and generalization rules for each attribute
    suppression_rules = {}
    generalization_rules = {}
    for attribute in data.columns:
        if attribute in qi_attributes:
            suppression_rules[attribute] = st.selectbox(
                f'Select the suppression rule for {attribute}:', options=['keep', 'remove'])
            generalization_rules[attribute] = st.selectbox(
                f'Select the generalization rule for {attribute}:', options=['keep', 'round', 'truncate'])
        elif attribute == sensitive_attribute:
            suppression_rules[attribute] = st.selectbox(
                f'Select the suppression rule for {attribute}:', options=['keep', 'remove'])
            generalization_rules[attribute] = st.selectbox(
                f'Select the generalization rule for {attribute}:', options=['keep', 'round', 'truncate'])
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
                    group[sensitive_attribute] = round(
                        group[sensitive_attribute], -3)
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

    dataf = pd.read_csv(uploaded_file, nrows=250)
    df = dataf.copy()
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
    k_level = st.number_input("Select k-anonymity level", 2, 10, 5)
    st.write("Selected k-anonymity level:", k_level)

    # Assume the dataset is stored in a pandas DataFrame object 'df'

    # quasi_identifiers = ['age', 'gender', 'race']
    quasi_identifiers = generalize_cols

    # k = 2# desired k-anonymity level
    k = k_level

    sensitive_attr_list = [
        col for col in df.columns if col not in quasi_identifiers]
    sensitive_attr = st.selectbox(
        "Select a sensitive attribute", sensitive_attr_list)

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

        # Compute hash of the anonymized data
        hash_object = hashlib.sha256(df.to_csv().encode())
        hash_value = hash_object.hexdigest()

        # Append hash value to the CSV file as a new line
        with open("anonymized.csv", "a") as file:
            file.write("\n")
            file.write("#Hash: {}".format(hash_value))

        st.write(df)

        # data = pd.read_csv(uploaded_file)
        data = dataf.copy()
        anonymized_data = pd.read_csv("anonymized.csv")

        # # calculate the "entropy" of the sensitive attribute before and after anonymization
        # # sensitive = "education"
        # sensitive = sensitive_attr
        # original_entropy = calculate_entropy(data[sensitive])
        # anonymized_entropy = calculate_entropy(anonymized_data[sensitive])
        # # create a bar chart showing the results
        # fig, ax = plt.subplots(figsize=(8, 6))
        # ax.bar(["Original", "Anonymized"], [original_entropy,
        #                                     anonymized_entropy], color=["blue", "orange"])
        # ax.set_ylabel("Entropy")
        # ax.set_title(
        #     "Entropy of Sensitive Attribute Before and After Anonymization")

        # # show the results in Streamlit
        # st.pyplot(fig)
        # calculate the proportion of records with a unique quasi-identifier before and after anonymization

        original_proportion = calculate_proportion_unique(
            data, quasi_identifiers)
        anonymized_proportion = calculate_proportion_unique(
            anonymized_data, quasi_identifiers)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(["Original", "Anonymized"], [original_proportion,
                                            anonymized_proportion], color=["blue", "orange"])
        # ax.set_ylabel("Proportion")
        # ax.set_title(
        #     "Proportion of Records with Unique Quasi-Identifier Before and After Anonymization")

        # # show the results in Streamlit
        # st.pyplot(fig)

        # Load the original dataset and the anonymized dataset
        # original_data = pd.read_csv('data.csv')
        original_data = dataf.copy()
        anonymized_data = pd.read_csv('anonymized.csv')
        print(len(original_data), len(anonymized_data))
        # Specify the quasi-identifier and sensitive attributes
        qi_attributes = quasi_identifiers
        # sensitive_attribute = 'occupation'
        sensitive_attribute = sensitive_attr
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


def mondrian(df, k_anonymity, partition_mode, agg_mode):
    arguments = ["", df, k_anonymity, partition_mode, agg_mode]

    # remove NaNs
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    df = df.iloc[:, 2:]
    dfcols = df.columns
    st.write("### Original dataset")
    st.write(df)

    # infer data types
    types = list(df.dtypes)
    # print(types)
    cat_indices = [i for i in range(len(types)) if types[i] == "object"]

    # convert df to numpy array
    df = np.array(df)

    # function to compute the span of a given column while restricted to a subset of rows (a data partition)
    def colSpans(df, cat_indices, partition):
        spans = dict()
        for column in range(len(types)):
            dfp = df[partition, column]  # restrict df to the current column
            if column in cat_indices:
                # span of categorical variables is its number of unique classes
                span = len(np.unique(dfp))
            else:
                # span of numerical variables is its range
                span = np.max(dfp) - np.min(dfp)
            spans[column] = span
        return spans

    # function to split rows of a partition based on median value (categorical vs. numerical attributes)

    def splitVal(df, dim, part, cat_indices, mode):
        # restrict whole dataset to a single attribute and rows in this partition
        dfp = df[part, dim]
        unique = list(np.unique(dfp))
        length = len(unique)
        if dim in cat_indices:  # for categorical variables
            if mode == 'strict':  # i do not mind about |lhs| and |rhs| being equal
                lhv = unique[:length//2]
                rhv = unique[length//2:]
                lhs_v = list(list(np.where(np.isin(dfp, lhv)))
                             [0])  # left partition
                rhs_v = list(list(np.where(np.isin(dfp, rhv)))
                             [0])  # right partition
                lhs = [part[i] for i in lhs_v]
                rhs = [part[i] for i in rhs_v]
            elif mode == 'relaxed':  # i want |lhs| = |rhs| +-1
                lhv = unique[:length//2]
                rhv = unique[length//2:]
                lhs_v = list(list(np.where(np.isin(dfp, lhv)))
                             [0])  # left partition
                rhs_v = list(list(np.where(np.isin(dfp, rhv)))
                             [0])  # right partition
                lhs = [part[i] for i in lhs_v]
                rhs = [part[i] for i in rhs_v]
                diff = len(lhs)-len(rhs)
                if diff == 0:
                    pass
                elif diff < 0:
                    # move first |diff|/2 indices from rhs to lhs
                    lhs1 = rhs[:(np.abs(diff)//2)]
                    rhs = rhs[(np.abs(diff)//2):]
                    lhs = np.concatenate((lhs, lhs1))
                else:
                    rhs1 = lhs[-(diff//2):]
                    lhs = lhs[:-(diff//2)]
                    rhs = np.concatenate((rhs, rhs1))
            else:
                lhs, rhs = splitVal(df, dim, part, cat_indices, 'relaxed')
        # for numerical variables, split based on median value (strict or relaxed)
        else:
            median = np.median(dfp)
            # strict partitioning (do not equally split indices of median values)
            if mode == 'strict':
                lhs_v = list(list(np.where(dfp < median))[0])
                rhs_v = list(list(np.where(dfp >= median))[0])
                lhs = [part[i] for i in lhs_v]
                rhs = [part[i] for i in rhs_v]
            elif mode == 'relaxed':  # exact median values are equally split between the two halves
                lhs_v = list(list(np.where(dfp < median))[0])
                rhs_v = list(list(np.where(dfp > median))[0])
                median_v = list(list(np.where(dfp == median))[0])
                lhs_p = [part[i] for i in lhs_v]
                rhs_p = [part[i] for i in rhs_v]
                median_p = [part[i] for i in median_v]
                # i need to have |lhs| = |rhs| +- 1
                diff = len(lhs_p)-len(rhs_p)
                if diff < 0:
                    med_lhs = np.random.choice(median_p, size=np.abs(
                        diff), replace=False)  # first even up |lhs_p| and |rhs_p|
                    # prepare remaining indices for equal split
                    med_to_split = [i for i in median_p if i not in med_lhs]
                    lhs_p = np.concatenate((lhs_p, med_lhs))
                else:  # same but |rhs_p| needs to be levelled up to |lhs_p|
                    med_rhs = np.random.choice(
                        median_p, size=np.abs(diff), replace=False)
                    med_to_split = [i for i in median_p if i not in med_rhs]
                    rhs_p = np.concatenate((rhs_p, med_rhs))
                # split remaining median indices equally between lhs and rhs
                med_lhs_1 = np.random.choice(med_to_split, size=(
                    len(med_to_split)//2), replace=False)
                med_rhs_1 = [i for i in med_to_split if i not in med_lhs_1]
                lhs = np.concatenate((lhs_p, med_lhs_1))
                rhs = np.concatenate((rhs_p, med_rhs_1))
            else:
                lhs, rhs = splitVal(df, dim, part, cat_indices, 'relaxed')
        return [int(x) for x in lhs], [int(x) for x in rhs]

    # create k-anonymous equivalence classes
    def partitioning(df, k, cat_indices, mode):

        final_partitions = []
        # start with full dataset
        working_partitions = [[x for x in range(len(df))]]

        while len(working_partitions) > 0:  # while there is at least one working partition left

            partition = working_partitions[0]  # take the first in the list
            # remove it from list of working partitions
            working_partitions = working_partitions[1:]

            if len(partition) < 2*k:  # if it is not at least 2k long, i.e. if i cannot get any new acceptable partition pair, at least k-long each
                # append it to final set of partitions
                final_partitions.append(partition)
                # and skip to the next partition
            else:
                # else, get spans of the feature columns restricted to this partition
                spans = colSpans(df, cat_indices, partition)
                # sort col indices in descending order based on their span
                ordered_span_cols = sorted(
                    spans.items(), key=lambda x: x[1], reverse=True)
                for dim, _ in ordered_span_cols:  # select the largest first, then second largest, ...
                    # try to split this partition
                    lhs, rhs = splitVal(df, dim, partition, cat_indices, mode)
                    # if new partitions are not too small (<k items), this partitioning is okay
                    if len(lhs) >= k and len(rhs) >= k:
                        working_partitions.append(lhs)
                        # re-append both new partitions to set of working partitions for further partitioning
                        working_partitions.append(rhs)
                        break  # break for loop and go to next partition, if available
                else:  # if no column could provide an allowable partitioning
                    # add the whole partition to the list of final partitions
                    final_partitions.append(partition)

        return final_partitions

    # print('Setting up partitioning...')

    # build k-anonymous equivalence classes
    k = int(arguments[2])
    if k > len(df):
        print('Invalid input. k must not exceed dataset size. Setting k to default 10.')
        k = 10

    modeArg = str(arguments[3])
    if modeArg not in ['s', 'r']:
        print("Invalid input. Partitioning mode must be 'r' for relaxed or 's' for strict.")
        print("Setting relaxed mode as default.")
    mode = 'relaxed'
    if modeArg == 's':
        mode = 'strict'

    equivalence_classes = partitioning(df, k, cat_indices, mode)
    sizes = []
    for part in equivalence_classes:
        sizes.append(len(part))
    min_size = np.min(sizes)
    # print('Partitioning completed.')
    print('{} equivalence classes were created. Minimum size is {}.'.format(
        len(equivalence_classes), min_size))

    # generate the anonymised dataset
    def anonymize_df(df, partitions, cat_indices, mode='range'):

        anon_df = []
        categorical = cat_indices

        for ip, p in enumerate(partitions):
            aggregate_values_for_partition = []
            partition = df[p]
            for column in range(len(types)):
                if column in categorical:
                    values = list(np.unique(partition[:, column]))
                    aggregate_values_for_partition.append(','.join(values))
                else:
                    if mode == 'mean':
                        aggregate_values_for_partition.append(
                            np.mean(partition[:, column]))
                    else:
                        col_min = np.min(partition[:, column])
                        col_max = np.max(partition[:, column])

                        if col_min == col_max:
                            aggregate_values_for_partition.append(col_min)
                        else:
                            aggregate_values_for_partition.append(
                                '{}-{}'.format(col_min, col_max))
            for i in range(len(p)):
                anon_df.append([int(p[i])]+aggregate_values_for_partition)

        df_anon = pd.DataFrame(anon_df)
        dfn1 = df_anon.sort_values(df_anon.columns[0])
        dfn1 = dfn1.iloc[:, 1:]
        return np.array(dfn1)

    # anonymise dataset
    aggregationArg = str(arguments[4])
    if aggregationArg not in ['m', 'r']:
        print("Invalid input. Aggregation metrics must either be 'r' for range or 'm' for mean.")
        print("Setting range metrics as default.")
    aggregation = 'range'
    if aggregationArg == 'm':
        aggregation = 'mean'

    dfn = anonymize_df(df, equivalence_classes, cat_indices, aggregation)
    dfn = pd.DataFrame(dfn)
    dfn.columns = dfcols
    dfn.to_csv("anon.csv")

    st.write("### Anonymized dataset")
    dfn


def mondrianlapace(df, k_anonymity, partition_mode, agg_mode):
    arguments = ["", df, k_anonymity, partition_mode, agg_mode]

    # remove NaNs
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    df = df.iloc[:, 1:]
    dfcols = df.columns
    st.write("### Original dataset")
    st.write(df)

    # infer data types
    types = list(df.dtypes)
    # print(types)
    cat_indices = [i for i in range(len(types)) if types[i] == "object"]

    # convert df to numpy array
    df = np.array(df)

    # function to compute the span of a given column while restricted to a subset of rows (a data partition)
    def colSpans(df, cat_indices, partition):
        spans = dict()
        for column in range(len(types)):
            dfp = df[partition, column]  # restrict df to the current column
            if column in cat_indices:
                # span of categorical variables is its number of unique classes
                span = len(np.unique(dfp))
            else:
                # span of numerical variables is its range
                span = np.max(dfp) - np.min(dfp)
            spans[column] = span
        return spans

    # function to split rows of a partition based on median value (categorical vs. numerical attributes)

    def splitVal(df, dim, part, cat_indices, mode):
        # restrict whole dataset to a single attribute and rows in this partition
        dfp = df[part, dim]
        unique = list(np.unique(dfp))
        length = len(unique)
        if dim in cat_indices:  # for categorical variables
            if mode == 'strict':  # i do not mind about |lhs| and |rhs| being equal
                lhv = unique[:length//2]
                rhv = unique[length//2:]
                lhs_v = list(list(np.where(np.isin(dfp, lhv)))
                             [0])  # left partition
                rhs_v = list(list(np.where(np.isin(dfp, rhv)))
                             [0])  # right partition
                lhs = [part[i] for i in lhs_v]
                rhs = [part[i] for i in rhs_v]
            elif mode == 'relaxed':  # i want |lhs| = |rhs| +-1
                lhv = unique[:length//2]
                rhv = unique[length//2:]
                lhs_v = list(list(np.where(np.isin(dfp, lhv)))
                             [0])  # left partition
                rhs_v = list(list(np.where(np.isin(dfp, rhv)))
                             [0])  # right partition
                lhs = [part[i] for i in lhs_v]
                rhs = [part[i] for i in rhs_v]
                diff = len(lhs)-len(rhs)
                if diff == 0:
                    pass
                elif diff < 0:
                    # move first |diff|/2 indices from rhs to lhs
                    lhs1 = rhs[:(np.abs(diff)//2)]
                    rhs = rhs[(np.abs(diff)//2):]
                    lhs = np.concatenate((lhs, lhs1))
                else:
                    rhs1 = lhs[-(diff//2):]
                    lhs = lhs[:-(diff//2)]
                    rhs = np.concatenate((rhs, rhs1))
            else:
                lhs, rhs = splitVal(df, dim, part, cat_indices, 'relaxed')
        # for numerical variables, split based on median value (strict or relaxed)
        else:
            median = np.median(dfp)
            # strict partitioning (do not equally split indices of median values)
            if mode == 'strict':
                lhs_v = list(list(np.where(dfp < median))[0])
                rhs_v = list(list(np.where(dfp >= median))[0])
                lhs = [part[i] for i in lhs_v]
                rhs = [part[i] for i in rhs_v]
            elif mode == 'relaxed':  # exact median values are equally split between the two halves
                lhs_v = list(list(np.where(dfp < median))[0])
                rhs_v = list(list(np.where(dfp > median))[0])
                median_v = list(list(np.where(dfp == median))[0])
                lhs_p = [part[i] for i in lhs_v]
                rhs_p = [part[i] for i in rhs_v]
                median_p = [part[i] for i in median_v]
                # i need to have |lhs| = |rhs| +- 1
                diff = len(lhs_p)-len(rhs_p)
                if diff < 0:
                    med_lhs = np.random.choice(median_p, size=np.abs(
                        diff), replace=False)  # first even up |lhs_p| and |rhs_p|
                    # prepare remaining indices for equal split
                    med_to_split = [i for i in median_p if i not in med_lhs]
                    lhs_p = np.concatenate((lhs_p, med_lhs))
                else:  # same but |rhs_p| needs to be levelled up to |lhs_p|
                    med_rhs = np.random.choice(
                        median_p, size=np.abs(diff), replace=False)
                    med_to_split = [i for i in median_p if i not in med_rhs]
                    rhs_p = np.concatenate((rhs_p, med_rhs))
                # split remaining median indices equally between lhs and rhs
                med_lhs_1 = np.random.choice(med_to_split, size=(
                    len(med_to_split)//2), replace=False)
                med_rhs_1 = [i for i in med_to_split if i not in med_lhs_1]
                lhs = np.concatenate((lhs_p, med_lhs_1))
                rhs = np.concatenate((rhs_p, med_rhs_1))
            else:
                lhs, rhs = splitVal(df, dim, part, cat_indices, 'relaxed')
        return [int(x) for x in lhs], [int(x) for x in rhs]

    # create k-anonymous equivalence classes
    def partitioning(df, k, cat_indices, mode):

        final_partitions = []
        # start with full dataset
        working_partitions = [[x for x in range(len(df))]]

        while len(working_partitions) > 0:  # while there is at least one working partition left

            partition = working_partitions[0]  # take the first in the list
            # remove it from list of working partitions
            working_partitions = working_partitions[1:]

            if len(partition) < 2*k:  # if it is not at least 2k long, i.e. if i cannot get any new acceptable partition pair, at least k-long each
                # append it to final set of partitions
                final_partitions.append(partition)
                # and skip to the next partition
            else:
                # else, get spans of the feature columns restricted to this partition
                spans = colSpans(df, cat_indices, partition)
                # sort col indices in descending order based on their span
                ordered_span_cols = sorted(
                    spans.items(), key=lambda x: x[1], reverse=True)
                for dim, _ in ordered_span_cols:  # select the largest first, then second largest, ...
                    # try to split this partition
                    lhs, rhs = splitVal(df, dim, partition, cat_indices, mode)
                    # if new partitions are not too small (<k items), this partitioning is okay
                    if len(lhs) >= k and len(rhs) >= k:
                        working_partitions.append(lhs)
                        # re-append both new partitions to set of working partitions for further partitioning
                        working_partitions.append(rhs)
                        break  # break for loop and go to next partition, if available
                else:  # if no column could provide an allowable partitioning
                    # add the whole partition to the list of final partitions
                    final_partitions.append(partition)

        return final_partitions

    # print('Setting up partitioning...')

    # build k-anonymous equivalence classes
    k = int(arguments[2])
    if k > len(df):
        print('Invalid input. k must not exceed dataset size. Setting k to default 10.')
        k = 10

    modeArg = str(arguments[3])
    if modeArg not in ['s', 'r']:
        print("Invalid input. Partitioning mode must be 'r' for relaxed or 's' for strict.")
        print("Setting relaxed mode as default.")
    mode = 'relaxed'
    if modeArg == 's':
        mode = 'strict'

    equivalence_classes = partitioning(df, k, cat_indices, mode)
    sizes = []
    for part in equivalence_classes:
        sizes.append(len(part))
    min_size = np.min(sizes)
    # print('Partitioning completed.')
    print('{} equivalence classes were created. Minimum size is {}.'.format(
        len(equivalence_classes), min_size))

    # generate the anonymised dataset
    def anonymize_df(df, partitions, cat_indices, mode='range'):

        anon_df = []
        categorical = cat_indices

        for ip, p in enumerate(partitions):
            aggregate_values_for_partition = []
            partition = df[p]
            for column in range(len(types)):
                if column in categorical:
                    values = list(np.unique(partition[:, column]))
                    aggregate_values_for_partition.append(','.join(values))
                else:
                    if mode == 'mean':
                        aggregate_values_for_partition.append(
                            np.mean(partition[:, column]))
                    else:
                        col_min = np.min(partition[:, column])
                        col_max = np.max(partition[:, column])

                        if col_min == col_max:
                            aggregate_values_for_partition.append(col_min)
                        else:
                            aggregate_values_for_partition.append(
                                '{}-{}'.format(col_min, col_max))
            for i in range(len(p)):
                anon_df.append([int(p[i])]+aggregate_values_for_partition)

        df_anon = pd.DataFrame(anon_df)
        dfn1 = df_anon.sort_values(df_anon.columns[0])
        dfn1 = dfn1.iloc[:, 1:]
        return np.array(dfn1)

    # anonymise dataset
    aggregationArg = str(arguments[4])
    if aggregationArg not in ['m', 'r']:
        print("Invalid input. Aggregation metrics must either be 'r' for range or 'm' for mean.")
        print("Setting range metrics as default.")
    aggregation = 'range'
    if aggregationArg == 'm':
        aggregation = 'mean'

    start = time.time()

    dfn = anonymize_df(df, equivalence_classes, cat_indices, aggregation)
    dfn = pd.DataFrame(dfn)
    dfn.columns = dfcols
    dfn.to_csv("anon.csv")
    # Compute hash of the anonymized data
    hash_object = hashlib.sha256(dfn.to_csv().encode())
    hash_value = hash_object.hexdigest()

    # Append hash value to the CSV file as a new line
    with open("anon.csv", "a") as file:
        file.write("\n")
        file.write("#Hash: {}".format(hash_value))

    end = time.time()

    print('Anonymization completed.')
    print('Execution time: {:.2f} seconds'.format(end - start))
    st.write("### Anonymized dataset")
    dfn
# Load the original dataset and the anonymized dataset
    # original_data = pd.read_csv('data.csv')
    original_data = df.copy()
    anonymized_data = pd.read_csv('anon.csv')
    # print(len(original_data), len(anonymized_data))
    #     # Specify the quasi-identifier and sensitive attributes
    qi_attributes = anonymized_data.columns
    #     # sensitive_attribute = 'occupation'
    # sensitive_attribute = sensitive_attr
    #     # Calculate the information loss
    # original_data_modes = original_data.groupby(qi_attributes)[
    #         sensitive_attribute].apply(lambda x: x.mode().iloc[0]).reset_index()
    # anonymized_data_modes = anonymized_data.groupby(qi_attributes)[
    #         sensitive_attribute].apply(lambda x: x.mode().iloc[0]).reset_index()
    # num_rows = min(len(original_data_modes), len(anonymized_data_modes))
    # original_data_modes = original_data_modes.sample(
    #         num_rows, random_state=42).reset_index()
    # anonymized_data_modes = anonymized_data_modes.sample(
    #         num_rows, random_state=42).reset_index()
    # iloss = np.sum(original_data_modes !=
    #                    anonymized_data_modes) / (num_rows)

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
    # st.write('## Information Loss')
    # st.write(iloss.drop('index'))
    st.write('## Re-identification Risk')
    st.write(create_scatter_plot(risks))


def mondrian_util(file):
    st.title("Mondrian Anonymity")

    # K-anonymity level
    k_anonymity = st.number_input(
        "K-Anonymity Level", min_value=1, step=1, value=5)

    # Partition mode
    partition_mode = st.selectbox("Partition Mode", ["Strict", "Relaxed"])
    partition_mode = partition_mode.lower()[0] if partition_mode else "r"

    # Aggregation mode
    agg_mode = st.selectbox("Aggregation Mode", ["Range", "Mean"])
    agg_mode = agg_mode.lower()[0] if agg_mode else "m"

    # Run the keanon function when the user clicks the "Run" button
    if st.button("Submit"):
        # Load the dataset
        try:
            df = pd.read_csv(file)
        except:
            st.error("Failed to load the dataset.")
            return

        # Call the mondrian function
        mondrianlapace(df, k_anonymity, partition_mode, agg_mode)


uploaded_file = st.file_uploader("Upload dataset", type="csv")
if uploaded_file is not None:
   # Read the CSV file into a Pandas dataframe
    algorithm = st.selectbox(
        'Select the k-anonymization algorithm:', ('Basic',  'Mondrian Anonymity'))
    if algorithm == "Basic":
        k_anonymity_algo()
    elif algorithm == "OLA":
        ola_anonymity()
    elif algorithm == "Mondrian Anonymity":
        mondrian_util(uploaded_file)
