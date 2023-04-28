import streamlit as st
import pandas as pd
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from datetime import datetime
import numpy as np
from diffprivlib.mechanisms import Laplace
from faker import Faker
import pandas_profiling as pp
import re

st.title("Data Anonymizer")
fake= Faker()
# Load the Twitter data CSV file
uploaded_file = st.file_uploader("Upload Tweets dataset", type="csv")
if uploaded_file is not None:
   # Read the CSV file into a Pandas dataframe
    df = pd.read_csv(uploaded_file, nrows=20)


    print("---------------------------------------------------------------------------------------")

    print(list(df.columns))

    print(df.dtypes)

    pattern = r".*(name|address|phone_number|ssn|code|pin|pan|aadhaar|account|number|mail|location|date|time|user|id).*" # Example pattern for common quasi-identifiers
    # Create an empty list to store the identified quasi-identifiers
    quasi_identifiers_regex = []
    # Loop over each column name in the dataset
    for col in df.columns:
        # Search for the pattern in the column name
        matches = re.findall(pattern, col, flags=re.IGNORECASE)
        # If any matches are found, add the column name to the quasi-identifiers list
        if matches:
            quasi_identifiers_regex.append(col)
    # Print the identified quasi-identifiers
    print("Identified quasi-identifiers:", quasi_identifiers_regex)




    # Generate a report using pandas-profiling
    report = pp.ProfileReport(df)
    # Print the report and look for columns with high cardinality or unique values
    report.to_file("outputbefore.html")

    print("---------------------------------------------------------------------------------------")


    # Show the original data table
    st.write("Original data table")
    st.write(df)

    # Define the regex pattern in a Presidio `Pattern` object:
    numbers_pattern = Pattern(name="numbers_pattern", regex="[A-Z]{5}[0-9]{4}[A-Z]{1}", score=0.5)

    # Define the recognizer with one or more patterns
    number_recognizer = PatternRecognizer(supported_entity="PAN_NUMBER", patterns=[numbers_pattern])

    # Analyzer output
    analyzer = AnalyzerEngine()
    analyzer.registry.add_recognizer(number_recognizer)

    # Anonymizer output
    anonymizer = AnonymizerEngine()

    # Define anonymization operators
    operators = {
        "LOCATION": OperatorConfig("replace", {"new_value": "<LOCATION>"}),
        "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
        "PHONE_NUMBER": OperatorConfig(
            "mask",
            {
                "type": "mask",
                "masking_char": "*",
                "chars_to_mask": 7,
                "from_end": True,
            },
        ),
        "EMAIL_ADDRESS": OperatorConfig(
            "mask",
            {
                "chars_to_mask": 7,
                "type": "mask",
                "from_end": False,
                "masking_char": "*"
            }
        ),
        "TITLE": OperatorConfig("redact", {}),
    }

    # Create a new column "Anonymized Tweet" with the anonymized tweet
    anonymized_tweets = []
    for tweet in df["Tweet Content"]:
        analyzer_results = analyzer.analyze(text=tweet, language="en")
        anonymized_results = anonymizer.anonymize(
            text=tweet, analyzer_results=analyzer_results, operators=operators
        )
        anonymized_tweet = anonymized_results.text
        anonymized_tweets.append(anonymized_tweet)
    df["Anonymized Tweet"] = anonymized_tweets

    # Replace the "Tweet Content" column with the "Anonymized Tweet" column
    df.drop(columns=["Tweet Content"], inplace=True)
    df.rename(columns={"Anonymized Tweet": "Tweet Content"}, inplace=True)

    # Show the final data table with the anonymized tweets in the "Tweet Content" column
    st.write("PII identified tweets")
    st.write(df)

    # k-anonymity
    # Identify the sensitive attributes
    sensitive_attributes = ['User Id', 'Name', 'Twitter Username', 'User Bio', 'Profile URL']

    # Determine the K value
    K = 5
    
    # Identify the quasi-identifiers
    quasi_identifiers = ['User Id','Name','Tweet Location', 'User Followers', 'User Following', 'User Account Creation Date', 'Tweet Posted Time (UTC)']

    # Group the individuals based on the quasi-identifiers
    groups = df.groupby(quasi_identifiers)

    # Function to generalize Tweet Location to broader geographic regions
    def generalize_location(location):
            geolocator = Nominatim(user_agent="my-app")
            try:
                location = geolocator.geocode(location)
                if location is not None:
                    lat= location.raw.get('lat')
                    lon= location.raw.get('lon')
                    loc = lat + ',' + lon
                    country = geolocator.reverse(loc)
                    return country.raw.get('address').get('country')
                return "Unknown"
            except (GeocoderTimedOut, GeocoderServiceError):
                return "Unknown"


    # Function to generalize User Followers and User Following to ranges
    def generalize_follower_following(count):
        sensitivity=1
        epsilon=0.1
        noise = Laplace(epsilon=epsilon, sensitivity=sensitivity).randomise(value=1)
        return count + noise.astype(int)

        if count < 100:
            return '0-99'
        elif count >= 100 and count < 500:
            return '100-499'
        elif count >= 500 and count < 1000:
            return '500-999'
        else:
            return '1000+'

    # Function to generalize User Account Creation Date to month and year
    def generalize_account_creation_date(date):
        # Convert the input string to a datetime object
        dt_object = datetime.strptime(date, "%d-%m-%Y %H:%M")

        # Convert the datetime object to the desired format
        month_year = dt_object.strftime("%b, %Y")

        return month_year
    def generalize_name(x):
        return fake.name()
    # Generalize or suppress the values of the quasi-identifiers in each group
    generalized_data = []
    for name, group in groups:
        generalized_group = group.copy()
        for col in quasi_identifiers:
            if col == 'Tweet Location':
                # Generalize Tweet Location to broader geographic regions
                generalized_group[col] = group[col].apply(lambda x: generalize_location(x))
            elif col == 'User Followers' or col == 'User Following':
                # Generalize User Followers and User Following to ranges
                generalized_group[col] = group[col].apply(lambda x: generalize_follower_following(x))
            elif col == 'User Account Creation Date':
                # Generalize User Account Creation Date to month and year
                generalized_group[col] = group[col].apply(lambda x: generalize_account_creation_date(x))
            elif col == 'Tweet Posted Time (UTC)':
                # Generalize User Account Creation Date to month and year
                generalized_group[col] = group[col].apply(lambda x: generalize_account_creation_date(x))
            elif col == 'Name':
                # Generalize User Name
                generalized_group[col] = group[col].apply(lambda x: generalize_name(x))
            elif col == 'User Id':
                # Generalize User Id 
                generalized_group[col] = group[col].apply(lambda x: str(generalize_follower_following(int(x.strip('"')))).replace(",", "")[:8])
        generalized_data.append(generalized_group)

    # Create a new DataFrame with the anonymized quasi-identifiers and the non-sensitive attribute Tweet Content
    anonymized_data = pd.concat(generalized_data)[['User Id','Name','Tweet Location', 'User Followers', 'User Following', 'User Account Creation Date', 'Tweet Content', 'Tweet Posted Time (UTC)']]

    # Verify that each group satisfies the K-Anonymity condition and merge groups if necessary
    k_anonymized_data = []
    for name, group in anonymized_data.groupby(quasi_identifiers):
        if len(group) < K:
            # Merge the group with the nearest group that satisfies the K-Anonymity condition
            merged_group = pd.concat([group, anonymized_data.loc[anonymized_data.groupby(quasi_identifiers).groups[name]].iloc[0:K-len(group)]])
            k_anonymized_data.append(merged_group)
        else:
            k_anonymized_data.append(group)

    # Concatenate all the K-Anonymized groups into a final anonymized DataFrame
    df = pd.concat(k_anonymized_data)
    df = df.rename(columns={'Tweet Posted Time (UTC)': 'Tweet Posted Date'})
    df = df.drop_duplicates()
    st.write("Anonymized Dataset")


    # Generate a report using pandas-profiling
    report = pp.ProfileReport(df)
    # Print the report and look for columns with high cardinality or unique values
    report.to_file("outputafter.html")



    st.write(df)
















# import streamlit as st
# import pandas as pd
# import streamlit as st
# import plotly.graph_objects as go
# from streamlit_extras.switch_page_button import switch_page

# # Set page title
# st.set_page_config(page_title="CSV File Selector")

# # Create file uploader
# uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

# if uploaded_file is not None:
#     # Load data from file
#     df = pd.read_csv(uploaded_file)

#     # Display column headers
#     st.write("## Generalizes")
    
#     # Get column names
#     columns = df.columns.tolist()

#     # Create checkboxes for column selection
#     generalize_cols = st.multiselect("Select columns", columns)

#     # Display generalize columns
#     st.write("## Generalize Selected Columns")
#     st.write(generalize_cols)

#     # Display column headers
#     st.write("## Privacy")

#     remaining_cols = list(set(df.columns) - set(generalize_cols))
#     select_remaining = st.multiselect("Select all remaining columns", remaining_cols)

#     # Display privacy columns
#     st.write("## Privacy Selected Columns")
#     st.write(select_remaining)

#     # Display remaining columns
#     st.write("## Remaining Columns")
#     remaining_cols_final = list(set(df.columns) - set(generalize_cols)-set(select_remaining))
#     st.write(remaining_cols_final)
    
#     # Create checkbox for Privacy
#     st.write("## Anonymization Level")
#     anonymization_option1 = st.checkbox("K-Diversity", key="option1")
#     anonymization_option1 = st.checkbox("L-Diversity", key="option2")
#     anonymization_option1 = st.checkbox("K Annoynmization", key="option3")

#     # Create submit button
#     if st.button("Submit"):
#         # Create two pie charts
#         fig1 = go.Figure(go.Pie(
#             labels=["Label 1", "Label 2"],
#             values=[30, 40],
#             marker=dict(colors=["red", "purple"]),
#             textinfo="label+percent",
#             title="Pie Chart 1"))

#         fig2 = go.Figure(go.Pie(
#             labels=["Label 3", "Label 4"],
#             values=[50, 25],
#             marker=dict(colors=["blue", "purple"]),
#             textinfo="label+percent",
#             title="Pie Chart 2"))

# # Display pie charts side by side
#         st.write("## Pie Charts")
#         st.plotly_chart(fig1, use_container_width=True)
#         st.plotly_chart(fig2, use_container_width=True)

# # Create checkbox for Privacy
#         st.write("## New Section")


