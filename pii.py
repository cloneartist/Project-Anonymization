from pprint import pprint
import json
import pandas as pd
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig, RecognizerResult
from presidio_analyzer import AnalyzerEngine, Pattern,PatternRecognizer

data=pd.read_csv("covid_data.csv")
df=pd.DataFrame(data,columns=["Tweet Content","Tweet Location"])
df1=df.dropna()
df1.to_csv('ContentLocation.csv', index=False)

length_of_data=len(df1)
print(length_of_data)
for i in range(length_of_data):
    str1=df["Tweet Content"][i]
    str2=df["Tweet Location"][i]
    text_value = str(str1) +" "+ str(str2)
    text_to_analyze =text_value
    print("This is the text to analyse")
    print(text_to_analyze)
    print("Analysing done")
    numbers_pattern = Pattern(name="numbers_pattern",regex="[A-Z]{5}[0-9]{4}[A-Z]{1}", score=0.5)
    aadhar_pattern = Pattern(name="aadhar_pattern",regex="[0-9]{4}[ ][0-9]{4}[ ][0-9]{4}", score=0.5)
    # Define the recognizer with one or more patterns
    number_recognizer = PatternRecognizer(supported_entity="PAN_NUMBER", patterns = [numbers_pattern])
    aadhar_recognizer = PatternRecognizer(supported_entity="AADHAR_NUMBER", patterns = [aadhar_pattern])
    # Analyzer output
    analyzer = AnalyzerEngine()
    analyzer.registry.add_recognizer(number_recognizer)
    analyzer.registry.add_recognizer(aadhar_recognizer)

    analyzer_results = analyzer.analyze(text=text_to_analyze, language="en")

    print("Anonymised result")
    print(analyzer_results)
    print("Analyser result done")

    anonymizer = AnonymizerEngine()

    # Define anonymization operators
    operators = {
        #"LOCATION": OperatorConfig("replace", {"new_value": "<LOCATION>"}),
        #"PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
        "PHONE_NUMBER": OperatorConfig(
            "mask",
            {
                "type": "mask",
                "masking_char": "*",
                "chars_to_mask": 7,
                "from_end": True,
            },
        ),
        "EMAIL_ADDRESS": OperatorConfig("mask", {
                "chars_to_mask": 7,
                "type": "mask",
                "from_end" : False,
        "masking_char": "*"}),
        "TITLE": OperatorConfig("redact", {}),
    }
    #
    anonymized_results = anonymizer.anonymize(
        text=text_to_analyze, analyzer_results=analyzer_results, operators=operators
    )
    print("Anonymised result")
    print(anonymized_results)
    print("Anonymised result done")


