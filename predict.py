import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from proofofconcept import analyze
import os
import argparse

def add_columns_with_default(df, columns, default_value=0):
    # Add columns with the default value
    for column in columns:
        df[column] = default_value
    return df

def load_classifier(pickle_path):
    """Load the MLP classifier from a pickle file."""
    with open(pickle_path, 'rb') as f:
        classifier = pickle.load(f)
    return classifier

def reduce_features(input_df, features):
    """Reduce the input DataFrame to the specified features."""
    reduced_df = input_df[features].copy()
    return reduced_df

def classify_data(classifier, input_df):
    """Classify the data using the provided classifier."""
    # Assuming the classifier requires scaling (e.g., StandardScaler was used during training)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(input_df)
    
    predictions = classifier.predict(scaled_data)
    return predictions

def main(pickle_path, input_df):
    # Load the classifier
    classifier = load_classifier(pickle_path)

    # Get the features required by the classifier
    features = classifier.feature_names_in_

    #columns_to_add = ['packer_type_Armadillov1xxv2xx', 'packer_type_NETDLLMicrosoft', 'packer_type_UPXv20MarkusLaszloReiser']

    #input_df = add_columns_with_default(input_df, columns_to_add)

    # Reduce the DataFrame to those features
    reduced_df = reduce_features(input_df, features)

    # Classify the data
    predictions = classify_data(classifier, reduced_df)
    
    # Add predictions to the original DataFrame
    reduced_df['prediction'] = predictions
    
    return reduced_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify data using a pre-trained MLP model.')
    parser.add_argument('source_path', type=str, help='Path to the source directory containing sample files.')
    #parser.add_argument('pickle_path', type=str, help='Path to the pickle file containing the pre-trained MLP model.')

    args = parser.parse_args()

    abs_path = os.path.abspath(args.source_path)
    input_df = analyze.pe_features(abs_path).extract_all()  # Replace with your DataFrame loading logic

    # Path to the pickle file
    pickle_path = 'mlp_model.pkl'

    # Classify the data and get the result DataFrame
    result_df = main(pickle_path, input_df)

    # Set options to display the entire DataFrame
    pd.set_option('display.max_rows', None)  # Show all rows
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.expand_frame_repr', False)  # Avoid wrapping of DataFrame



    # Display the first few rows of the result
    print(result_df['prediction'].values)