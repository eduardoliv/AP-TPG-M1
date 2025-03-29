#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Grupo 03
Description: Merge cleaned input and output datasets to create a training dataset for Models.
             The script processes only file pairs that exist in both clean_input_datasets and clean_output_datasets.
             It merges them by ID and filters rows based on the word count (ignoring punctuation) of the Text column.
             The filtered data is saved into test_input_dataset and test_output_dataset.
"""

import os
import glob
import re
import csv
import argparse
import pandas as pd

def count_words(text):
    """
    Count words in the text ignoring punctuation.
    Uses re.findall to count sequences of alphanumeric characters.
    """
    return len(re.findall(r'\w+', text))

def process_file_pair(input_file, output_file, min_words, max_words):
    """
    Process a single pair of input and output files:
      - Read both CSVs (tab-delimited)
      - Verify that headers match expected values: ["ID", "Text"] for input and ["ID", "Label"] for output.
      - Merge dataframes on the "ID" column.
      - Filter rows by word count in the Text column (based on min_words and max_words).
      - Return a DataFrame with columns: ["ID", "Text", "Label"].
    """
    try:
        df_input = pd.read_csv(input_file, sep="\t", dtype=str, quoting=csv.QUOTE_MINIMAL)
        df_output = pd.read_csv(output_file, sep="\t", dtype=str, quoting=csv.QUOTE_MINIMAL)
    except Exception as e:
        print(f"Error reading files {input_file} or {output_file}: {e}")
        return None

    expected_input_cols = ["ID", "Text"]
    expected_output_cols = ["ID", "Label"]

    if list(df_input.columns) != expected_input_cols:
        print(f"File {input_file} does not have expected columns {expected_input_cols}. Skipping.")
        return None
    if list(df_output.columns) != expected_output_cols:
        print(f"File {output_file} does not have expected columns {expected_output_cols}. Skipping.")
        return None

    # Merge on the "ID" column (inner join ensures that only matching IDs are considered)
    merged_df = pd.merge(df_input, df_output, on="ID", how="inner")

    # Count words in the "Text" column (ignoring punctuation) and filter rows
    merged_df['word_count'] = merged_df['Text'].apply(lambda x: count_words(x) if isinstance(x, str) else 0)
    filtered_df = merged_df[(merged_df['word_count'] >= min_words) & (merged_df['word_count'] <= max_words)]
    
    # Drop the auxiliary word_count column before returning
    filtered_df = filtered_df.drop(columns=["word_count"])
    
    return filtered_df

def main():
    parser = argparse.ArgumentParser(
        description="Merge cleaned datasets to create a shuffled training dataset based on word count criteria."
    )
    parser.add_argument("--clean_input_dir", default="clean_input_datasets", help="Directory for cleaned input datasets")
    parser.add_argument("--clean_output_dir", default="clean_output_datasets", help="Directory for cleaned output datasets")
    parser.add_argument("--test_input_dir", default="test_input_dataset", help="Directory to store merged input training dataset")
    parser.add_argument("--test_output_dir", default="test_output_dataset", help="Directory to store merged output training dataset")
    parser.add_argument("--min_words", type=int, default=90, help="Minimum number of words in the Text field")
    parser.add_argument("--max_words", type=int, default=135, help="Maximum number of words in the Text field")
    args = parser.parse_args()
    
    # Create output directories if they don't exist
    for d in [args.test_input_dir, args.test_output_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
    
    # Global list to accumulate merged data from all valid file pairs
    merged_data_list = []
    
    # Process all input files in the clean_input_dir following the naming convention (*_inputs.csv)
    input_files = glob.glob(os.path.join(args.clean_input_dir, "*.csv"))
    
    for input_file in input_files:
        base_name = os.path.basename(input_file)
        if not base_name.endswith("_inputs.csv"):
            print(f"File {input_file} does not follow expected naming convention (_inputs.csv). Skipping.")
            continue
        
        # Derive the common prefix (remove '_inputs.csv')
        common_prefix = base_name[:-11]
        # Build the corresponding output file name (common_prefix + '_outputs.csv')
        output_file = os.path.join(args.clean_output_dir, common_prefix + "_outputs.csv")
        if not os.path.exists(output_file):
            print(f"Corresponding output file {output_file} not found for input file {input_file}. Skipping.")
            continue
        
        print(f"Processing file pair: {input_file} and {output_file}")
        df_merged = process_file_pair(input_file, output_file, args.min_words, args.max_words)
        if df_merged is not None and not df_merged.empty:
            merged_data_list.append(df_merged)
    
    if not merged_data_list:
        print("No valid merged data found from the given file pairs.")
        return

        # Concatenate all merged data into a single DataFrame
    global_df = pd.concat(merged_data_list, ignore_index=True)
    
    # If dataset has more than 110 rows, balance the labels (equal number of "AI" and "Human")
    if len(global_df) > 110:
        count_ai = (global_df['Label'] == 'AI').sum()
        count_human = (global_df['Label'] == 'Human').sum()
        min_count = min(count_ai, count_human)
        print(f"Balancing dataset: AI count = {count_ai}, Human count = {count_human}. Using {min_count} samples from each class.")
        global_df = pd.concat([
            global_df[global_df['Label'] == 'AI'].sample(min_count, random_state=42),
            global_df[global_df['Label'] == 'Human'].sample(min_count, random_state=42)
        ]).reset_index(drop=True)
    
    # Shuffle the merged data (using a fixed seed for reproducibility)
    global_df = global_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split into separate DataFrames for input and output
    df_train_input = global_df[["ID", "Text"]]
    df_train_output = global_df[["ID", "Label"]]
    
    # Define output file paths (single file for each)
    out_input_path = os.path.join(args.test_input_dir, "merged_inputs.csv")
    out_output_path = os.path.join(args.test_output_dir, "merged_outputs.csv")
    
    # Save the resulting DataFrames
    df_train_input.to_csv(out_input_path, sep="\t", index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    df_train_output.to_csv(out_output_path, sep="\t", index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    print(f"Saved merged input file: {out_input_path}")
    print(f"Saved merged output file: {out_output_path}")

if __name__ == '__main__':
    main()