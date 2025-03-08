#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created by: Grupo 03

import argparse
import os
import glob
import pandas as pd
import chardet
import csv

def detect_file_encoding(file_path):
    with open(file_path, 'rb') as f:
        rawdata = f.read()
    result = chardet.detect(rawdata)
    return result['encoding']

def clean_dataset_pipeline(file_path, output_path, sep, encoding, id_column, text_label_column, human_value, ai_value, is_output=False):
    input_encoding = encoding
    if encoding.lower() == "auto":
        input_encoding = detect_file_encoding(file_path)

    df = pd.read_csv(file_path, sep=sep, encoding=input_encoding, engine='python')
    
    df = df[[id_column, text_label_column]]
    
    # Rename columns to standard names and clean values
    if is_output:
        df = df.rename(columns={id_column: "ID", text_label_column: "Label"})
        if human_value and ai_value:
            df["Label"] = df["Label"].astype(str).str.strip().apply(
                lambda x: "Human" if x == human_value else ("AI" if x == ai_value else x)
        )
    else:
        df = df.rename(columns={id_column: "ID", text_label_column: "Text"})
        df["Text"] = df["Text"].astype(str).str.strip()
    
    df.to_csv(output_path, sep=sep, index=False, encoding="utf-8", quoting=csv.QUOTE_NONE,)

def process_files_in_directory(original_dir, clean_dir, is_output, sep, encoding, id_column, text_label_column, human_value=None, ai_value=None):
    if not os.path.exists(clean_dir):
        os.makedirs(clean_dir)
    
    file_list = glob.glob(os.path.join(original_dir, "*.csv"))
    if not file_list:
        return
    
    for file_path in file_list:
        filename = os.path.basename(file_path)
        output_path = os.path.join(clean_dir, filename)
        clean_dataset_pipeline(file_path, output_path, sep, encoding, id_column, text_label_column, human_value, ai_value, is_output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Default directories
    parser.add_argument("--original_input_dir", default="original_input_datasets", help="Directory containing original input datasets (default: original_input_datasets)")
    parser.add_argument("--clean_input_dir", default="clean_input_datasets", help="Directory to save cleaned input datasets (default: clean_input_datasets)")
    parser.add_argument("--original_output_dir", default="original_output_datasets", help="Directory containing original output datasets (default: original_output_datasets)")
    parser.add_argument("--clean_output_dir", default="clean_output_datasets", help="Directory to save cleaned output datasets (default: clean_output_datasets)")
    
    # Delimiter and Encoding parameters
    parser.add_argument("--sep", default="\t", help="Delimiter used in the CSV (default: '\\t')")
    parser.add_argument("--encoding", default="utf-8", help="Encoding to use (default: 'utf-8')")
    
    # Parameters for input datasets
    parser.add_argument("--input_id_column", default="ID", help="Name of the ID column in input files (default: 'ID')")
    parser.add_argument("--input_text_column", default="Text", help="Name of the text column in input files (default: 'Text')")
    
    # Parameters for output datasets
    parser.add_argument("--output_id_column", default="ID", help="Name of the ID column in output files (default: 'ID')")
    parser.add_argument("--output_label_column", default="Label", help="Name of the label column in output files (default: 'Label')")
    parser.add_argument("--human_value", default="Human", help="Value representing 'Human' in output files (default: 'Human')")
    parser.add_argument("--ai_value", default="AI", help="Value representing 'AI' in output files (default: 'AI')")
    
    args = parser.parse_args()
    
    # Process input datasets
    process_files_in_directory(
        original_dir=args.original_input_dir,
        clean_dir=args.clean_input_dir,
        is_output=False,
        sep=args.sep,
        encoding=args.encoding,
        id_column=args.input_id_column,
        text_label_column=args.input_text_column
    )
    
    # Process output datasets
    process_files_in_directory(
        original_dir=args.original_output_dir,
        clean_dir=args.clean_output_dir,
        is_output=True,
        sep=args.sep,
        encoding=args.encoding,
        id_column=args.output_id_column,
        text_label_column=args.output_label_column,
        human_value=args.human_value,
        ai_value=args.ai_value
    )