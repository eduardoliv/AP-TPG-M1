#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Grupo 03

Cleaning pipeline for various dataset formats

Processes files from:
  - class_raw_datasets (which may contain combined [ID, Text, Label] files or split ones)
  - original_input_datasets (files with ID and Text)
  - original_output_datasets (files with ID and Label)
The outputs are written with a tab delimiter (.csv extension) and cleaned accordingly.

Note: Previous cleaning has been manually applied for cases where tabs or semicolons are inside the Text column. 
"""

import argparse
import os
import glob
import re
import csv
import chardet
import pandas as pd
from io import StringIO

def detect_file_encoding(file_path):
    """Detect file encoding using chardet."""
    with open(file_path, 'rb') as f:
        rawdata = f.read()
    result = chardet.detect(rawdata)
    return result['encoding']

def join_lines_into_records(file_path, encoding):
    """
    Read file lines and merge them into complete records.
    A new record is assumed to begin with a valid ID pattern.
    Also, extra tabs are collapsed and if a quoted field is not complete,
    lines are joined until the quotes are balanced.
    """
    with open(file_path, 'rb') as f:
        raw = f.read()
    try:
        content = raw.decode(encoding)
    except Exception:
        content = raw.decode("utf-8", errors="replace")
    lines = content.splitlines()
    records = []
    buffer = ""
    # Regex for a valid ID: a sequence of digits OR letters+digits with optional dash and digits.
    id_pattern = re.compile(r'^(?:\d+|[A-Za-z]+\d+(?:-\d+)?)')
    for line in lines:
        # Collapse multiple tabs into one and strip trailing newline characters.
        line = re.sub(r'\t+', '\t', line).rstrip("\r\n")
        if not line.strip():
            continue
        # If the line starts with a valid ID, it might be a new record.
        if id_pattern.match(line):
            if buffer:
                # Check if quotes are balanced.
                if buffer.count('"') % 2 == 0:
                    records.append(buffer)
                    buffer = line
                else:
                    buffer += " " + line
            else:
                buffer = line
        else:
            # Continuation of the current record.
            buffer += " " + line
    if buffer:
        records.append(buffer)
    return records

def detect_delimiter(record):
    """
    Detect the delimiter using csv.Sniffer restricted to tab and semicolon.
    If detection fails, fallback on comparing counts.
    """
    try:
        dialect = csv.Sniffer().sniff(record, delimiters=["\t", ";"])
        return dialect.delimiter
    except csv.Error:
        if record.count("\t") >= record.count(";"):
            return "\t"
        else:
            return ";"

def parse_records_to_df(records, delimiter):
    """
    Join records into a single string and parse them using csv.reader,
    which properly handles quotes. Returns a pandas DataFrame.
    """
    data = "\n".join(records)
    reader = csv.reader(StringIO(data), delimiter=delimiter, quotechar='"')
    rows = list(reader)
    df = pd.DataFrame(rows)
    return df

def clean_text(text):
    """Clean text by stripping and normalizing newlines and removing extra tabs."""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'\t+', ' ', text)
    return text

def get_clean_filename(original_filename, target_type):
    """
    Generate a new filename with the proper suffix.
    target_type should be "inputs" or "outputs".
    If the original filename already ends with "_inputs" or "_outputs",
    adjust accordingly.
    """
    base, _ = os.path.splitext(os.path.basename(original_filename))
    if base.endswith("_inputs"):
        new_base = base if target_type == "inputs" else base[:-7] + "_outputs"
    elif base.endswith("_outputs"):
        new_base = base if target_type == "outputs" else base[:-8] + "_inputs"
    else:
        new_base = base + "_" + target_type
    return new_base + ".csv"

def remove_header_if_present(df, expected):
    """
    If the first row of df matches the expected header (case-insensitive),
    remove it.
    """
    if df.shape[0] > 0:
        first_row = df.iloc[0].astype(str).str.strip().tolist()
        if all(a.lower() == b.lower() for a, b in zip(first_row, expected)):
            df = df.iloc[1:].reset_index(drop=True)
    return df

def process_file(file_path, clean_input_dir, clean_output_dir, human_value, ai_value):
    """
    Process a single file:
      - Preprocess raw lines to merge broken records.
      - Detect delimiter and parse records using csv.reader.
      - Remove header row if present.
      - Based on the number of columns:
          • If 2 columns, use a heuristic on the second field (if it equals a label marker, treat as output; otherwise, input).
          • If 3 or more columns, assume [ID, Text, Label] and split into two outputs.
    """
    encoding = detect_file_encoding(file_path)
    records = join_lines_into_records(file_path, encoding)
    if not records:
        print(f"No records found in {file_path}. Skipping.")
        return

    # Remove header record if present.
    if records[0].strip().lower().startswith("id"):
        records = records[1:]
    if not records:
        print(f"No data after header removal in {file_path}.")
        return

    delimiter = detect_delimiter(records[0])
    df = parse_records_to_df(records, delimiter)
    num_cols = df.shape[1]
    original_filename = os.path.basename(file_path)

    # Remove accidental header rows.
    if num_cols == 2:
        first_row = df.iloc[0].astype(str).str.strip().tolist()
        if first_row[0].lower() == "id" and first_row[1].lower() in {"text", "label"}:
            df = df.iloc[1:].reset_index(drop=True)
    elif num_cols >= 3:
        first_row = df.iloc[0].astype(str).str.strip().tolist()
        if len(first_row) >= 3 and first_row[0].lower() == "id" and first_row[1].lower() == "text" and first_row[2].lower() == "label":
            df = df.iloc[1:].reset_index(drop=True)

    if num_cols == 2:
        sample_val = str(df.iloc[0, 1]).strip()
        if sample_val in {human_value, ai_value}:
            df[0] = df[0].astype(str).str.strip()
            df[1] = df[1].astype(str).str.strip()
            df_clean = df.rename(columns={0: "ID", 1: "Label"})
            out_path = os.path.join(clean_output_dir, get_clean_filename(original_filename, "outputs"))
            df_clean.to_csv(out_path, sep="\t", index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
        else:
            df[0] = df[0].astype(str).str.strip()
            df[1] = df[1].astype(str).apply(clean_text)
            df_clean = df.rename(columns={0: "ID", 1: "Text"})
            out_path = os.path.join(clean_input_dir, get_clean_filename(original_filename, "inputs"))
            df_clean.to_csv(out_path, sep="\t", index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    elif num_cols >= 3:
        df = df.iloc[:, :3]
        df.columns = ["ID", "Text", "Label"]
        df["ID"] = df["ID"].astype(str).str.strip()
        df["Text"] = df["Text"].astype(str).apply(clean_text)
        df["Label"] = df["Label"].astype(str).str.strip()
        # Normalize label if marker is present.
        df["Label"] = df["Label"].apply(lambda x: "Human" if human_value in x else ("AI" if ai_value in x else x))
        out_input_path = os.path.join(clean_input_dir, get_clean_filename(original_filename, "inputs"))
        out_output_path = os.path.join(clean_output_dir, get_clean_filename(original_filename, "outputs"))
        df[["ID", "Text"]].to_csv(out_input_path, sep="\t", index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
        df[["ID", "Label"]].to_csv(out_output_path, sep="\t", index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    else:
        print(f"Skipping file {file_path} due to insufficient columns.")

def process_directory(directory, clean_input_dir, clean_output_dir, human_value, ai_value):
    """Process all CSV files within a directory."""
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist. Skipping.")
        return
    for file_path in glob.glob(os.path.join(directory, "*.csv")):
        process_file(file_path, clean_input_dir, clean_output_dir, human_value, ai_value)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_raw_dir", default="class_raw_datasets", help="Directory for class raw datasets")
    parser.add_argument("--original_input_dir", default="original_input_datasets", help="Directory for original input datasets")
    parser.add_argument("--original_output_dir", default="original_output_datasets", help="Directory for original output datasets")
    parser.add_argument("--clean_input_dir", default="clean_input_datasets", help="Directory for cleaned input datasets")
    parser.add_argument("--clean_output_dir", default="clean_output_datasets", help="Directory for cleaned output datasets")
    parser.add_argument("--encoding", default="utf-8", help="Encoding to use")
    parser.add_argument("--human_value", default="Human", help="Value representing Human")
    parser.add_argument("--ai_value", default="AI", help="Value representing AI")
    args = parser.parse_args()

    for d in [args.clean_input_dir, args.clean_output_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    process_directory(args.original_input_dir, args.clean_input_dir, args.clean_output_dir, args.human_value, args.ai_value)
    process_directory(args.original_output_dir, args.clean_input_dir, args.clean_output_dir, args.human_value, args.ai_value)
    process_directory(args.class_raw_dir, args.clean_input_dir, args.clean_output_dir, args.human_value, args.ai_value)
