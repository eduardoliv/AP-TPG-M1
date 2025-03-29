#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Grupo 03

Validate ipsis verbis sentences on google custom search
"""

import requests
import pandas as pd
import time
import argparse

# API credentials
API_KEY = ""
CSE_ID = ""

def google_search(query, api_key, cse_id, **kwargs):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cse_id,
        "q": f'"{query}"'
    }
    params.update(kwargs)
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def check_exact_match(text, api_key, cse_id):
    try:
        results = google_search(text, api_key, cse_id)
        items = results.get("items", [])
        return (len(items) > 0), items
    except Exception as e:
        print("Error during search:", e)
        return False, []

def main():
    parser = argparse.ArgumentParser(description="Search each CSV row's text for an exact match using Google Custom Search API")
    parser.add_argument("--input", type=str, required=True, help="Path to the input CSV file (tab-separated)")
    parser.add_argument("--output", type=str, required=True, help="Path and file name for the output CSV file")
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output

    # Load the CSV file
    df = pd.read_csv(input_file, sep="\t")

    # Prepare a list to store search results
    results_list = []

    for idx, row in df.iterrows():
        id_val = row["ID"]
        text = row["Text"]

        match_found, items = check_exact_match(text, API_KEY, CSE_ID)
        num_results = len(items)
        
        results_list.append({
            "ID": id_val,
            "Exact_Match_Found": match_found,
            "Num_Results": num_results
        })
        
        print(f"ID {id_val} - Exact match: {match_found} - {num_results} results")
        # API rate limits
        time.sleep(1)

    # Save the results to the specified output CSV file
    pd.DataFrame(results_list).to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
