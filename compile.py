import pandas as pd
import os
import openpyxl
from datetime import datetime

def compiling(working_output_path, country, countrytype, user):

    # Get the current date
    now = datetime.now()
    month, day, year = now.strftime("%m"), now.strftime("%d"), now.strftime("%Y")

    # Load the workbook 
    workbook = openpyxl.load_workbook(working_output_path)
    dfs = [pd.read_excel(working_output_path, sheet_name=sheet_name, usecols="A:D", engine='openpyxl') for sheet_name in workbook.sheetnames]
    combined_df = pd.concat(dfs, ignore_index=True)

    # Define the path to the output CSV file
    sorted_root_path = r"../mnt/sortingpinning/Countries"
    sorted_folder = "Output"
    final_output_file = f"categoryposition_{year}{month}{day}_{countrytype}_01 {user}.csv"
    final_output_path = os.path.join(sorted_root_path, country, sorted_folder, final_output_file)

    # Save the output CSV file
    combined_df.to_csv(final_output_path, index=False)
    print('Compiling complete')