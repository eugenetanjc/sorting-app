# Import libraries
import streamlit as st
import pandas as pd
import os
import io
import base64
import backend
import compile
from datetime import datetime

# Define global variables
image_filename = 'data/ck_logo.png'
with open(image_filename, 'rb') as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode()

# Create dictionaries for input options
year = int(datetime.now().strftime("%Y")) 
week_options = [{'label': str(i), 'value': str(i)} for i in range(1, 53)]
year_options = [{'label': str(i), 'value': str(i)} for i in range(year-1, year+2)]

# Country options
country_options = [{'label': country, 'value': country} for country in [
    'INDIA', 'INDONESIA', 'JAPAN', 'SAUDI ARABIA', 'SINGAPORE', 'SOUTH KOREA', 'THAILAND', 'VIETNAM'
    ]]
countrytype_options = [{'label': countrytype, 'value': countrytype} for countrytype  in ['Hot', 'Cold', 'ANZ']]

def peform_sorting(contents, s_country, s_year, s_week, s_ctype):
    global username, input_country, input_ctype

    if not all([contents, s_country, s_year, s_week, s_ctype]):
        st.error("Please fill in all required fields.")

    # Check user input
    print(f'Selected Country: {s_country}, Year: {s_year}, Week: {s_week}, Seasonality: {s_ctype}')

    # Read parameters file into pandas df
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    xls = pd.ExcelFile(io.BytesIO(decoded), engine='openpyxl')
    params_dict = {sheet_name: pd.read_excel(xls, sheet_name=sheet_name) for sheet_name in xls.sheet_names}

    # Run sorting function
    sorted_output_path, sorted_all_path = backend.sorting(s_country, s_year, s_week, s_ctype, params_dict)
    st.success("Sorting performed successfully!")

def compile_file(filename, country, ctype):
    if not filename:
        st.error("Please upload a file.")

    if not all([country, ctype]):
        st.error("Please fill in all required fields.")
        
    compiled_file_path = compile.compiling(country, ctype)
    st.success("Compiling performed successfully!")

def sorting_layout():

    st.markdown(
    f'<div style="text-align: center; margin: 10px;">'
    f'<div><img src="data:image/png;base64,{encoded_image}" style="height: 40%; width: 40%;"/></div>'
    f'<br/>'
    f'<div style="text-align: center; margin-bottom: 15px; display: inline-block; font-family: Bahnschrift; font-weight: bold;">'
    f'<h1>ECOM Sorting App</h1>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True
    )
    st.markdown('<hr>', unsafe_allow_html=True)

    st.header("Sorting")

    # Upload Parameters File
    st.subheader("Upload Parameters File")
    uploaded_file = st.file_uploader("Drag and Drop or Select Parameters file", type=["xlsx", "xls"])
    if uploaded_file is not None:
        params_dict = pd.read_excel(uploaded_file, sheet_name=None)
        st.success("File uploaded successfully!")

    # Select your country
    st.subheader("Select your country")
    country = st.selectbox("Select a country", [opt['value'] for opt in country_options], index=0)

    # Select your year
    st.subheader("Select your year")
    year = st.selectbox("Select a year", [opt['value'] for opt in year_options], index=1)

    # Select your week
    st.subheader("Select your week")
    week = st.selectbox("Select a week", options=[opt['value'] for opt in week_options], index=23)

    # Select your country type
    st.subheader("Select your country type")
    country_type = st.selectbox("Select a country type", options=[opt['value'] for opt in countrytype_options], index=0)

    # Run sorting now button
    if st.button("Run sorting now"):
        st.write("Sorting is being executed...")

        outcome = peform_sorting(country, year, week, country_type, params_dict)
        st.success("Sorting completed successfully!")

    # Function to create a downloadable link for CSV file
    def get_csv_download_link(df, filename="sorted_data.csv"):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # Encode as base64
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV file</a>'
        return href

    # Display download link if outcome exists and user clicks the button
    if st.button("Download sorted data as CSV"):
        st.write("Downloading sorted data...")
        st.markdown(get_csv_download_link(outcome), unsafe_allow_html=True)

# Define the Streamlit app layout for the Compilation tab
def compile_layout():
    st.header("Compilation")

    # Select Country folder to save .csv file into
    st.subheader("Select Country folder to save .csv file into:")
    country_input = st.selectbox("Select a country for compile input", options=[opt['value'] for opt in country_options])

    # Select country type for your .csv file extension
    st.subheader("Select country type for your .csv file extension:")
    countrytype_input = st.selectbox("Select a country type for compile input", options=[opt['value'] for opt in countrytype_options])

    # Input your working file
    st.subheader("Input your working file (should be a .xlsx file) into the area below for compile:")
    uploaded_files = st.file_uploader("Drag and Drop or Select .xlsx working file (it can take 10-15 seconds to load)", type=["xlsx"], accept_multiple_files=True)

    # Compile button
    if st.button("Compile working file"):
        if uploaded_files:
            st.write("Compiling working file...")

            for uploaded_file in uploaded_files:
                df = pd.read_excel(uploaded_file)
                # Perform necessary processing on df
            st.success("Compilation complete!")
        else:
            st.error("Please upload at least one file.")

# Main app layout
def serve_layout():
    st.sidebar.title("ECOM Sorting App")
    tab_selection = st.sidebar.radio("Go to", ["Sorting", "Compilation"])

    if tab_selection == "Sorting":
        sorting_layout()

    elif tab_selection == "Compilation":
        compile_layout()

if __name__ == "__main__":
    serve_layout()