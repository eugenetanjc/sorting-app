# Import libraries
import streamlit as st
import pandas as pd
import os
import io
from io import BytesIO
import requests
from PIL import Image as PILImage
import base64
import backend
import compile
from datetime import datetime
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl import load_workbook

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

def peform_sorting(s_country, s_year, s_week, s_ctype, params_dict):
    global username, input_country, input_ctype

    if not all([s_country, s_year, s_week, s_ctype]):
        st.error("Please fill in all required fields.")

    # Check user input
    print(f'Selected Country: {s_country}, Year: {s_year}, Week: {s_week}, Seasonality: {s_ctype}')

    # Run sorting function
    output_df = backend.sorting(s_country, s_year, s_week, s_ctype, params_dict)
    st.success("Sorting performed successfully!")
    return output_df

def compile_file(filename, country, ctype):
    if not filename:
        st.error("Please upload a file.")

    if not all([country, ctype]):
        st.error("Please fill in all required fields.")
        
    compiled_file_path = compile.compiling(country, ctype)
    st.success("Compiling performed successfully!")

# Function to create a downloadable link for CSV file
def get_csv_download_link(df, filename="sorted_data.xlsx"):
    
    # Loop through the rows of the DataFrame and split the data into sheets based on the 'Category ID' values
    sheets = {}
    for _, row in df.iterrows():
        category_id = row['Category ID']
        if category_id not in sheets:
            sheets[category_id] = pd.DataFrame(columns=df.columns)
        sheets[category_id] = pd.concat([sheets[category_id], row.to_frame().T])

    # Create a new workbook object for the output file
    output_workbook = Workbook()

    # Loop through the sheets and add them to the output workbook
    for category_id, sheet_df in sheets.items():
        sheet_df.drop_duplicates(subset=['Article'], keep='first', inplace=True)
        output_worksheet = output_workbook.create_sheet(category_id)
        row_index = 0

        # Loop through each row in the sheet DataFrame
        for index, row in sheet_df.iterrows():
            article_number = row['Article']
            directory = 'https://images.e-charleskeith.com/Article/'
            image_path = directory + str(article_number) + '.jpg'

            try: 
                response = requests.get(image_path, verify=False, timeout = (30, 30))

                if response.status_code == 200: 
                    pil_image = PILImage.open(BytesIO(response.content))
                    image_width, image_height = pil_image.size

                    if image_height > image_width:
                        pil_image = pil_image.rotate(90)

                    # Convert PIL image back to bytes
                    image_bytes = BytesIO()
                    rgb_img = pil_image.convert('RGB')
                    rgb_img.save(image_bytes, format='JPEG')
                    image_bytes.seek(0)
                    
                    img = Image(image_bytes)
                    
                    column_width = 2.0 / 2.54 * 160
                    row_height = 3.0 / 2.54 * 64

                    img.width = int(column_width)
                    img.height = int(row_height)

                    cell = 'F{}'.format(row_index + 2)
                    output_worksheet.add_image(img, cell)

            except requests.Timeout:
                print(f"The request timed out for article {article_number}")
                continue
            except requests.RequestException as e:
                print(f"An error occurred for article {article_number}: {e}")
                continue
            
            row_index += 1

        for i in range(1, sheet_df.shape[0] + 2):
            output_worksheet.row_dimensions[i].height = 59.5
            for col in ['E', 'F']:
                output_worksheet.column_dimensions[col].width = 18

        # Append column headers
        output_worksheet.append(sheet_df.columns.tolist())
        
        # Append data rows
        for row in sheet_df.itertuples(index=False, name=None):
            output_worksheet.append(row)

    output_workbook.remove(output_workbook['Sheet'])  

    # Save the workbook to a BytesIO object
    excel_buffer = BytesIO()
    output_workbook.save(excel_buffer)
    excel_buffer.seek(0)

    b64 = base64.b64encode(excel_buffer.read()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download Excel file</a>'
    return href
    
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
        xls = pd.ExcelFile(uploaded_file, engine='openpyxl')
        params_dict = {sheet_name: pd.read_excel(xls, sheet_name=sheet_name) for sheet_name in xls.sheet_names}
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
        st.session_state['outcome'] = outcome
        st.markdown(get_csv_download_link(st.session_state['outcome']), unsafe_allow_html=True)

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