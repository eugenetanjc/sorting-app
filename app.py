# Import libraries
import pandas as pd
import os
import io
import dash
import base64
from dash import dcc, html, Output, Input, State
import backend
import compile
import logger
from flask import Flask, request, send_file
from datetime import datetime
from office365.runtime.auth.authentication_context import AuthenticationContext
from office365.sharepoint.client_context import ClientContext

# Define global variables
image_filename = 'data/ck_logo.png'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

# Create dictionaries for input options
year = int(datetime.now().strftime("%Y")) 
week_options = [{'label': str(i), 'value': str(i)} for i in range(1, 53)]
year_options = [{'label': str(i), 'value': str(i)} for i in range(year-1, year+2)]

# Country options
country_options = [{'label': country, 'value': country} for country in ['INDIA', 'INDONESIA', 'JAPAN', 'SAUDI ARABIA', 'SINGAPORE', 'SOUTH KOREA', 'THAILAND', 'VIETNAM']]

# Define app layout
def serve_layout():
    layout = html.Div(
        id='app-body',
        style={
            'margin': '10px',
            'text-align': 'center'
        },
        children=[
            html.Div([
                html.Img(
                    src='data:image/png;base64,{}'.format(encoded_image.decode()),
                    style={'height': '40%',
                        'width': '40%'}
                )
            ]),        
            html.Br(),
            html.Div([html.H1('ECOM Sorting App')], 
                    style={'text-align': 'center',
                            'margin-bottom': '15px',
                            'display': 'inline-block',
                            'font-family': 'Bahnschrift',
                            'fontWeight':'bold'}),
            html.Div(
                id='tabs-area',
                children=[
                    dcc.Tabs(
                        id='tabs',
                        children=[
                            dcc.Tab(
                                id='authentication-tab',
                                label='Authentication',
                                style={'font-family': 'Helvetica'},
                                children=[
                                    html.Div([
                                        html.Br(),
                                        html.Label('Sharepoint username:', style={'fontWeight': 'bold'}),
                                        dcc.Input(
                                            id='username-input',
                                            type='text',
                                            placeholder='Enter your username',
                                            style={'width': '300px', 'margin-left': '10px'},
                                        ),
                                        html.Br(),
                                        html.Label('Sharepoint password:', style={'fontWeight': 'bold'}),
                                        dcc.Input(
                                            id='password-input',
                                            type='password',
                                            placeholder='Enter your password',
                                            style={'width': '300px', 'margin-left': '10px'},
                                        ),
                                        html.Br(),
                                        html.Button('Authenticate', 
                                                    id='authenticate-button', 
                                                    n_clicks=0, 
                                                    className='submit-button', 
                                                    style={'margin-bottom': '20px', 'margin-top': '10px'}),
                                        html.Br(),
                                        html.Div(id='authenticate-output'),
                                    ])
                                ]
                            ),

                            dcc.Tab(
                                id='sorting-tab',
                                label='Sorting',
                                style={'font-family': 'Helvetica'},
                                children=[
                                    html.Div([                     
                                        html.Br(),
                                        html.Label('Upload Parameters File:', style={'fontWeight': 'bold'}),
                                        dcc.Upload(
                                            id='upload-params',
                                            children=html.Div(['Drag and Drop or ', html.A('Select Parameters file')]),
                                            style={
                                                'width': '100%',
                                                'height': '60px',
                                                'lineHeight': '60px',
                                                'borderWidth': '1px',
                                                'borderStyle': 'dashed',
                                                'borderRadius': '5px',
                                                'textAlign': 'center',
                                                'margin-right': '10px',
                                                'margin-bottom': '10px'
                                            },
                                            multiple=False
                                        ),
                                        html.Div(id='upload-params-message', style={'height':'30px'}),             
                                        html.Label('Select your country:', style={'fontWeight':'bold', 'margin-top': '10px'}),
                                        html.Br(),
                                        dcc.Dropdown(
                                            id='country-dropdown',
                                            options=country_options,
                                            value='INDIA',
                                            placeholder='Select a country'
                                        ),
                                        html.Br(),
                                        html.Label('Select your year:', style={'fontWeight':'bold'}),
                                        html.Br(),
                                        dcc.Dropdown(
                                            id='year-dropdown',
                                            options=year_options,
                                            value='2024',
                                            placeholder='Select a year'
                                        ),
                                        html.Br(),
                                        html.Label('Select your week:', style={'fontWeight':'bold'}),
                                        html.Br(),
                                        dcc.Dropdown(
                                            id='week-dropdown',
                                            options=week_options,
                                            value='24',
                                            placeholder='Select a week'
                                        ),
                                        html.Br(),
                                        html.Label('Select your country type:', style={'fontWeight':'bold'}),
                                        html.Br(),
                                        dcc.Dropdown(
                                            id='seasonality-dropdown',
                                            options=[
                                                {'label': 'Hot', 'value': 'Hot'},
                                                {'label': 'Cold', 'value': 'Cold'},
                                                {'label': 'ANZ', 'value': 'ANZ'}
                                            ],
                                            value='Hot',
                                            placeholder='Select a country type'
                                        ),
                                        html.Br(),
                                        html.Button(
                                            'Run sorting now', 
                                            id='run-backend-button', 
                                            n_clicks=0, 
                                            className='submit-button', 
                                            style={'margin-bottom': '10px'}
                                        ),
                                        html.Div(id='output'),
                                        html.Div(id='download-link'),
                                        html.Div(id='outcome-sorting')
                                    ])
                                ]
                            ),
                            dcc.Tab(
                                id='compile-tab',
                                label='Compilation',
                                style={'font-family': 'Helvetica'},
                                children=[
                                    html.Div([
                                        html.Br(),
                                        html.Label('Select Country folder to save .csv file into: ', style={'fontWeight':'bold'}),
                                        html.Br(),
                                        dcc.Dropdown(
                                            id='country-input',
                                            options=country_options,
                                            value=None,
                                            placeholder='Select a country for compile input'
                                        ),                                    
                                        html.Br(),
                                        html.Label('Select country type for your .csv file extension ', style={'fontWeight':'bold'}),
                                        dcc.Dropdown(
                                            id='countrytype-input',
                                            options=[
                                                {'label': 'Hot', 'value': 'Hot'},
                                                {'label': 'Cold', 'value': 'Cold'},
                                                {'label': 'ANZ', 'value': 'ANZ'}
                                            ],
                                            value=None,
                                            placeholder='Select a country type for compile input'
                                        ),
                                        html.Br(),
                                        html.Label('Input your working file (should be a .xlsx file) into the area below for compile:', style={'fontWeight':'bold'}),                                    
                                        html.Br(),
                                        dcc.Upload(
                                            id='upload-data',
                                            children=html.Div([
                                                'Drag and Drop or ',
                                                html.A('Select .xlsx working file (it can take 10-15 seconds to load)')
                                            ]),
                                            style={
                                                'width': '100%',
                                                'height': '60px',
                                                'lineHeight': '60px',
                                                'borderWidth': '1px',
                                                'borderStyle': 'dashed',
                                                'borderRadius': '5px',
                                                'textAlign': 'center',
                                                'margin-right': '10px',
                                                'margin-bottom': '10px'
                                            },
                                            multiple=True
                                        ),
                                        html.Div(id='upload-message'),  
                                        html.Button('Compile working file', 
                                                    id='submit-button-compile', 
                                                    n_clicks=0, 
                                                    className='submit-button', 
                                                    style={'margin-bottom': '10px'}),
                                        html.Div(id='outcome-compile'),
                                    ])
                                ]
                            ),
                        ]
                    )
                ]
            ),
            html.Br(),
            html.Div(
                [
                    html.P('Property of Charles & Keith Pte Ltd | 2024')
                ],
                style={'color': 'LightGray',
                    'fontSize': 14,
                    'text-align': 'center',
                    'font-family': 'Bahnschrift'
                    }
            )
        ]
    )
    return layout

# Define Flask app instance
server = Flask(__name__)

# Define Dash app instance
app = dash.Dash(__name__, 
                server=server, 
                external_stylesheets=['/assets/simpstyles.css'], 
                prevent_initial_callbacks=True,
                routes_pathname_prefix='/sortingapp/')
app.config.suppress_callback_exceptions = True
app.layout = serve_layout()

# Define global variables
authenticated = True
user = None

# Printing folder contents for verification
def print_folder_contents(ctx, folder_url):
    try:
        folder = ctx.web.get_folder_by_server_relative_url(folder_url)
        sub_folders = folder.files
        ctx.load(sub_folders)
        ctx.execute_query()
        folder_names = [s_folder.properties["Name"] for s_folder in sub_folders]

        return folder_names
    
    except Exception as e:
        print('Problem printing out library contents: ', e)

# Authentication check
@app.callback(
    Output('authenticate-output', 'children'),
    [Input('authenticate-button', 'n_clicks')],
    [State('username-input', 'value')],
    [State('password-input', 'value')]
)

def authenticate(n_clicks, username, password):
    global authenticated, user
    
    if n_clicks > 0:
        url_shrpt = 'https://charleskeith.sharepoint.com/sites/CKEInventory'
        username_shrpt = 'it.dataanalysts@charleskeith.com'
        password_shrpt = 'ITd@t@analyst24!'

        try:
            # For authenticating into your sharepoint site
            ctx_auth = AuthenticationContext(url_shrpt)
            if ctx_auth.acquire_token_for_user(username_shrpt, password_shrpt):
                authenticated = True
                ctx = ClientContext(url_shrpt, ctx_auth)
                web = ctx.web
                ctx.load(web)
                ctx.execute_query()
                print('Authenticated into sharepoint as: ',web.properties['Title'])
            else:
                print(ctx_auth.get_last_error())

            # For printing out the contents of the shared documents folder
            folder_url_shrpt = '/sites/CKEInventory/Shared%20Documents/Merchandising/Pinning%20Guideline/'
            print(print_folder_contents(ctx,folder_url_shrpt)) 
            user = username.split('@')[0]
            return html.Div(
                [html.H6("Authenticated successfully!")],
                style={'color': 'green', 'font-family': 'Bahnschrift'}
            )
        
        # If authentication fails
        except Exception:
            return html.Div(
                [html.H6("Authentication failed, please try again.")],
                style={'color': 'red', 'font-family': 'Bahnschrift'}
            )
    return ''

# Sorting callback
@app.callback(
    Output('output', 'children'),
    Output('download-link', 'children'),
    Output('outcome-sorting', 'children'),
    [Input('run-backend-button', 'n_clicks')],
    [Input('upload-params', 'contents')],
    [State('username-input', 'value')],
    [State('password-input', 'value')],
    [State('country-dropdown', 'value')],
    [State('year-dropdown', 'value')],
    [State('week-dropdown', 'value')],
    [State('seasonality-dropdown', 'value')]
)

def peform_sorting(n_clicks, contents, s_username, s_password, s_country, s_year, s_week, s_ctype):
    global username, input_country, input_ctype, authenticated

    if n_clicks > 0:
        if not all([s_country, s_year, s_week, s_ctype]):
            error_message = "Please fill in all required fields."
            return None, None, html.Div(
                [
                    html.H6("An error occurred:"),
                    html.Pre(error_message),
                ],
                style={'color': 'red', 'font-family': 'Bahnschrift'}
            )
        if authenticated == False:
            error_message = "Please authenticate your sharepoint credentials first."
            return None, None, html.Div(
                [
                    html.H6("An error occurred:"),
                    html.Pre(error_message),
                ],
                style={'color': 'red', 'font-family': 'Bahnschrift'}
            )

        # Check user input
        username = "".join([str(x) for x in s_username])
        name = username.split('@')[0]
        print(f'Selected Country: {s_country}, Year: {s_year}, Week: {s_week}, Seasonality: {s_ctype}, User: {name}')

        # Read parameters file into pandas df
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        xls = pd.ExcelFile(io.BytesIO(decoded), engine='openpyxl')
        params_dict = {sheet_name: pd.read_excel(xls, sheet_name=sheet_name) for sheet_name in xls.sheet_names}

        # Run sorting function
        sorted_output_path, sorted_all_path = backend.sorting(s_username, s_password, s_country, s_year, s_week, s_ctype, name, params_dict)
        download_link = html.A('Download sorted data', href=f'/download_excel?path={sorted_all_path}')
        success_message = html.Div(
                            [html.P("Sorting performed successfully!")],
                            style={'color': 'green', 'font-family': 'Bahnschrift'}
                            )

        return sorted_output_path, download_link, success_message
                
    return '', '', ''

# Download sorted file
@app.server.route('/download_excel')
def download_excel():
    sorted_all_path = request.args.get('path')
    return send_file(sorted_all_path, as_attachment=True)

# Upload params file
@app.callback(Output('upload-params-message', 'children'),
            [Input('upload-params', 'filename')])

def update_upload_message(filename):
    if filename is not None:
        filename = filename[0] if isinstance(filename, list) else filename
        return html.Div([html.H6(f'{filename} uploaded.')],
                        style = {'color': 'green', 
                                 'font-family': 'Bahnschrift', 
                                 'margin-bottom': '10px'})
    return ''

# Upload file for compile
@app.callback(Output('upload-message', 'children'),
            [Input('upload-data', 'filename')])

def update_upload_message(filename):
    if filename is not None:
        filename = filename[0] if isinstance(filename, list) else filename
        return html.Div([html.H6(f'{filename} uploaded.')],
                        style = {'color': 'green', 
                                 'font-family': 'Bahnschrift', 
                                 'margin-bottom': '10px'})
    return ''


# Compile callback
@app.callback(Output('outcome-compile', 'children'),
              [Input('submit-button-compile', 'n_clicks')],
              [Input('upload-data', 'filename')],
              [State('country-input', 'value'),
               State('countrytype-input', 'value')])

def compile_file(n_clicks, filename, country, ctype):
    if n_clicks > 0:
        if authenticated == False:
            error_message = "Please authenticate your sharepoint credentials first."
            return html.Div(
                [
                    html.H6("An error occurred:"),
                    html.Pre(error_message),
                ],
                style={'color': 'red', 'font-family': 'Bahnschrift'}
            )
        if not filename:
            error_message = "Please upload a file."
            return html.Div(
                [
                    html.H6("An error occurred:"),
                    html.Pre(error_message),
                ],
                style={'color': 'red', 'font-family': 'Bahnschrift'}
            )

        if not all([country, ctype]):
            error_message = "Please fill in all required fields."
            return html.Div(
                [
                    html.H6("An error occurred:"),
                    html.Pre(error_message),
                ],
                style={'color': 'red', 'font-family': 'Bahnschrift'}
            )
        
        try:
            folder_path = country_folder_path
            filename = filename[0] if isinstance(filename, list) else filename
            file_path = os.path.join(folder_path, country, "Output", filename)

            # if file_path.endswith('.xlsx'):
            #     df = pd.read_excel(file_path, engine='openpyxl')
            # elif file_path.endswith('.csv'):
            #     df = pd.read_csv(file_path)
            
        except Exception as e:
            return html.Div([
                html.H6('There was an error processing the file.')],
                style = {'color': 'red', 'font-family': 'Bahnschrift'})
        
        compiled_file_path = compile.compiling(file_path, country, ctype, user)
        shared_folder_path = r"//192.168.1.250/CKI Inventory/52. Data Science/SortingPinning/Countries"
        compiled_file_path_suffix = compiled_file_path.split('Countries')[1]
        final_compiled_path = shared_folder_path + compiled_file_path_suffix

        return html.Div([html.P('CSV file compiled successfully!'),
                         html.P(f'File saved at {final_compiled_path}')],
                        style={'color': 'green', 'font-family': 'Bahnschrift'})
    
    return ''
