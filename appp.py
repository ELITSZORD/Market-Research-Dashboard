# Import Library
import dash
from dash import Dash
from dash import dcc
from dash import html, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash import callback, ctx, State
from dash import register_page, page_container

import PIL.Image as Image
import plotly.express as px 


import librosa
import librosa.display


import matplotlib
import matplotlib.axes
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os

import io
import base64


# ----------------------------------------------------------SECTION 1 - APP----------------------------------------------------------------------
# 1.1. Dash App
app = Dash(__name__, use_pages = True,external_stylesheets=[dbc.themes.LUX])
app.config['suppress_callback_exceptions']=True

## 1.2. Dashboard Title
#app.title = 'Market Research Dashboard'

## 1.3. Style
style0 = {'width':'1500px'}

hstyle2 = {"background": "#50596E", "color": "white", "font-size": "30px",'width': '100%', 'text-align':'center'}
hstyle3 = {"background": "#55936D", "color": "blue", "font-size": "20px"}

hstyle4 = {"background": "#339288", "color": "white", "font-size": "19px",'width': '20%'}

hstyle5 = {"color": "black", "font-size": "14px",'width': '22%', 'text-align':'center'}
hstyle6 = {"color": "black", "font-size": "20px",'width': '20%', 'text-align':'center'}


## 1.4. Dashboard Core Components, Python Code, others. 
### Logo
linkedin = "https://static-00.iconduck.com/assets.00/linkedin-icon-2048x2048-ya5g47j2.png"
rstudio = "https://www.pngall.com/wp-content/uploads/2017/05/Copyright-Symbol-R-Free-Download-PNG.png"
github = "https://www.svgrepo.com/show/361181/github.svg"

### 1.4.1. Navbar Core Components
navbar = dbc.NavbarSimple(    
    children=[
        #dbc.NavItem(dbc.NavLink("Home", id='home1',href=dash.page_registry['pages.Home']['path'])),                           
        dbc.DropdownMenu(
            [
                dbc.DropdownMenuItem(page["name"], href=page["path"])
                for page in dash.page_registry.values()
                if page["module"] != "pages.Overview"
                and page["module"] != "pages.not_found_404"
                if page["module"] != "pages.Profile"
                and page["module"] != "pages.not_found_404"

            ],
            nav=True,
            in_navbar=True,
            label="Analysis"
        ), 
        dbc.NavItem(dbc.NavLink("Overview Results", id='overview',href=dash.page_registry['pages.Overview']['path'])),        
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("Future Development", header=True),                
                dbc.DropdownMenuItem("Real Time New Feature"),
                dbc.DropdownMenuItem("Text Summarization"),
                dbc.DropdownMenuItem("Video Emotion Classification")],
        nav=True,
        in_navbar=True
        ,
        label="More"),
        dbc.NavItem(dbc.NavLink("Profile", id='prof',href=dash.page_registry['pages.Profile']['path']))
        ,
        html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=linkedin, height="45px")),
                    ],
                    align="center",
                    className="g-0",
                ),
                href="https://www.linkedin.com/in/faisal-adhisthana-nugraha-111503125/",
                style={"textDecoration": "none"},
            ),
        html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=rstudio, height="45px")),
                    ],
                    align="center",
                    className="g-0",
                ),
                href="https://rpubs.com/ELITSZORD",
                style={"textDecoration": "none"},
            ),
        html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=github, height="45px")),
                    ],
                    align="center",
                    className="g-0",
                ),
                href="https://github.com/ELITSZORD",
                style={"textDecoration": "none"},
            )
    ],
brand="Market Research Dashboard", 
brand_href="#",
color="#3A4856",
dark=True,style=style0
)


# ----------------------------------------------------------SECTION 2 - UI-----------------------------------------------------------------------
# 2. User Interface
app.layout = dbc.Container(
    [
        navbar, dash.page_container
        # ,
        # dbc.Row([html.Div(id='over1')]) 
              
    ], fluid=True
)



# ----------------------------------------------------------SECTION 3.1 - Analysis Callback-----------------------------------------------------------------



# ----------------------------------------------------------SECTION 4 Run App-----------------------------------------------------------------
# 4. Run App at local
if __name__ == '__main__':
    app.run_server(debug=True)
    
    
    
    
    # #### ----ROW1----
    #     dbc.Row([
    #         #navbar, dash.page_container
    #     ]),html.Br(),
        
    # #### ----ROW2_HomePage----
    #     dbc.Row([
    #         dbc.Col([],width=2),
    #         dbc.Col([
                
    #         ],width=10),
    #         dbc.Col([],width=2)    
    #     ])