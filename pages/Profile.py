import dash
import dash_html_components as html
import base64
from PIL import Image
from dash import callback
from dash.dependencies import Input, Output
from dash import callback, ctx, State
from dash import register_page

# Register
dash.register_page(__name__, path='/profile')

#Using Pillow to read the the image
pil_img = Image.open("asset/My_profile.png")

layout = html.Div(id='over2')

@callback(Output('over2', 'children'),
              [Input('prof', 'n_clicks')])

def prof1 (n_clicks):
    if n_clicks is not None:
        return html.Div([html.Img(src=pil_img,style={"width":"1500px","height":"659px" })
        ])

# layout = html.Div([
#     #html.Img(src='data:image/png;base64,{}'.format(encoded_image)),
#     html.Img(src=pil_img,style={"width":"1500px","height":"659px" }) #
# ])


