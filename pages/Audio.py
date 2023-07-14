# Library
import dash
from dash import Dash
from dash import dcc
from dash import html, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash import callback, ctx, State
from dash import register_page
import pickle

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

# Register
dash.register_page(__name__, path='/Audio')

# Style
hstyle = {"background": "#2D847A", "color": "white", "font-size": "30px",'width': '100%'}
hstyle1 = {"background": "#96B1AC", "color": "white", "font-size": "20px",'width': '100%'}

# Dash Components
## Upload
upl = dbc.Row(                    
                dcc.Upload(
                    id='upload-audio',
                    children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                    ]),
                    style={
                            'width': '99%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                    },
                multiple=False                        
                )                    
                )

# Layout
layout = dbc.Col([
    html.Div([
        html.H1("Upload The Audio WAV File", style=hstyle),
        html.Br(),
        html.Div([upl]), #Upload File
        html.Br(),
        html.Button(id='submit-button-state', n_clicks=0, children='Predict'),
        html.Div(id='output-state',style={'text-align':'center'}),
        html.Br(),
        html.Div(id='output'), # Plots
        
    ])
])


#------------------------------------------------------------------------------------------------------------------------------------
# Create 3 feature extraction methods.
def extract_feature(X, sr, mfcc, chroma, mel, pitch_mean, pitch_std, energy):
        result = np.array([])
        if chroma:
            stft = np.abs(librosa.stft(X))
            result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sr).T, axis=0)
            result = np.hstack((result,mel))
        if pitch_mean:
            pitches, magnitudes = librosa.piptrack(y=X, sr=sr)
            result = np.hstack((result, np.mean(pitches)))
        if pitch_std:
            pitches, magnitudes = librosa.piptrack(y=X, sr=sr)
            result = np.hstack((result, np.std(pitches)))
        if energy:
            S, phase = librosa.magphase(librosa.stft(X))
            rms = np.mean(librosa.feature.rms(S=S).T,axis=0)
            result = np.hstack((result,rms))
        return result

# State (untuk menyimpan nilai) = id -> upload-audio
# Input -> buat dash component html.Button(id='submit-button-state', n_clicks=0, children='Submit')
# Output, buat pada layout section -> html.Div(id='output-state')
@callback(Output('output-state', 'children'),
          Input('submit-button-state', 'n_clicks'),
          State('upload-audio', 'contents'))
    
def ml(n_clicks,contents): #,mfcc,chroma,mel,pitch_mean,pitch_std,energy
        x_test = []
        if contents is not None:
            audio_string = contents.split(',')[1]
            audio_bytes = base64.b64decode(audio_string)
            file_pathx = 'uploaded_audio.wav'

            with open(file_pathx, 'wb') as f:
                f.write(audio_bytes)
                
            #elif "submit-button-state" == ctx.triggered_id:
            X, sr = librosa.load(file_pathx, sr=None)
            feature = extract_feature(X, sr, mfcc=True, chroma=False, mel=False, pitch_mean=True, pitch_std=True, energy=True)
            x_test.append(feature)
            # Convert to NumPy arrays
            x_test = np.array(x_test)
            
            x_test = pd.DataFrame(list(x_test))
            
            # Load Model XGBoost with 80:20
            xgb820 = pickle.load(open('model/xgb820.pkl', 'rb'))
            
            # Predict
            y_pred = xgb820.predict(x_test)[0]
            
            if y_pred == 0:
                output = 'Anger'
            elif y_pred == 1:
                output = 'Happiness'
            elif y_pred == 2:
                output = 'Neutral'
            else:
                output = 'Sadness'
             
            # Return data
            return html.H1(f'The predicted Emotion is {output}.')   #y_pred # html.Div([])
        else:
            return html.Div() 
    
    
    
#------------------------------------------------------------------------------------------------------------------------------------    


#Callback

## Audio Analysis Output1 - WAVEFORM

@callback(Output('output', 'children'),
              [Input('upload-audio', 'contents')])

                
        
def update_output(contents):
    if contents is not None:
        audio_string = contents.split(',')[1]
        audio_bytes = base64.b64decode(audio_string)
        file_path = 'uploaded_audio.wav'

        with open(file_path, 'wb') as f:
            f.write(audio_bytes)
            
        y, sr = librosa.load(file_path, sr=None)
        # Wave
        fig, ax = plt.subplots(figsize=(11, 4))
        librosa.display.waveshow(y, sr=sr, alpha=0.4, ax=ax)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_yticks(np.arange(-1, 1.25, 0.5))
        ax.set_title("Waveform")
        a1 = plt.tight_layout()
        
        buf1 = io.BytesIO(a1)
        plt.savefig(buf1, format='png')
        buf1.seek(0)
        buffer_image1 = base64.b64encode(buf1.getvalue()).decode('utf-8')    
        
        # Energy
        ## Display RMS Energy
        S, phase = librosa.magphase(librosa.stft(y))
        rms = librosa.feature.rms(S=S)

        fig, ax = plt.subplots(figsize=(14, 5), nrows=2, sharex=True)
        times = librosa.times_like(rms)
        ax[0].semilogy(times, rms[0], label='RMS Energy')
        ax[0].set(xticks=[])
        ax[0].legend()
        ax[0].label_outer()
        librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                                y_axis='log', x_axis='time', ax=ax[1])
        ax[1].set_title("log Power spectrogram of")
        a2 = plt.tight_layout()
        
        buf2 = io.BytesIO(a2)
        plt.savefig(buf2, format='png')
        buf2.seek(0)
        buffer_image2 = base64.b64encode(buf2.getvalue()).decode('utf-8')
        
        # Pitch
        S1 = np.abs(librosa.stft(y, n_fft=4096))**2
        chroma = librosa.feature.chroma_stft(S=S1, sr=sr)
        
        ## display pitch
        fig, ax = plt.subplots(figsize=(14, 6),nrows=2, sharex=True)
        img3 = librosa.display.specshow(librosa.amplitude_to_db(S1, ref=np.max),
                                    y_axis='log', x_axis='time', ax=ax[0])
        #fig.colorbar(img3, ax=[ax[0]], location = 'right')
        ax[0].label_outer()
        img3 = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax[1])
        #fig.colorbar(img3, ax=[ax[1]], location = 'right') #ax.colorbar() 

        ax[1].set_xlabel("Time (s)")
        ax[0].set_ylabel("Hz")
        ax[1].set_ylabel("Pitch Class")
        ax[0].set_title("STFT Amplitude to dB")  
        ax[1].set_title("Pitch")                     
        a3 = plt.tight_layout()
        
        buf3 = io.BytesIO(a3)
        plt.savefig(buf3, format='png')
        buf3.seek(0)
        buffer_image3 = base64.b64encode(buf3.getvalue()).decode('utf-8')
        
        # MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr)
    
        # Display the MFCCs with the file name as the plot title
        fig, ax5 = plt.subplots(figsize=(16, 3))
        img5 = librosa.display.specshow(mfccs, sr=sr, x_axis='time',ax=ax5)
        fig.colorbar(img5, ax=ax5)
        ax5.set_title("MFCC") 
        ax5.set_ylabel("MFCC Coefficient")
        
        a4 = plt.tight_layout()
            
        buf = io.BytesIO(a4)
        plt.savefig(buf, format='png')
        buf.seek(0)
        buffer_image4 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return html.Div([
            html.H3('Uploaded WAV file:',style=hstyle1),
            dbc.Row(html.Div([
                html.Audio(src=contents, controls=True),
                html.Img(src='data:image/png;base64,' + buffer_image1),
            ])),
            dbc.Row(html.Div([
                html.Br(),
                html.H3("Feature Extraction",className="fw-bold fst-italic",style=hstyle1),
                html.Br(),
                html.H4('Audio Energy:'),
                html.Br(),
                html.Img(src='data:image/png;base64,' + buffer_image2),
                html.Br(),
                dbc.Row([dbc.Col(
                    dbc.Button(
                    "Definition",
                    id="left",
                    className="mb-1",
                    color="primary",
                    n_clicks=0,size='sm'
                    )
                    ),
                    dbc.Col(
                    dbc.Button(
                    "Notes",
                    color="primary",
                    id="right",
                    className="mb-1",
                    n_clicks=0,size='sm'
                    )
                    )
                ]),
                
                dbc.Row(
                [
                    dbc.Col(
                        dbc.Collapse(
                            dbc.Card(dbc.CardBody([
                                html.P(
                                "• The first graph shows that the root mean square (RMS) energy of the audio signal is a measure of the average power of the signal."
                                "It can be used to track the overall loudness of the signal over time."
                                ,
                                className="card-text",style={'font-size':13}),
                                html.P("• The second graph shows that the log power spectrogram which is a visualization of the power of the signal as a function of frequency and time. It can be used to see how the power of the signal varies over time and frequency.",
                                className="card-text",style={'font-size':13})
                            ])),
                            id="left-collapse",
                            is_open=False,
                        )
                    ),
                    dbc.Col(
                        dbc.Collapse(
                            dbc.Card(dbc.CardBody([
                                html.P(
                                "• The number of peaks in the RMS Energy plot can indicate the type of emotion being expressed, the volume of the speaker's voice, and the background noise."
                                "However, determining an emotion requires considering multiple parameters, where energy is one of the supporting factors."
                                ,
                                className="card-text",style={'font-size':13}),
                                html.P("• In a spectrogram energy plot, the high energy / loudness of audio represented by the whiter color of the spectrogram at specific time and frequency."
                                ,
                                className="card-text",style={'font-size':13})
                            ])),
                            id="right-collapse",
                            is_open=False,
                        )
                    ),
                ],
                className="mt-2",
                )                
            ])),
            dbc.Row(html.Div([
                html.Br(),
                html.H4('Audio Pitch:'),
                html.Img(src='data:image3/png;base64,' + buffer_image3),
                html.Br(),
                dbc.Row([dbc.Col(
                    dbc.Button(
                    "Definition",
                    id="left3",
                    className="mb-3",
                    color="primary",
                    n_clicks=0,size='sm'
                    )
                    ),
                    dbc.Col(
                    dbc.Button(
                    "Notes",
                    color="primary",
                    id="right3",
                    className="mb-3",
                    n_clicks=0,size='sm'
                    )
                    )
                ]),
                
                dbc.Row(
                [
                    dbc.Col(
                        dbc.Collapse(
                            dbc.Card(dbc.CardBody([
                                html.P(
                                "• The STFT process divides the audio signal into overlapping short segments, also applies the Fourier Transform to each segment and resulting in a frequency representation for each time segment."
                                "Then the amplitude of audio signal is converted to dB."
                                ,
                                className="card-text",style={'font-size':13}),
                                html.P("• Pitch can provide information about the pitch of the sound level which can help machine learning in determining the emotion of an audio file.",
                                className="card-text",style={'font-size':13})
                            ])),
                            id="left-collapse4",
                            is_open=False,
                        )
                    ),
                    dbc.Col(
                        dbc.Collapse(
                            dbc.Card(dbc.CardBody([
                                html.P(
                                "• The pitch level can see over time and at a specific frequency where that information can be beneficial to the model in determining the emotion type."
                                ,
                                className="card-text",style={'font-size':13})
                            ])),
                            id="right-collapse4",
                            is_open=False,
                        )
                    ),
                ],
                className="mt-4",
                )
            ])),
            dbc.Row(html.Div([
                html.Br(),
                html.H4('MFCC:'),
                html.Img(src='data:image3/png;base64,' + buffer_image4),
                html.Br(),
                dbc.Row([dbc.Col(
                    dbc.Button(
                    "Definition",
                    id="left5",
                    className="mb-5",
                    color="primary",
                    n_clicks=0,size='sm'
                    )
                    ),
                    dbc.Col(
                    dbc.Button(
                    "Notes",
                    color="primary",
                    id="right5",
                    className="mb-5",
                    n_clicks=0,size='sm'
                    )
                    )
                ]),
                
                dbc.Row(
                [
                    dbc.Col(
                        dbc.Collapse(
                            dbc.Card(dbc.CardBody([
                                html.P(
                                "• MFCC is a method used to extract features from audio signals and transform the representation of audio signals from the time domain to the frequency domain, taking into account human hearing characteristics." 
                                "These coefficients provide an overview of the frequency energy distribution in the signal and reflect the human auditory system's ability to process and distinguish sounds."
                                ,
                                className="card-text",style={'font-size':13})
                            ])),
                            id="left-collapse6",
                            is_open=False,
                        )
                    ),
                    dbc.Col(
                        dbc.Collapse(
                            dbc.Card(dbc.CardBody([
                                html.P(
                                "• The brighter the plot, then the highest coefficient at a specific time and frequency."
                                "Therefore, to observe its patterns/characteristics, you can compare the MFCC plots between different emotions."
                                ,
                                className="card-text",style={'font-size':13})
                            ])),
                            id="right-collapse6",
                            is_open=False,
                        )
                    ),
                ],
                className="mt-6",
                ) 
            ]))     
        ])        
    else:
            return html.Div() 


# Callback Energy Description   

@callback(
    Output("left-collapse", "is_open"),
    [Input("left", "n_clicks")]
    , State("left-collapse", "is_open"),
)
def toggle_left(n_left, is_open):
    if n_left:
        return not is_open
    return is_open

@callback(
    Output("right-collapse", "is_open"),
    [Input("right", "n_clicks")]
    , [State("right-collapse", "is_open")],
)
def toggle_right(n_right, is_open):
    if n_right:
        return not is_open
    return is_open


# Callback Pitch Description   

@callback(
    Output("left-collapse4", "is_open"),
    [Input("left3", "n_clicks")]
    , State("left-collapse4", "is_open"),
)
def toggle_left(n_left, is_open):
    if n_left:
        return not is_open
    return is_open

@callback(
    Output("right-collapse4", "is_open"),
    [Input("right3", "n_clicks")]
    , [State("right-collapse4", "is_open")],
)
def toggle_right(n_right, is_open):
    if n_right:
        return not is_open
    return is_open


# Callback MFCC Description   

@callback(
    Output("left-collapse6", "is_open"),
    [Input("left5", "n_clicks")]
    , State("left-collapse6", "is_open"),
)
def toggle_left(n_left, is_open):
    if n_left:
        return not is_open
    return is_open

@callback(
    Output("right-collapse6", "is_open"),
    [Input("right5", "n_clicks")]
    , [State("right-collapse6", "is_open")],
)
def toggle_right(n_right, is_open):
    if n_right:
        return not is_open
    return is_open