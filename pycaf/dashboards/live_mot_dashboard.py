import dash
from dash.dependencies import Output, Input
from dash import dcc
from dash import html
import plotly
import random
import plotly.graph_objs as go
from collections import deque

from pycaf import LiveMOT


config_path = "C:\\ControlPrograms\\pycaf\\config_bec.json"
interval = 0.1


X = deque(maxlen=20)
X.append(1)

Y = deque(maxlen=20)
Y.append(1)

live_mot = LiveMOT(config_path=config_path, interval=interval)


app = dash.Dash(__name__)
app.layout = html.Div(
    [
        dcc.Graph(id='live-graph', animate=True),
        dcc.Graph(id='live-graph-2', animate=True),
        dcc.Interval(
            id='graph-update',
            interval=1000,
            n_intervals=0
        ),
    ]
)


@app.callback(
    [
        Output('live-graph', 'figure'),
        Output('live-graph-2', 'figure')
    ],
    [
        Input('graph-update', 'n_intervals')
    ]
)
def update_graph_scatter(n):
    """live_mot(
        "MOTBasicMultiTrigger",
        "MOTCoilsOnValue",
        0.4,
        0.0
    )"""
    X.append(X[-1]+1)
    Y.append(Y[-1]+Y[-1] * random.uniform(-0.1, 0.1))
    data = plotly.graph_objs.Scatter(
            x=list(X),
            y=list(Y),
            name='Scatter',
            mode='lines+markers'
    )

    output_1 = {
        'data': [data],
        'layout': go.Layout(
            xaxis=dict(range=[min(X), max(X)]),
            yaxis=dict(range=[min(Y), max(Y)])
        )
    }

    output_2 = {
        'data': [data],
        'layout': go.Layout(
            xaxis=dict(range=[min(X), max(X)]),
            yaxis=dict(range=[min(Y), max(Y)])
        )
    }

    return output_1, output_2


if __name__ == '__main__':
    app.run_server()
