from typing import Dict
from nicegui import ui, app
from datetime import datetime
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from lakeshore import Model336

from pycaf.modules import MaxiGauge


config_path: str = "C:\\ControlPrograms\\pycaf\\config_bec.json"
ts_all = np.zeros((1, 8), dtype=float)
ps_all = np.zeros((1, 6), dtype=float)
times = np.array([], dtype=float)
legend_counter = 0


@ui.page('/')
def index():
    with open(config_path, "r") as f:
        app.storage.user['config'] = json.load(f)
    lakeshore_config = \
        app.storage.user['config']["plugin_modules"]["lakeshore"]
    pfeiffer_config = \
        app.storage.user['config']["plugin_modules"]["pfeiffer"]
    model336 = Model336()
    maxigauge = MaxiGauge(config_path)
    ts_descr: Dict[str, str] = lakeshore_config["description"]
    ps_descr: Dict[str, str] = pfeiffer_config["description"]

    with ui.tabs().classes('w-full') as tabs:
        tab_state = ui.tab('State')
        tab_expt = ui.tab('Experiment')
        tab_config = ui.tab('Configuration')
    with ui.tab_panels(tabs, value=tab_state).classes('w-full'):
        with ui.tab_panel(tab_state):
            ui.label('State of the experiment')
            fig = go.Figure()
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02
                )
            fig.update_layout(
                height=600, width=800,
                yaxis_type="log"
            )
            fig.update_xaxes(title_text="Time", row=1, col=1)
            fig.update_yaxes(title_text="Pressure [mbar]", row=1, col=1)
            fig.update_yaxes(title_text="Temperature [K]", row=2, col=1)
            plot = ui.plotly(fig).classes('w-full')

            def update_line_plot() -> None:
                global times, ts_all, ps_all, legend_counter
                ts = np.array(model336.get_all_kelvin_reading()).reshape(
                    (1, 8)
                )
                _ps = maxigauge.pressures()
                ps = np.array([ips.pressure for ips in _ps]).reshape(
                    (1, 6)
                )
                if legend_counter == 0:
                    times = np.array([datetime.now()])
                    ts_all = ts
                    ps_all = ps
                else:
                    times = np.append(times, datetime.now())
                    ts_all = np.append(ts_all, ts, axis=0)
                    ps_all = np.append(ps_all, ps, axis=0)
                if legend_counter >= 10000:
                    times = times[1:]
                    ts_all = ts_all[1:, :]
                    ps_all = ps_all[1:, :]
                plot.clear()
                for i in range(8):
                    fig.add_trace(
                        go.Scatter(
                            x=times,
                            y=ts_all[:, i],
                            mode="lines+markers",
                            showlegend=True if legend_counter == 0 else False,
                            name=f"{ts_descr[f'{i}']}"
                        ),
                        row=2,
                        col=1
                    )
                for i in range(6):
                    fig.add_trace(
                        go.Scatter(
                            x=times,
                            y=ps_all[:, i],
                            mode="lines+markers",
                            showlegend=True if legend_counter == 0 else False,
                            name=f"{ps_descr[f'{i+1}']}"
                        ),
                        row=1,
                        col=1
                    )
                legend_counter += 1
                plot.update()

            line_updates = ui.timer(10, update_line_plot, active=True)
            ui.checkbox('display').bind_value(line_updates, 'active')

        with ui.tab_panel(tab_expt):
            ui.label('Current experiments')
        with ui.tab_panel(tab_config):
            ui.label('Loaded JSON configuration')
            ui.json_editor(
                {'content': {'json': app.storage.user['config']}},
                on_select=lambda e: ui.notify(f'Select: {e}'),
                on_change=lambda e: ui.notify(f'Change: {e}')
            )


ui.run(
    port=9000,
    storage_secret='private key to secure the browser session cookie'
)
