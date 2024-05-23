from typing import Dict, Any, List
from nicegui import ui, app
import json
import os
from local_file_picker import local_file_picker
from lakeshore import Model336

from pycaf.modules import MaxiGauge, PressureReading
from pycaf import Experiment, Scope


ui.colors(accent='#6AD4DD')

config: Dict[str, Any] = None
config_path: str = None
script_path: str = None
experient: Experiment = None
scope: Scope = None
model336: Model336 = None
maxigauge: MaxiGauge = None
columns = [
    {
        'name': 'sensor',
        'label': 'SensorID',
        'field': 'sensor',
        'required': True,
        'align': 'left'
    },
    {
        'name': 'reading',
        'label': 'Reading',
        'field': 'reading',
        'required': True,
        'align': 'left'
    },
    {
        'name': 'unit',
        'label': 'Unit',
        'field': 'unit',
        'required': True,
        'align': 'left'
    },
]


@ui.page('/')
def index():
    global config, config_path, experient, scope, \
        maxigauge, model336, script_path
    ui.page_title('CaFBEC')
    if "config_path" in app.storage.user:
        config_path = app.storage.user['config_path']
        config = app.storage.user['config']
        ui.notify('Loading config from last session')
        model336 = Model336()
        ui.notify('Model336 instantiated successfully')
        maxigauge = MaxiGauge(config_path)
        ui.notify('MaxiGauge instantiated successfully')
        experient = Experiment(config_path, 0.1)
        ui.notify('Experiment instantiated successfully')

    with ui.page_sticky(x_offset=18, y_offset=18):

        async def choose_config_file() -> None:
            global config, config_path, experient, scope, maxigauge, model336
            _config_path = await local_file_picker('C:\\', multiple=False)
            if _config_path is not None:
                config_path = _config_path[0]
                with open(config_path, "r") as f:
                    config = json.load(f)
                app.storage.user['config'] = config
                app.storage.user['config_path'] = config_path
                ui.notify(
                    f"You have chosen {config_path} " +
                    "to configure the experiment"
                )
                model336 = Model336()
                ui.notify('Model336 instantiated successfully')
                maxigauge = MaxiGauge(config_path)
                ui.notify('MaxiGauge instantiated successfully')
                experient = Experiment(config_path, 0.1)
                ui.notify('Experiment instantiated successfully')
            else:
                ui.notify('No path chosen, halting instantiation process')

        with ui.button(
            icon='settings',
            on_click=choose_config_file
        ).props('fab color=accent'):
            ui.tooltip(
                'Select the configuration .json file'
            ).classes('bg-green')

    with ui.tabs().classes('w-full') as tabs:
        tab_hardware_systems = ui.tab('Asynchronus Operations')
        tab_expt_and_analysis = ui.tab('Real Time Operations')
        tab_history = ui.tab('History')

    with ui.tab_panels(tabs, value=tab_expt_and_analysis).classes('w-full'):
        with ui.tab_panel(tab_hardware_systems):
            with ui.button_group().props("dense"):
                with ui.dropdown_button(
                    'Heater',
                    auto_close=True,
                    color="secondary"
                ).props("dense"):
                    ui.item(
                        'ON',
                        on_click=lambda: ui.notify('You clicked item 1')
                    ).props("dense")
                    ui.item(
                        'OFF',
                        on_click=lambda: ui.notify('You clicked item 2')
                    ).props("dense")
                with ui.dropdown_button(
                    'Cryocooler',
                    auto_close=True,
                    color="secondary"
                ).props("dense"):
                    ui.item(
                        'ON',
                        on_click=lambda: ui.notify('You clicked item 1')
                    ).props("dense")
                    ui.item(
                        'OFF',
                        on_click=lambda: ui.notify('You clicked item 2')
                    ).props("dense")
                with ui.dropdown_button(
                    'Cycle Source',
                    auto_close=True,
                    color="secondary"
                ).props("dense"):
                    ui.item(
                        'Start',
                        on_click=lambda: ui.notify('You clicked item 1')
                    ).props("dense")
                    ui.item(
                        'Stop',
                        on_click=lambda: ui.notify('You clicked item 2')
                    ).props("dense")

        with ui.tab_panel(tab_expt_and_analysis):

            with ui.row().classes('w-full'):
                expt_container = ui.card().style("width: 20vw")

                def handle_click_add_scan_parameter():
                    with expt_container:
                        col_container = ui.column().classes('w-full')
                        with col_container:
                            ui.input(
                                label='Parameter Name',
                                placeholder='check spelling properly',
                                on_change=lambda: ui.notify('good choice')
                            ).classes('w-full').props("dense")
                            with ui.row().classes('w-full'):
                                ui.number(
                                    label='start',
                                    value=0,
                                    format='%.2f',
                                    on_change=handle_set_scan_start
                                ).style("width: 30%").props("dense")
                                ui.number(
                                    label='stop',
                                    value=0,
                                    format='%.2f',
                                    on_change=handle_set_scan_stop
                                ).style("width: 30%").props("dense")
                                ui.number(
                                    label='increment',
                                    value=0,
                                    format='%.2f',
                                    on_change=handle_set_scan_increment
                                ).style("width: 30%").props("dense")
                            with ui.row().classes('w-full'):
                                ui.toggle(
                                    ["Int", "Double"],
                                    value="Int"
                                ).props("dense")
                                ui.space()
                                with ui.button_group().props("dense"):
                                    ui.button(
                                        icon='remove_circle_outline',
                                        on_click=lambda: col_container.delete(),
                                        color="secondary"
                                    ).props("dense")
                                    ui.button(
                                        icon='add_circle',
                                        on_click=handle_click_add_scan_parameter,
                                        color="secondary"
                                    ).props("dense")

                def handle_click_start_scan():
                    ui.notify('Experiment started')

                def handle_click_stop_scan():
                    ui.notify('Experiment stopped')

                def handle_set_scan_start():
                    ui.notify('Scan start set')

                def handle_set_scan_stop():
                    ui.notify('Scan stop set')

                def handle_set_scan_increment():
                    ui.notify('Scan increment set')

                async def choose_script_file() -> None:
                    global script_path
                    _script_path = await local_file_picker('~', multiple=False)
                    if _script_path is not None:
                        script_path = os.path.basename(_script_path[0])
                        script_path_label.set_text(
                            f"Current selected script: {script_path}"
                        )
                        ui.notify(f'You chose {script_path}')

                with expt_container:
                    with ui.column().classes('w-full'):
                        with ui.row().classes('w-full'):
                            with ui.button_group().props("dense"):
                                with ui.button(
                                    icon="code",
                                    on_click=choose_script_file,
                                    color="secondary"
                                ).props("dense"):
                                    ui.tooltip(
                                        'select the .cs file ' +
                                        'containing the patterns'
                                    ).classes('bg-green')
                                with ui.button(
                                    icon="send",
                                    on_click=handle_click_start_scan,
                                    color="secondary"
                                ).props("dense"):
                                    ui.tooltip(
                                        'Start running the experiments'
                                    ).classes('bg-green')
                                with ui.button(
                                    icon="stop_circle",
                                    on_click=handle_click_stop_scan,
                                    color="secondary"
                                ).props("dense"):
                                    ui.tooltip(
                                        'Stop the experiments immediately'
                                    ).classes('bg-green')
                            ui.space()
                            with ui.knob(
                                value=20,
                                min=10,
                                max=100,
                                step=5,
                                show_value=True,
                                color="secondary",
                                track_color='grey-2',
                                on_change=lambda: ui.notify('You changed')
                            ).props("dense"):
                                ui.tooltip(
                                    'number of iterations'
                                ).classes('bg-green')
                            with ui.circular_progress(
                                0,
                                min=0,
                                max=100,
                                show_value=True,
                                color="green"
                            ).props("dense"):
                                ui.tooltip(
                                    'current progress'
                                ).classes('bg-green')
                        script_path_label = ui.label()
                        ui.input(
                            label='Describe the experiment',
                            placeholder='provide as much detail as possible'
                        ).classes('w-full').props("dense")
                        handle_click_add_scan_parameter()

                analysis_container = ui.card().style("width: 55vw")

                with analysis_container:
                    with ui.column().classes('w-full'):
                        ui.label('Card content')

                hardware_table = ui.table(
                    columns=columns,
                    rows=[],
                    row_key='sensor'
                ).props('dense').style("width: 18vw")

                def read_hardware_parameters():
                    rows = []
                    if config is not None:
                        ts_desc: Dict[str, str] = \
                            config["plugin_modules"]["lakeshore"]["description"]
                        ps_desc: Dict[str, str] = \
                            config["plugin_modules"]["pfeiffer"]["description"]
                        if model336 is not None:
                            ts: List[str] = model336.get_all_kelvin_reading()
                            for key, val in ts_desc.items():
                                rows.append({
                                    "sensor": val,
                                    "reading": ts[int(key)],
                                    "unit": "K"
                                })
                        if maxigauge is not None:
                            ps: List[PressureReading] = maxigauge.pressures()
                            for key, val in ps_desc.items():
                                rows.append({
                                    "sensor": val,
                                    "reading": f"{ps[int(key)-1].pressure:.2e}",
                                    "unit": "mBar"
                                })
                    hardware_table.update_rows(rows=rows)

                ui.timer(10, read_hardware_parameters, active=True)
                

        with ui.tab_panel(tab_history):
            ui.label('History')
            ui.date(
                value='2023-01-01',
                on_change=lambda e: result.set_text(e.value)
            )
            result = ui.label()


ui.run(
    port=9000,
    storage_secret='private key to secure the browser session cookie'
)
