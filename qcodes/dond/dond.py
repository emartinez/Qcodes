'''Multi-dimensional scans.'''


import time
import numpy as np
from progressbar import progressbar
from qdev_wrappers.dataset.doNd import (_process_params_meas,
                                        _catch_keyboard_interrupts,
                                        _register_parameters,
                                        _set_write_period,
                                        _register_actions)
from qcodes.dataset.measurements import Measurement
from qcodes.dataset.plotting import plot_by_id
from qcodes.instrument.base import _BaseParameter, ParameterWithSetpoints


def _make_slow_setpoints(params, setpoint_args):
    all_setpoint_values = []
    for param in params:
        args = setpoint_args[param]
        start = args[0]
        stop = args[1]
        num_points = args[2]
        setpoint_values = np.linspace(start, stop, num_points)
        all_setpoint_values.append(setpoint_values)
    setpoint_grids = np.meshgrid(*all_setpoint_values)
    flat_setpoint_grids = [np.ravel(grid, order='F') for grid in setpoint_grids]
    if flat_setpoint_grids:
        flat_setpoints = np.vstack(flat_setpoint_grids).T
    else:
        flat_setpoints = [tuple()]
    return flat_setpoints

def _parse_dond_args(*args):
    params = []
    setpoints = []
    meas_callables = []
    setpoint_args = {}
    for arg in args:
        if isinstance(arg, _BaseParameter) or callable(arg):
            params.append(arg)
            setpoint_args[arg] = []
            last_param = arg
        else: # A numerical argument:
            try:
                float(arg)
                setpoint_args[last_param].append(arg)
            except TypeError:
                raise TypeError('Invalid argument type: {}'.format(arg))
    for param in params:
        if setpoint_args[param]:
            setpoints.append(param)
        else:
            meas_callables.append(param)
    return setpoints, setpoint_args, meas_callables

def _do_nested_loops(setpoint_params, setpoint_args, meas_params,
                     enter_actions=None, register_actions=None,
                     write_period=1., do_plot=False):
    meas = Measurement()
    _register_parameters(meas, setpoint_params)
    _register_parameters(meas, meas_params, setpoints=setpoint_params)
    _set_write_period(meas, write_period)
    slow_setpoints = _make_slow_setpoints(setpoint_params, setpoint_args)
    _register_actions(meas, enter_actions, exit_actions)

    last_setpoints = dict([(param, None) for param in setpoint_params])
    with _catch_keyboard_interrupts() as _, meas.run() as datasaver:
        for setpoints in progressbar(slow_setpoints):
            result = []
            for setpoint_param, setpoint in zip(setpoint_params, setpoints):
                if not setpoint == last_setpoints[setpoint_param]:
                    delay = setpoint_args[setpoint_param][3]
                    setpoint_param(setpoint)
                    time.sleep(delay)
                    last_setpoints[setpoint_param] = setpoint
                result.append((setpoint_param, setpoint))
            result += _process_params_meas(meas_params)
            datasaver.add_result(*result)
        qc_dataset = datasaver._dataset

    if do_plot:
        plot_by_id(qc_dataset.run_id)

    return qc_dataset

def _configure_setpoint(setpoint, args):
    start = args[0]
    stop = args[1]
    num_points = args[2]
    delay = args[3]
    values = np.linspace(start, stop, num_points)
    setpoint.set_hw_ramp(values, delay)

def _add_fast_setpoints(fast_setpoints, param_args, meas_callables):
    for param in meas_callables:
        if isinstance(param, ParameterWithSetpoints):
            param.setpoints = fast_setpoints

def dond(*args, fast_setpoints=None):
    '''Scan N parameters, looping from slowest (first) to fastest (last).

    Argument syntax is:
    dond(setpoint_param_1, start_1, stop_1, num_points_1, delay_1, ...,
         setpoint_param_n, start_n, stop_n, num_points_1, delay_n,
         meas_param_1, meas_param_2, ..., meas_param_m).'''

    if fast_setpoints is None:
        fast_setpoints = []
    setpoint_params, setpoint_args, meas_callables = _parse_dond_args(*args)
    slow_setpoints = [p for p in setpoint_params if p not in fast_setpoints]
    fast_setpoints_ordered = [p for p in setpoint_params if p in fast_setpoints]
    fast_setpoints = fast_setpoints_ordered
    for setpoint in fast_setpoints:
        _configure_setpoint(setpoint, setpoint_args[setpoint])
    _add_fast_setpoints(fast_setpoints, setpoint_args, meas_callables)
    dataset = _do_nested_loops(slow_setpoints, setpoint_args, meas_callables)
    return dataset
