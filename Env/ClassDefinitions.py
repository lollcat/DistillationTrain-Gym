import numpy as np
from collections import deque

class Valve:
    def __init__(self, StreamNumber, Tout, Pout):
        self.StreamNumber = StreamNumber
        self.Outlet_Temperature = Tout
        self.Outlet_Pressure = Pout

class Stream:
    def __init__(self, number, flows, temperature, pressure):
        """
        :param number: stream number, starts from zero
        :param flows: stream flowrates in moles, numpy array
        :param temperature: stream temperature
        """
        assert type(flows) == np.ndarray
        self.number = number
        self.flows = np.round(flows, 6)
        self.temperature = temperature
        self.pressure = pressure

class Column:
    def __init__(self, in_number, tops_number, bottoms_number, n_stages, reflux_ratio, reboil_ratio, OperatingPressure, InletTemperature, TAC):
        self.inlet_number = in_number
        self.tops_number = tops_number
        self.bottoms_number = bottoms_number
        self.n_stages = n_stages
        self.reflux_ratio = reflux_ratio
        self.reboil_ratio = reboil_ratio
        self.OperatingPressure = OperatingPressure
        self.InletTemperature = InletTemperature
        self.TAC = TAC

class State:
    """
    Keeps track of the state as well as the flowsheet layout
    For now the state includes temperature and pressure as added straight to the end of the stream vector
    """
    def __init__(self, feed_stream, max_streams, simple=True):
        self.streams = deque([feed_stream]) # this is for the state
        self.max_streams = max_streams
        self.temp_norm = feed_stream.temperature
        self.pressure_norm = feed_stream.pressure
        self.flow_norm = feed_stream.flows.max()
        self.simple_state = simple
        if simple:
            self.state = np.zeros((1, len(feed_stream.flows) + 2))  # +2 is for T & P
        else:
            self.state = np.zeros((self.max_streams, len(feed_stream.flows) + 2))
        self.create_state()
        self.final_outlet_streams = []
        self.all_streams = [feed_stream]  # this is a record of all streams
        self.column_data = []  # contains tuple of (inlet stream no, tops stream no, bottoms stream no, (actions), TAC)

    def create_state(self):
        if self.simple_state:
            if self.n_streams > 0:
                self.state = np.array([list(self.streams[0].flows/self.flow_norm) +
                                                         [self.streams[0].temperature/self.temp_norm, self.streams[0].pressure/self.pressure_norm]
                                                     ], dtype="float32")
            else:
                self.state = np.zeros(self.state.shape, dtype="float32")
        else:
            self.state = np.zeros(self.state.shape)
            if self.n_streams > 0:
                self.state[0:self.n_streams] = np.array([list(stream.flows / self.flow_norm) +
                                       [stream.temperature / self.temp_norm,
                                        stream.pressure / self.pressure_norm]
                                       for stream in self.streams], dtype="float32")

    def update_streams(self, new_streams, is_product=(False, False)):
        self.streams.popleft()
        for i, stream in enumerate(new_streams):
            self.all_streams.append(stream)
            if is_product[i]:
                self.final_outlet_streams.append(stream)
            else:
                self.streams.append(stream)

    def update_state(self, new_streams, *kwargs):
        self.update_streams(new_streams, *kwargs)
        self.create_state()


    def get_next_state(self, tops, bottoms):
        assert self.simple_state is True
        tops_state = np.array([list(tops.flows / self.flow_norm) +
                         [tops.temperature / self.temp_norm, tops.pressure / self.pressure_norm]
                         ], dtype="float32")
        bottoms_state = np.array([list(bottoms.flows / self.flow_norm) +
                            [bottoms.temperature / self.temp_norm, bottoms.pressure / self.pressure_norm]
                            ], dtype="float32")
        return tops_state, bottoms_state  # for next state


    def submit_stream(self):
        # submit current stream as end product
        self.final_outlet_streams.append(self.streams[0])
        self.streams.popleft()
        self.create_state()
        if self.n_streams == 0:
            return True
        else:
            return False

    """
    def add_valve_data(self, inlet_StreamNumber, Tout, Pout):
        self.valves.append(Valve(inlet_StreamNumber, Tout, Pout))
    """

    def add_column_data(self, in_number, tops_number, bottoms_number, n_stages, reflux_ratio, reboil_ratio,
                        OperatingPressure, InletTemperature, TAC):
        self.column_data.append(Column(in_number, tops_number, bottoms_number, n_stages, reflux_ratio, reboil_ratio,
                                       OperatingPressure, InletTemperature, TAC))

    @property
    def n_streams(self):
        return len(self.streams)

    @property
    def n_outlet_streams(self):
        return len(self.final_outlet_streams)

    @property
    def n_total_streams(self):
        return len(self.all_streams)