import numpy as np
from collections import deque


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
    def __init__(self, in_number, tops_number, bottoms_number, actions, TAC):
        self.inlet_number = in_number
        self.tops_number = tops_number
        self.bottoms_number = bottoms_number
        self.n_stages = actions[0]
        self.reflux_ratio = actions[1]
        self.reboil_ratio = actions[2]
        self.inlet_pressure = actions[3]
        self.TAC = TAC

class State:
    """
    Keeps track of the state as well as the flowsheet layout
    For now the state includes temperature and pressure as added straight to the end of the stream vector
    """
    def __init__(self, feed_stream, max_streams):
        self.streams = deque([feed_stream]) # this is for the state
        self.max_streams = max_streams
        self.temp_norm = feed_stream.temperature
        self.pressure_norm = feed_stream.pressure
        self.flow_norm = feed_stream.flows.max()
        self.state = np.zeros((1, len(feed_stream.flows) + 2))  # +2 is for T & P
        self.create_state()
        self.final_outlet_streams = []
        self.all_streams = [feed_stream]  # this is a record of all streams
        self.column_data = []  # contains tuple of (inlet stream no, tops stream no, bottoms stream no, (actions), TAC)

    def create_state(self):
        self.state = np.array([list(self.streams[0].flows/self.flow_norm) +
                                                 [self.streams[0].temperature/self.temp_norm, self.streams[0].pressure/self.pressure_norm]
                                                 ])

    def update_streams(self, tops, bottoms):
        """
        :param selected_stream_position: the selected stream's position in the state
        :param tops top stream (Stream Class object)
        :param bottoms bottoms stream (Stream Class object)
        """
        self.streams.popleft()
        self.streams.append(tops)
        self.streams.append(bottoms)
        self.all_streams.append(tops)
        self.all_streams.append(bottoms)

    def update_state(self, tops, bottoms):
        self.update_streams(tops, bottoms)
        self.create_state()
        tops_state = np.array([list(tops.flows / self.flow_norm) +
                         [tops.temperature / self.temp_norm, tops.pressure / self.pressure_norm]
                         ])
        bottoms_state = np.array([list(bottoms.flows / self.flow_norm) +
                            [bottoms.temperature / self.temp_norm, bottoms.pressure / self.pressure_norm]
                            ])
        return tops_state, bottoms_state  # for next state

    def submit_stream(self):
        self.final_outlet_streams.append(self.streams[0])
        self.streams.popleft()
        if self.n_streams == 0:
            return True
        else:
            self.create_state()
            return False

    def add_column_data(self, in_number, tops_number, bottoms_number, actions, TAC):
        self.column_data.append(Column(in_number, tops_number, bottoms_number, actions, TAC))

    @property
    def n_streams(self):
        return len(self.streams)

    @property
    def n_outlet_streams(self):
        return len(self.final_outlet_streams)

    def n_total_streams(self):
        return len(self.all_streams)