from Env.DC_gym import DC_Gym
from Env.ClassDefinitions import Stream
import numpy as np

class DC_gym_reward(DC_Gym):
    """
    This version of the gym only has a single stream as the state. Discrete actions are just to seperate or not
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, action):
        continuous_actions, discrete_action = action
        real_continuous_actions = self.get_real_continuous_actions(continuous_actions)
        if discrete_action == self.discrete_action_space.n - 1:  # submit
            self.State.submit_stream()
            if self.State.n_streams == 0 or self.State.final_outlet_streams == self.max_outlet_streams:
                done = True
            else:
                done = False
            info = {}
            tops_state = "NA"
            bottoms_state = "NA"
            revenue = 0
            TAC = 0
            return tops_state, bottoms_state, revenue, TAC, done, info

        # put the selected stream (flows, temperature, pressure) as the input to a new column
        selected_stream = self.State.streams[0]
        self.set_inlet_stream(selected_stream.flows, selected_stream.temperature, selected_stream.pressure)

        n_stages = int(real_continuous_actions[0])
        reflux_ratio = round(real_continuous_actions[1], 2)
        reboil_ratio = round(real_continuous_actions[2], 2)
        pressure_drop_ratio = round(real_continuous_actions[3], 2)

        self.set_unit_inputs(n_stages, reflux_ratio, reboil_ratio, pressure_drop_ratio)
        sucessful_solve = self.solve()
        self.error_counter["total_solves"] += 1
        if sucessful_solve is False:  # This is currently just telling the
            self.failed_solves += 1
            self.error_counter["error_solves"] += 1
            if self.failed_solves >= 2: # reset if we fail twice
                done = True
            else:
                done = False
            info = {"failed solve": 1}
            state = self.State.state
            # basically like returning state as tops, 0 as bottoms because nothing has happened in the seperation
            tops = state.copy()
            bottoms = np.zeros(tops.shape)
            return tops, bottoms, 0, 0, done, info

        # TAC includes operating costs so we actually don't need these duties
        TAC, condenser_duty, reboiler_duty = self.get_outputs()

        tops_info, bottoms_info = self.get_outlet_info()
        tops_flow, tops_temperature, tops_pressure = tops_info
        bottoms_flow, bottoms_temperature, bottoms_pressure = bottoms_info
        tops_state, bottoms_state = self.State.update_state(Stream(self.State.n_streams, tops_flow, tops_temperature, tops_pressure),
                                        Stream(self.State.n_streams+1, bottoms_flow, bottoms_temperature, bottoms_pressure))
        annual_revenue = self.stream_value(tops_flow) + self.stream_value(bottoms_flow) - self.stream_value(selected_stream.flows)

        if self.State.n_streams == self.max_outlet_streams or self.State.final_outlet_streams == self.max_outlet_streams:
            # this just sets a cap on where the episode must end
            # either all the streams are outlets in which case n_streams is zero
            # (this only happens via submitted steam action so doen't come into play in this part of the step)
            # or we reach a cap on max number of streams
            done = True
        else:
            done = False
        info = {}
        self.import_file()  # current workaround is to reset the file after each solve
        return tops_state, bottoms_state, annual_revenue, TAC, done, info

