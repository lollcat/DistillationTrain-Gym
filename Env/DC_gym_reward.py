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
        self.import_file()  # current workaround is to reset the file before each solve
        continuous_actions, discrete_action = action
        real_continuous_actions = self.get_real_continuous_actions(continuous_actions)
        if discrete_action == 1:  # submit
            self.current_step += 1
            done = self.State.submit_stream()  # if this results in 0 outlet streams then done
            info = {}
            tops_state = "NA"
            bottoms_state = "NA"
            revenue = 0
            TAC = 0
            return tops_state, bottoms_state, revenue, TAC, done, info

        # put the selected stream (flows, temperature, pressure) as the input to a new column
        selected_stream = self.State.streams[0]
        assert selected_stream.flows.max() > self.min_total_flow
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

        self.current_step += 1  # if there is a sucessful solve then step the counter
        # TAC includes operating costs so we actually don't need these duties
        TAC, condenser_duty, reboiler_duty = self.get_outputs()

        tops_info, bottoms_info = self.get_outlet_info()
        tops_flow, tops_temperature, tops_pressure = tops_info
        bottoms_flow, bottoms_temperature, bottoms_pressure = bottoms_info
        tops = Stream(self.State.n_total_streams() + 1, tops_flow, tops_temperature, tops_pressure)
        bottoms = Stream(self.State.n_total_streams() + 2, bottoms_flow, bottoms_temperature, bottoms_pressure)
        tops_state, bottoms_state = self.State.update_state(tops, bottoms)

        annual_revenue = self.stream_value(tops_flow) + self.stream_value(bottoms_flow) - self.stream_value(selected_stream.flows)
        mass_balance_rel_error = np.absolute(((selected_stream.flows/self.State.flow_norm - (
                tops_state[:, 0:self.n_components] +
                bottoms_state[:, 0:self.n_components]))
                                  / np.maximum(selected_stream.flows, 0.01)/self.State.flow_norm)) # max to prevent divide by 0
        if mass_balance_rel_error.max() >= 0.05:
            print("should catch in breakpoint")
        assert mass_balance_rel_error.max() < 0.05, f"Max error: {mass_balance_rel_error.max()}"

        if self.State.n_streams + self.State.n_outlet_streams >= self.max_outlet_streams:
            # this just sets a cap on where the episode must end
            # either all the streams are outlets in which case n_streams is zero
            # (this only happens via submitted steam action so doen't come into play in this part of the step)
            # or we reach a cap on max number of streams
            done = True
        else:
            done = False
        self.State.add_column_data(selected_stream.number, tops.number, bottoms.number,
                               (n_stages, reflux_ratio, reboil_ratio, tops_pressure), TAC)
        info = {}
        return tops_state, bottoms_state, annual_revenue, -TAC, done, info

