import numpy as np
from itertools import permutations

from Env.ClassDefinitions import Stream, State
from gym import spaces

from Env.DC_class import SimulatorDC

class DC_Gym(SimulatorDC):
    """
    NB - units of flowsheet must be configured!!!
    This version of the gym only has a single stream as the state. Discrete actions are just to seperate or not
    this is currently just inherited by DC_gym_reward
    """
    def __init__(self, document_path, sales_prices,
                 annual_operating_hours=8000, required_purity=0.95, simple_state=True, auto_submit=True):
        super().__init__(document_path)
        self.simple_state = simple_state
        self.auto_submit = auto_submit
        self.sales_prices = sales_prices
        self.required_purity = required_purity
        self.annual_operating_seconds = annual_operating_hours*3600

        feed_conditions = self.get_inlet_stream_conditions()
        self.original_feed = Stream(1, feed_conditions["flows"],
                                    feed_conditions["temperature"],
                                    feed_conditions["pressure"]
                                    )
        self.n_components = len(self.original_feed.flows)
        # now am pretty flexible in number of max streams, to prevent simulation going for long set maximum to 10
        self.max_outlet_streams = self.n_components*2
        self.stream_table = [self.original_feed]

        self.State = State(self.original_feed, self.max_outlet_streams, simple=simple_state)
        self.min_total_flow = self.State.flow_norm/20  # definately aren't interested in streams with 1/20th of the flow

        if simple_state:
            # Now configure action space
            self.discrete_action_names = ['seperate_yes_or_no']
            self.discrete_action_space = spaces.Discrete(2)
        else:
            self.discrete_action_names = ['stream selection']
            self.discrete_action_space = spaces.Discrete(self.max_outlet_streams + 1)

        # number of stages will currently be rounded off
        # pressure drop is as a fraction of the current pressure
        self.continuous_action_names = ['number of stages', 'reflux ratio', 'reboil ratio', 'pressure drop ratio']
        # these will get converted to numbers between -1 and 1
        self.real_continuous_action_space = spaces.Box(low=np.array([5, 0.1, 0.1, 0]), high=np.array([100, 10, 10, 0.9]),
                                                       shape=(4,))
        self.continuous_action_space = spaces.Box(low=-1, high=1, shape=(4,))
        # define gym space objects
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.State.state.shape)
        self.failed_solves = 0
        self.error_counter = {"total_solves": 0,
                              "error_solves": 0}  # to get a general idea of how many solves are going wrong
        self.current_step = 0

    def step(self, action):
        continuous_actions, discrete_action = action
        if discrete_action == self.discrete_action_space.n - 1:  # submit
            self.current_step += 1
            done = self.State.submit_stream()  # if this results in 0 outlet streams then done
            info = {}
            revenue = 0
            TAC = 0
            next_state = self.State.state
            if self.simple_state:
                return ("NA", "NA"), revenue, TAC, done, info
            else:
                return next_state, revenue, TAC, done, info

        self.import_file()  # current workaround is to reset the file before each solve

        selected_stream = self.State.streams[discrete_action]
        assert selected_stream.flows.max() > self.min_total_flow

        # put the selected stream (flows, temperature, pressure) as the input to a new column
        real_continuous_actions = self.get_real_continuous_actions(continuous_actions)
        self.set_inlet_stream(selected_stream.flows, selected_stream.temperature, selected_stream.pressure)

        n_stages = int(real_continuous_actions[0])
        reflux_ratio = round(real_continuous_actions[1], 2)
        reboil_ratio = round(real_continuous_actions[2], 2)
        pressure_drop_ratio = round(real_continuous_actions[3], 2)

        self.set_unit_inputs(n_stages, reflux_ratio, reboil_ratio, pressure_drop_ratio)
        sucessful_solve = self.solve()
        self.error_counter["total_solves"] += 1
        if sucessful_solve is False:
            self.failed_solves += 1
            self.error_counter["error_solves"] += 1
            TAC = 0
            revenue = 0
            if self.failed_solves >= 3: # reset if we fail 3 times
                done = True
            else:
                done = False
            info = {"failed solve": 1}
            next_state = self.State.state
            if self.simple_state:
                # basically like returning state as tops, 0 as bottoms because nothing has happened in the seperation
                tops = next_state.copy()
                bottoms = np.zeros(tops.shape)
                return (tops, bottoms), revenue, TAC, done, info
            else:
                return next_state, revenue, TAC, done, info

        self.current_step += 1  # if there is a sucessful solve then step the counter
        # TAC includes operating costs so we actually don't need these duties
        TAC, condenser_duty, reboiler_duty = self.get_outputs()

        tops_info, bottoms_info = self.get_outlet_info()
        tops_flow, tops_temperature, tops_pressure = tops_info
        bottoms_flow, bottoms_temperature, bottoms_pressure = bottoms_info
        tops = Stream(self.State.n_total_streams + 1, tops_flow, tops_temperature, tops_pressure)
        bottoms = Stream(self.State.n_total_streams + 2, bottoms_flow, bottoms_temperature, bottoms_pressure)


        if self.auto_submit is True:
            is_product = [False, False]
            tops_revenue = self.stream_value(tops_flow)
            bottoms_revenue = self.stream_value(bottoms_flow)
            if tops_revenue > 0:
                is_product[0] = True
            if bottoms_revenue > 0:
                is_product[1] = True
            annual_revenue = tops_revenue + bottoms_revenue
            self.State.update_state([tops, bottoms], is_product)
            info = is_product
        else:
            annual_revenue = self.stream_value(tops_flow) + self.stream_value(bottoms_flow) - self.stream_value(selected_stream.flows)
            self.State.update_state([tops, bottoms])
            info = {}

        if self.simple_state is True:
            next_state = self.State.get_next_state(tops, bottoms)
        else:
            next_state = self.State.state


        mass_balance_rel_error = np.absolute(
            (selected_stream.flows-(tops.flows+bottoms.flows)) / np.maximum(selected_stream.flows, 0.01)) # max to prevent divide by 0
        if mass_balance_rel_error.max() >= 0.05:
            print("MB error!!!")
        #assert mass_balance_rel_error.max() < 0.05, f"Max error: {mass_balance_rel_error.max()}"

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

        return next_state, annual_revenue/1e6, -TAC/1e6, done, info  # convert rewards to million $

    def get_real_continuous_actions(self, continuous_actions):
        # interpolation
        real_continuous_actions = self.real_continuous_action_space.low + \
                             (continuous_actions - self.continuous_action_space.low)/\
                             (self.continuous_action_space.high - self.continuous_action_space.low) *\
                             (self.real_continuous_action_space.high - self.real_continuous_action_space.low)
        return real_continuous_actions

    @property
    def legal_discrete_actions(self):
        """
        Illegal actions:
         - Choose Null Stream in stream table
        """
        legal_actions = [i for i in range(0, self.State.n_streams)]
        if self.State.n_streams > 3: # for now only let submission after at least 2 columns
            legal_actions.append(self.discrete_action_space.n - 1)
        return legal_actions

    def sample(self):
        discrete_action = self.discrete_action_space.sample()
        continuous_action = self.continuous_action_space.sample()
        return continuous_action, discrete_action

    def reset(self):
        self.reset_flowsheet()
        self.stream_table = [self.original_feed]
        self.State = State(self.original_feed, self.max_outlet_streams)
        self.column_streams = []
        self.failed_solves = 0
        self.current_step = 0
        return self.State.state.copy()

    def reward_calculator(self, inlet_flow, tops_flow, bottoms_flow, TAC):
        annual_revenue = self.stream_value(tops_flow) + self.stream_value(bottoms_flow) - self.stream_value(inlet_flow)
        reward = annual_revenue - TAC  # this represents the direct change annual profit caused by the additional column

        return reward

    def stream_value(self, stream_flow):
        if max(stream_flow / sum(stream_flow)) >= self.required_purity:
            revenue_per_annum = max(stream_flow) * self.sales_prices[np.argmax(stream_flow)] * self.annual_operating_seconds
            return revenue_per_annum
        else:
            return 0