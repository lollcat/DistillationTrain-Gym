import numpy as np
from itertools import permutations

from Env.ClassDefinitions import Stream, State
from gym import spaces

from Env.DC_class import SimulatorDC

class DC_Gym(SimulatorDC):
    """
    This version of the gym only has a single stream as the state. Discrete actions are just to seperate or not
    this is currently just inherited by DC_gym_reward
    """
    def __init__(self, document_path, sales_prices,
                 annual_operating_hours=8000, required_purity=0.95):
        super().__init__(document_path)
        self.sales_prices = sales_prices
        self.required_purity = required_purity
        self.annual_operating_hours = annual_operating_hours

        feed_conditions = self.get_inlet_stream_conditions()
        self.original_feed = Stream(0, feed_conditions["flows"],
                                    feed_conditions["temperature"],
                                    feed_conditions["pressure"]
                                    )
        self.n_components = len(self.original_feed.flows)
        # now am pretty flexible in number of max streams, to prevent simulation going for long set maximum to 10
        self.max_outlet_streams = self.n_components*2
        self.stream_table = [self.original_feed]

        self.State = State(self.original_feed, self.max_outlet_streams)

        # contains a tuple of 3 (in, tops, bottoms) stream numbers describing the connections of streams & columns
        self.column_streams = []

        # Now configure action space
        self.discrete_action_names = ['seperate_yes_or_no']
        self.discrete_action_space = spaces.Discrete(2)
        # number of stages will currently be rounded off
        # pressure drop is as a fraction of the current pressure
        self.continuous_action_names = ['number of stages', 'reflux ratio', 'reboil ratio', 'pressure drop ratio']
        # these will get converted to numbers between -1 and 1
        self.real_continuous_action_space = spaces.Box(low=np.array([5, 0.2, 0.2, 0]), high=np.array([50, 5, 5, 0.9]),
                                                       shape=(4,))
        self.continuous_action_space = spaces.Box(low=-1, high=1, shape=(4,))
        # define gym space objects
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.State.state.shape)

        # have to limit additional memories so they are in proportion to the number that accor
        # number of possibilities when there is one stream limits the number of memories
        self.max_new_memories = self.max_outlet_streams * (self.max_outlet_streams - 1)
        self.failed_solves = 0
        self.error_counter = {"total_solves": 0,
                              "error_solves": 0}  # to get a general idea of how many solves are going wrong


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
        return self.State.state.copy()

    def reward_calculator(self, inlet_flow, tops_flow, bottoms_flow, TAC):
        annual_revenue = self.stream_value(tops_flow) + self.stream_value(bottoms_flow) - self.stream_value(inlet_flow)
        reward = annual_revenue - TAC  # this represents the direct change annual profit caused by the additional column

        return reward

    def stream_value(self, stream_flow):
        if max(stream_flow / sum(stream_flow)) >= self.required_purity:
            revenue_per_annum = max(stream_flow) * self.sales_prices[np.argmax(stream_flow)] * self.annual_operating_hours
            return revenue_per_annum
        else:
            return 0

    def augment_data(self, experience, shuffle_next_states=True):
        """
        Can shuffle both the state and next state. Some adjustments made to number of shuffles selected to ensure same
        number of memories returned each time
        """
        state, action_continuous, action_discrete, reward, next_state, one_minus_done = experience
        permutation_list = []
        n_streams = np.sum(state.any(1))  # number of streams that aren't null
        state_perms = list(permutations(list(np.arange(self.max_outlet_streams)), int(n_streams)))
        next_state_perm_n = self.max_outlet_streams - n_streams
        for perm_index in np.random.choice(len(state_perms), size=int(round(self.max_new_memories/next_state_perm_n))):
            perm = state_perms[perm_index]
            state_perm = np.zeros_like(state)
            state_perm[perm, :] = state[0:n_streams]
            next_state_perm = state_perm
            if action_discrete == self.discrete_action_space.n - 1:
                action_discrete_perm = action_discrete
            else:
                action_discrete_perm = perm[action_discrete]
            # TODO this will be wasteful for sumbit == true action, fix this
            for index in np.random.permutation(np.where(~state_perm.any(1)))[0]: # locations for new streams are where the null streams are
                next_state_perm[index, :] = state[n_streams, :]
                permutation_list.append((state_perm, action_continuous, action_discrete_perm, reward,
                                         next_state_perm, one_minus_done))
                if len(permutation_list) == self.max_new_memories:
                    return permutation_list
        raise Exception("For some reason the loop didnt reach enough permutations to get to n= max_new_memories")
