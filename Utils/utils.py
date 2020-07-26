import matplotlib.pyplot as plt
import numpy as np
import time
from IPython.display import Image
import pydot
import imageio



class Plotter:
    def __init__(self, score_history, episodes, config_string, metric=0, freeze_point=False, show_heuristics=True):
        self.config = config_string
        self.show_heuristics = show_heuristics
        self.score_history = score_history
        self.episodes = episodes + 1
        self.metric = metric
        self.freeze_point = freeze_point
        if episodes < 100:
            raise ValueError("Not enough episodes")
        if metric is 0:
            self.by_random = 0.49282674
            self.by_lightness = 9.57710821
            self.by_flowrate = 9.29304865
            self.by_volatility = 9.57810267
        else:
            self.by_random = 0.8182666909582774
            self.by_lightness = 10.4331691
            self.by_flowrate = 10.12371856
            self.by_volatility = 10.43425249

    def running_mean(self, x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / N

    def plot(self, save=False):
        freeze_point = self.freeze_point
        episodes = np.arange(self.episodes)
        smoothed_rews = self.running_mean(self.score_history, 100)
        plt.plot(episodes[-len(smoothed_rews):], smoothed_rews, color="blue")
        plt.plot(episodes, self.score_history, color='grey', alpha=0.3)
        if self.show_heuristics is True:
            plt.plot([episodes[0], episodes[-1]], [self.by_flowrate, self.by_flowrate], alpha=0.3)
            plt.plot([episodes[0], episodes[-1]], [self.by_volatility, self.by_volatility], alpha=0.6)
            plt.plot([episodes[0], episodes[-1]], [self.by_lightness, self.by_lightness], alpha=0.6)
            plt.plot([episodes[0], episodes[-1]], [self.by_random, self.by_random], alpha=0.6)
        if freeze_point is not False or 0:
            plt.plot([freeze_point, freeze_point], [min(self.score_history), max(self.score_history)], "--", color="black")
        #plt.yscale("symlog")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        if self.show_heuristics is True:
            if freeze_point is False:
                plt.legend(["Average Reward", "Reward", "Flowrate Heuristic", "Volatility heuristic",
                                "Boiling Point Heuristic", "Random Average"])
            else:
                plt.legend(["Average Reward", "Reward", "Flowrate Heuristic", "Volatility Heuristic",
                            "Boiling Point Heuristic", "Random Average", "Freeze Point"])
        else:
            if freeze_point is False:
                plt.legend(["Average Reward", "Reward"])
            else:
                plt.legend(["Average Reward", "Reward", "Freeze Point"])

        if save is True:
            if freeze_point is True:
                plt.savefig(f"Data_Plots/{self.config}_learn_" + str(time.time()) + ".png", bbox_inches='tight')
            else:
                plt.savefig(f"Data_Plots/{self.config}_learn_" + str(time.time()) + ".png", bbox_inches='tight')
        plt.show()

class Visualiser:
    def __init__(self, env):
        self.env = env

    def visualise(self, show_all=True):
        space = " "
        env = self.env
        G = pydot.Dot(graph_type="digraph", rankdir="LR")
        outlet_nodes = []
        nodes = []
        edges = []
        image_list = []
        if show_all is True:
            feed_string = "".join([f"{env.compound_names[i]} {round(env.feed[i])} kmol/h\n"
                                            for i in range(env.feed.shape[0])])
        else:
            feed_string = " "
        feed_node = pydot.Node(feed_string, shape="square", color="white")
        G.add_node(feed_node)
        for i in range(len(env.sep_order)):
            LK = env.sep_order[i]
            split = round(env.split_order[i][0]*100, 1)
            n_trays = int(env.column_dimensions[i][0])
            capital_cost = round(env.capital_cost[i][0]/1e6, 1)
            label = f'Column {i + 1} \nLK {LK} \nsplit {split}% \nntrays {n_trays} \ncapital cost R{capital_cost} M'
            nodes.append(pydot.Node(label, shape="square"))
            G.add_node(nodes[i])
            if i > 0:
                stream_in = env.column_streams[i][0]
                column_link, loc = self.find_column(stream_in)
                edges.append(pydot.Edge(nodes[column_link], nodes[i], label=int(stream_in+1), headport="w",
                                        tailport=loc))
                G.add_edge(edges[i - 1])
            else:
                G.add_edge(pydot.Edge(feed_node, nodes[0], label=1, headport="w", tailport="e"))

            # add outlet streams
            tops, bottoms = env.column_streams[i][1:]
            if tops in env.state_streams:
                stream = env.stream_table[tops]
                flowrate = int(stream.sum()+0.5)
                purity = round((100 * stream.max() / stream.sum()), 1)
                compound = stream.argmax()
                compound = env.compound_names[compound]
                revenue = round(stream.max() * env.product_prices[np.argmax(stream)] * 8000/1e6, 1)  # TODO fix this silly workaround
                if show_all is False:
                    label = f"{flowrate} kmol/h \n{purity}% {compound}"
                else:
                    if purity >= 96 and stream.max() > 1:
                        product_classification = f"Product{i*space}\n Revenue R{revenue} M \n Purity {purity}% \n"
                    else:
                        product_classification = f"Non-Product{i*space}\n"
                    label = product_classification + "".join(
                        [f"{env.compound_names[i]} {round(stream[i] + 0.05, 1)} kmol/h\n"
                         for i in range(env.feed.shape[0])])  # if stream[i] > 0.05])

                outlet_nodes.append(
                    pydot.Node(label, shape="box", color="white"))
                G.add_node(outlet_nodes[-1])
                G.add_edge(pydot.Edge(nodes[i], outlet_nodes[-1], label=int(tops+1), headport="w", tailport="ne"))

            if bottoms in env.state_streams:
                stream = env.stream_table[bottoms]
                flowrate = int(stream.sum()+0.5)
                purity = round((100 * stream.max() / stream.sum()), 1)
                compound = stream.argmax()
                compound = env.compound_names[compound]
                revenue = round(stream.max() * env.product_prices[np.argmax(stream)] * 8000/1e6, 1)  # TODO fix this silly workaround
                if show_all is False:
                    label = f"{flowrate} kmol/h \n{purity}% {compound}"
                else:
                    if purity >= 96 and stream.max() > 1:
                        product_classification = f"Product{i*space} \n Revenue R{revenue} M \n Purity {purity}% \n"
                    else:
                        product_classification = f"Non-Product{i*space} \n"
                    label = product_classification + "".join(
                        [f"{env.compound_names[i]} {round(stream[i] + 0.05, 1)} kmol/h\n"
                         for i in range(env.feed.shape[0])])  # if stream[i] > 0.05])

                outlet_nodes.append(pydot.Node(label, shape="box", color="white"))
                G.add_node(outlet_nodes[-1])
                G.add_edge(pydot.Edge(nodes[i], outlet_nodes[-1], label=int(bottoms+1), headport="w", tailport="se"))
        BFD = imageio.imread(G.create_png())
        return BFD


    def find_column(self, stream):
        env = self.env
        for i in range(len(env.column_streams)):
            if stream in env.column_streams[i]:
                if env.column_streams[i][1] == stream:
                    loc = "ne"
                    return i, loc
                elif env.column_streams[i][2] == stream:
                    loc = "se"
                    return i, loc
                else:
                    print("error")
