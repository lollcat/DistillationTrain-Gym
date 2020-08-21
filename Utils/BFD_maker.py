import pydot
from IPython.display import Image, display

class Visualiser:
    def __init__(self, env):
        self.env = env

    def visualise(self, show_all=True):
        space = " "
        G = pydot.Dot(graph_type="digraph", rankdir="LR")
        outlet_nodes = []
        nodes = []
        edges = []
        image_list = []
        if show_all is True:
            feed_string = "feed \n" + "".join([self.env.compound_names[i] + " " + str(round(flow, 2)) + " mol/s \n"
                                               for i, flow in enumerate(self.env.original_feed.flows)])
        else:
            feed_string = " "
        feed_node = pydot.Node(feed_string, shape="square", color="white")
        G.add_node(feed_node)
        for i, column_info in enumerate(self.env.State.column_data):
            label = f'Column {i + 1} \nn_stages {column_info.n_stages} \nRR ' +  str(round(column_info.reflux_ratio, 1)) + ' \n' + \
                    f"BR " + str(round(column_info.reboil_ratio, 1)) + f"\nTAC $ " + str(round(column_info.TAC/1e6, 2))+ " M"
            nodes.append(pydot.Node(label, shape="square"))
            G.add_node(nodes[i])
            if i > 0:
                stream_in = column_info.inlet_number
                column_link, loc = self.find_column(stream_in)
                if show_all is True:
                    stream_info = self.env.State.all_streams[stream_in-1]
                    assert stream_info.number == stream_in
                    stream_label = f"stream {stream_info.number} \n" + f"{int(stream_info.temperature)} K {int(stream_info.pressure)} Pa \n" + "".join(
                        [str(round(flow, 2)) + " mol/s \n" for flow in stream_info.flows])
                else:
                    stream_label = int(stream_in + 1)
                edges.append(pydot.Edge(nodes[column_link], nodes[i], label=stream_label, headport="w",
                                        tailport=loc))
                G.add_edge(edges[i - 1])
            else:
                G.add_edge(pydot.Edge(feed_node, nodes[0], label=1, headport="w", tailport="e"))

        # add outlet streams
        for outlet_stream in self.env.State.final_outlet_streams:
            column_link, loc = self.find_column(outlet_stream.number)
            label = f"{outlet_stream.number} \n" + f"revenue $" + \
                    str(round(self.env.stream_value(outlet_stream.flows)/1e6, 2)) + \
                    " M\n" + \
                    "".join([str(round(flow, 2)) + " mol/s \n" for flow in outlet_stream.flows])
            outlet_nodes.append(
                pydot.Node(label, shape="box", color="white"))
            G.add_node(outlet_nodes[-1])
            G.add_edge(pydot.Edge(nodes[column_link], outlet_nodes[-1], label=f" Submitted {outlet_stream.number}"
                                  ,headport="w", tailport=loc))

        for outlet_stream in self.env.State.streams:
            label = f"{outlet_stream.number} \n" +f"revenue $" + \
                    str(round(self.env.stream_value(outlet_stream.flows)/1e6, 2)) + \
                    " M\n" + \
                    "".join([str(round(flow, 2)) + " mol/s \n" for flow in outlet_stream.flows])
            column_link, loc = self.find_column(outlet_stream.number)
            outlet_nodes.append(
                pydot.Node(label, shape="box", color="white"))
            G.add_node(outlet_nodes[-1])
            G.add_edge(pydot.Edge(nodes[column_link], outlet_nodes[-1], label=f" not-submitted {outlet_stream.number}"
                                  ,headport="w", tailport=loc))
        plt = Image(G.create_png())
        display(plt)
        return G

    def find_column(self, stream):
        for i, column_info in enumerate(self.env.State.column_data):
            if stream == column_info.tops_number:
                loc = "ne"
                return i, loc
            elif stream == column_info.bottoms_number:
                loc = "se"
                return i, loc