
from scipy import stats
import pandas as pd

def L(z):
    pdf = stats.norm.pdf(z, 0, 1)
    cdf = stats.norm.cdf(z)
    return pdf - z * (1 - cdf)


def transpose4pulp(data, fleet_list, columnname):
    flight_number_list = data['Flight number'].values
    transpose_var = None
    for fleet_type in fleet_list:
        if transpose_var is None:
            transpose_var = pd.DataFrame(data[f'{columnname}_{fleet_type}'].values.reshape(1, -1),
                                         columns=[f'{columnname}_{fleet_type}_{i}' for i in flight_number_list])
        else:
            trans = pd.DataFrame(data[f'{columnname}_{fleet_type}'].values.reshape(1, -1),
                                 columns=[f'{columnname}_{fleet_type}_{i}' for i in flight_number_list])
            transpose_var = pd.concat((transpose_var, trans), axis=1)
    return transpose_var


class Fleet:
    def __init__(self, fleet_type=None, airplane_type=None, num_airplane=None, num_seats=None, casm=None, rasm=None):
        self.type = fleet_type
        self.airplane_type = airplane_type
        self.num_airplane = num_airplane
        self.num_seats = num_seats
        self.casm = casm
        self.rasm = rasm


class Network:
    def __init__(self):
        self.airport_list = []
        self.airport_nodes = {}
        self.node_seq = {'ATL': [110, 131, 111, 132, 112, 133],
                         'BOS': [116, 137, 117, 138, 118, 139],
                         'IAD': [140, 119, 141, 120, 142, 121],
                         'LAX': [101, 102, 122, 103, 123, 124],
                         'MIA': [113, 134, 114, 135, 115, 136],
                         'ORD': [107, 128, 108, 129, 109, 130],
                         'SFO': [104, 105, 125, 106, 126, 127],
                         'JFK': [140, 125, 122, 137, 116, 119,
                                  131, 128, 107, 134, 110, 117,
                                  141, 113, 138, 101, 104, 132,
                                  129, 135, 142, 108, 120, 126,
                                  111, 123, 118, 114, 133, 136,
                                  102, 105, 124, 121, 127, 109,
                                  112, 130, 115, 139, 103, 106]}

    def add_nodes(self, airport, nodes):
        self.airport_nodes[airport] = nodes

