import os
import pickle
import numpy as np
import pandas as pd
from util import Fleet, Network, L


#input fleet info.
fleet = {  'A' : Fleet(fleet_type='A',
                       airplane_type='737-800',
                       num_airplane=9,
                       num_seats=162,
                       casm=14.14,
                       rasm=17.21),
           'B' : Fleet(fleet_type='B',
                       airplane_type='757-200',
                       num_airplane=6,
                       num_seats=200,
                       casm=15.18,
                       rasm=17.21)}

#load flight-scheduling
data = pd.read_csv('flight-scheduling.csv')
print('obs : {}, variable : {}'.format(data.shape[0], data.shape[1]))
print('variable list')
print(data.columns.values)

# operating cost
# operating_cost = casm of fleet × the distance × number of seats
# spill cost
# Expected spill cost for a fleet = expected number of passengers spill × RASM × distance
# L(z) = ϕ(z)−z*(1−Φ(z)) ϕ:pdf Φ:cdf
fleet_list = list(fleet.keys())
for fleet_type in fleet_list:
    # operating cost
    operating_cost = fleet[fleet_type].casm * data.Distance * fleet[fleet_type].num_seats
    data[f'operating_cost_{fleet_type}'] = operating_cost.astype(int)

for fleet_type in fleet_list:
    # spill cost
    spill = L((fleet[fleet_type].num_seats - data.Demand.values)/data['S.D'].values) * data['S.D'].values
    spill_cost = spill * fleet[fleet_type].rasm * data.Distance
    modified_spill_cost = spill * fleet[fleet_type].rasm * data.Distance * 0.85
    data[f'modified_spill_cost_{fleet_type}'] = modified_spill_cost.astype(int)

for fleet_type in fleet_list:
    # total cost
    data[f'total_cost_{fleet_type}'] = data[f'operating_cost_{fleet_type}'] + data[f'modified_spill_cost_{fleet_type}'].astype(int)

# assign aircraft
for fleet_type in fleet_list:
    data[f'x_{fleet_type}'] = 0

data.to_csv('flight-scheduling-with-cost.csv')

# check airport which has only origin or destination
if ~(np.unique(data.Destination.values) == np.unique(data.Origin.values)).all():
    raise print('origin and destination should be matched')


# select a airport
airport_list = np.unique(data[['Destination', 'Origin']].values.reshape(-1))
network = Network()
network.airport_list = airport_list

for airport in airport_list:
    node_idx = np.where(data[['Destination', 'Origin']] == airport)[0]
    # get nodes list at the airport
    airport_nodes = data[['Flight number',
                          'Origin',
                          'Departure Time',
                          'Destination',
                          'Arrival Time'] + [f'x_{fleet_type}' for fleet_type in fleet_list]].iloc[node_idx, :]

    #sort by Departure Time
    #nodes_time = pd.to_datetime(airport_nodes['Departure Time'], format='%H:%M').dt.time # convert to time format
    #nodes_sorting_index = nodes_time.argsort()
    #airport_nodes = airport_nodes.iloc[nodes_sorting_index,:]
    #airport_nodes.reset_index(drop=True, inplace=True)
    #sort by dataa lista
    idx = airport_nodes.index.values
    airport_nodes.index = airport_nodes['Flight number']
    airport_nodes = airport_nodes.loc[network.node_seq[airport], :]
    airport_nodes = airport_nodes.reset_index(drop=True)

    # add
    network.add_nodes(airport, airport_nodes)

#save data
with open('network.pkl', 'wb') as output:  # Overwrites any existing file.
    pickle.dump(network, output)

with open('fleet.pkl', 'wb') as output:
    pickle.dump(fleet, output)