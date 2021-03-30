import os
import pickle
import numpy as np
import pandas as pd
import pulp

from util import Fleet, Network, transpose4pulp

# load flight-scheduling data
data = pd.read_csv('flight-scheduling-with-cost.csv', index_col=0)

# load network node data
with open('./network.pkl', 'rb') as f:
    network = pickle.load(f)
# load fleet data
with open('./fleet.pkl', 'rb') as f:
    fleet = pickle.load(f)

print("== for COMPLEX model")
LPmodel_complex = pulp.LpProblem(
    name="Complex_problem",
    sense=pulp.LpMinimize
)
fleet_list = list(fleet.keys())
fleet_type = fleet_list[0]

############################################
# Define VARIABLE
# 기존에는 각 변수를 한번에 하나씩만 만들었지만, 아래와 같이 dictionary로 변수들을 한번에 만들 수도있음.
# 또한, constraint에서도 이 딕셔너리에 key로 접근하여 변수들의 제한사항을 입력해줘야 함.
# 또한, var이 많아지면, 아래와 같이 dict로 한꺼번에 만들수도있음.
# transpose x which is fleet assign
flight_number_list = data['Flight number'].values
assigned = transpose4pulp(data, fleet_list, 'x')
index_list = [f'{fleet_type}_{i}' for fleet_type in fleet_list for i in flight_number_list]

vars_x = pulp.LpVariable.dicts(
    name='x',  # prefix of each LP var
    indexs=index_list,
    lowBound=0,
    upBound=1,
    cat='Binary'
)
print(vars_x)

index_list = []
for fleet_type in fleet_list:
    for airport in network.airport_list:
        for i in network.airport_nodes[airport].index.values:
            index_list.append(f'{fleet_type}_{airport}_{i}')

vars_y = pulp.LpVariable.dicts(
    name='node',  # prefix of each LP var
    indexs=index_list,
    lowBound=0,
    cat='Integer'
)

############################################
# Define OBJECTIVE function
# 이 때는 pulp.lpSum을 사용해서 비교적 간단하게 처리할 수도 있음.
coefficient = transpose4pulp(data, fleet_list, 'total_cost').astype(int)
obj_function = pulp.lpSum(c * v for c, v in zip(coefficient.values.reshape(-1, 1), vars_x.values()))
LPmodel_complex.objective = obj_function
print(LPmodel_complex.objective)

###
# 앞에서 variable을 dictionary를 사용해서 만들어줬으므로, 여기서도 key로 접근하여 constrain를 작성해줘야 함.
## constraint 1 : one flight assigned at least 1 aircraft
constraint_1 = []
for f_num in flight_number_list:
    g = None
    for fleet_type in fleet_list:
        g += vars_x[f'{fleet_type}_{f_num}']
    constraint_1.append(g == 1)

## constraint 2 : balance const.
constraint_2 = []
for fleet_type in fleet_list:
    for airport in network.airport_list:
        net_nodes = network.airport_nodes[airport][['Flight number',
                                                    'Origin',
                                                    'Destination']].astype('object')
        rep = len(net_nodes) // 2
        idx = np.tile([0, 1], rep)
        left_node = net_nodes.iloc[np.where(idx == 0)[0]]['Flight number'].values
        right_node = net_nodes.iloc[np.where(idx == 1)[0]]['Flight number'].values

        l_g = None
        r_g = None
        for node_num in left_node:
            l_g += vars_x[f'{fleet_type}_{node_num}']
        for node_num in right_node:
            r_g += vars_x[f'{fleet_type}_{node_num}']

        constraint_2.append(l_g == r_g)

## constraint 3 :
constraint_3 = []
for fleet_type in fleet_list:
    g = None
    for airport in network.airport_list:
        net_nodes = network.airport_nodes[airport][['Flight number',
                                                    'Origin',
                                                    'Destination']].astype('object')
        last_node_num = net_nodes.index.values[-1]
        g += vars_y[f'{fleet_type}_{airport}_{last_node_num}']
    constraint_3.append(g <= fleet[fleet_type].num_airplane)

## constraint 4 :
constraint_4 = []
for airport in network.airport_list:
    # set t-1 node flight
    net_nodes = network.airport_nodes[airport][['Flight number',
                                                'Origin',
                                                'Destination']].astype('object')
    seq_node = net_nodes.index.values
    seq_node_1 = np.concatenate((seq_node[-1:], seq_node[:-1]), axis=0)
    net_nodes['t'] = seq_node
    net_nodes['t_1'] = seq_node_1
    # assign 1 or -1 along origin or destination
    net_nodes['arrive'] = (net_nodes.Origin != airport).astype('int') * 2 - 1  # re-scaling -1 to 1
    net_nodes = net_nodes[['t', 't_1', 'Flight number', 'arrive']]

    # yA1,1 = yA6,1 - X110,1
    for idx in range(net_nodes.shape[0]):
        for fleet_type in fleet_list:
            net_node_inst = net_nodes.iloc[idx, :]
            constraint_4.append(vars_y[f'{fleet_type}_{airport}_{net_node_inst.t_1}'] + net_node_inst.arrive * vars_x[
                f'{fleet_type}_{net_node_inst["Flight number"]}'] == vars_y[
                                    f'{fleet_type}_{airport}_{net_node_inst.t}'])

## add constraint into model
constraints = constraint_1 + constraint_3 + constraint_4
for i, c in enumerate(constraints):
    constraint_name = f"const_{i}"
    LPmodel_complex.constraints[constraint_name] = c

## solve problem
LPmodel_complex.solve()
# 잘 풀렸는지 확인, infeasible 등이 없는지 확인할 것.
print("Status:", pulp.LpStatus[LPmodel_complex.status])
# for v in LPmodel_complex.variables():
#    print(f"Produce {v.varValue:5.1f} Cake {v}")

## arrange result as dictionary
#for v in LPmodel_complex.variables():
#    print(f"{v}:{v.value():5.1f}")
result_dict = {str(v): int(v.value()) for v in LPmodel_complex.variables()}


## y_jk
for fleet_type in fleet_list:
    for airport in network.airport_list:
        network.airport_nodes[airport][f'node_{fleet_type}'] = 0
        index_list = network.airport_nodes[airport].index.values
        for idx in index_list:
            node = f'node_{fleet_type}_{airport}_{idx}'
            network.airport_nodes[airport].loc[idx, f'node_{fleet_type}'] = result_dict[node]

result_df = None
result_summary = None
for airport in network.airport_list:
    if result_df is None:
        result_df = network.airport_nodes[airport]
        result_df['airport'] = airport
        result_summary = result_df.iloc[-1:, :]
    else:
        tmp = network.airport_nodes[airport]
        tmp['airport'] = airport
        result_df = pd.concat((result_df, tmp))
        result_summary = pd.concat((result_summary, tmp.iloc[-1:, :]))

result_summary.reset_index(drop=True, inplace=True)
result_summary = result_summary[['airport', 'Origin', 'Destination', 'node_A', 'node_B']]
result_summary = result_summary.append({'airport':'Total',
                                        'node_A':result_summary.iloc[:,-2].sum(),
                                        'node_B':result_summary.iloc[:,-1].sum()},
                                       ignore_index=True)
result_df.reset_index(drop=True, inplace=True)

## x_ij
flight_number_list = result_df['Flight number'].values
for idx in range(result_df.shape[0]):
    for fleet_type in fleet_list:
        fnum = result_df.loc[idx, 'Flight number']
        result_df.loc[idx, f'x_{fleet_type}'] = result_dict[f'x_{fleet_type}_{fnum}']
result_df.columns

result_df = result_df[['airport', 'Flight number', 'Origin',
                       'Departure Time', 'Destination',
                       'Arrival Time', 'x_A', 'x_B', 'node_A',
                       'node_B', ]]
result_cost = result_df[['Flight number', 'Origin',
                       'Departure Time', 'Destination',
                       'Arrival Time', 'x_A', 'x_B']]

result_cost = result_cost.drop_duplicates()
result_cost.reset_index(drop=True, inplace=True)
result_cost = pd.merge(data[['Flight number', 'total_cost_A', 'total_cost_B']], result_cost, how ='left', on='Flight number')
result_cost['loss'] = result_cost['total_cost_A'] * result_cost['x_A'] + result_cost['total_cost_B'] * result_cost['x_B']
result_cost = result_cost.append({'Flight number':'Total',
                                        'loss':result_cost.iloc[:,-1].sum()},
                                       ignore_index=True)



result_df.to_csv('result.csv', index=False)
result_summary.to_csv('result_summary.csv', index=False)
result_cost.to_csv('result_cost.csv', index=False)