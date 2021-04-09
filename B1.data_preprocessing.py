import os
import pickle
import numpy as np
import pandas as pd


data = pd.read_csv('flight-scheduling.csv')
data_result = pd.read_csv('./result/result_cost.csv').iloc[:-1,:]
data_result['Flight number'] = data_result['Flight number'].astype('int')

data = pd.merge(data, data_result[['Flight number', 'total_cost_A', 'x_A']], how='inner', on='Flight number')
data = data[data['x_A'] == 1] # get flight number assigned fleet A

data.columns

# convert time format HHMM to only minutes
_time = pd.to_datetime(data['Departure Time'], format='%H:%M')
departure_time = _time.dt.hour * 60 + _time.dt.minute
_time = pd.to_datetime(data['Arrival Time'], format='%H:%M')
arrive_time = _time.dt.hour * 60 + _time.dt.minute

data['departure_time'] = departure_time
data['arrive_time'] = arrive_time

data_sub = data[['Flight number', 'Origin', 'Destination', 'departure_time',
                 'arrive_time', 'Flight Hours', 'Distance', 'total_cost_A']]
colname = np.array(['flight_number', 'origin', 'destination', 'departure_time',
                    'arrive_time', 'flight_hours', 'distance', 'total_cost_A'])
data_sub.columns = colname

i = 'origin'

data_dict = {}
for i in colname:
    maine_var = i
    data_dict[maine_var] = {}
    sub_var = colname[~np.isin(colname, maine_var)]
    for j in range(len(data_sub)):
        obs = data_sub.iloc[j, :]

        if obs[maine_var] not in list(data_dict[maine_var].keys()):
            data_dict[maine_var][obs[maine_var]] = {}

        for k in sub_var:
            if k not in list(data_dict[maine_var][obs[maine_var]].keys()):
                data_dict[maine_var][obs[maine_var]][k] = []
                data_dict[maine_var][obs[maine_var]][k].append(obs[k])
            else:
                data_dict[maine_var][obs[maine_var]][k].append(obs[k])


fnum_list = np.array(list(data_dict['flight_number'].keys()))
arrive_time_list = np.array(list(data_dict['arrive_time'].keys()))
departure_time_list = np.array(list(data_dict['departure_time'].keys()))

preparetime = 30
# get flight info


def nextstep(data_dict, fnum, preparetime=30):
    _ori = data_dict['flight_number'][fnum]['origin']
    _dest = data_dict['flight_number'][fnum]['destination'][0]
    _dept = data_dict['flight_number'][fnum]['departure_time'][0]
    arrive_time = data_dict['flight_number'][fnum]['arrive_time'][0]

    # get possible departure time
    readydep = arrive_time + preparetime

    # get list of flight list at certain possible depature time
    dest_time_list = np.array(data_dict['origin'][_dest]['departure_time'])
    dest_fnum_list = np.array(data_dict['origin'][_dest]['flight_number'])
    next_fnum_list = dest_fnum_list[dest_time_list > readydep]
    return list(next_fnum_list)


def search(data_dict, flight_number_list, rout, depth_dict, depth, _preparetime):
    depth += 1
    for _fnum in flight_number_list:
        depth_dict[depth] = _fnum
        flight_number_list = nextstep(data_dict, _fnum, _preparetime)
        if len(flight_number_list) == 0:
            rout_seq = ','.join([str(depth_dict[depth_i]) for depth_i in range(1, depth + 1)])
            rout[rout_seq] = {'start': depth_dict[1],
                              'start_airport': data_dict['flight_number'][depth_dict[1]]['origin'][0],
                              'last': depth_dict[depth],
                              'last_airport': data_dict['flight_number'][depth_dict[depth]]['destination'][0]}

        search(data_dict, flight_number_list, rout, depth_dict, depth, _preparetime)


fnum_list = list(data_dict['flight_number'].keys())
depth_dict = {}; depth = 0; rout = {}
search(data_dict, fnum_list, rout, depth_dict, depth, preparetime)



total_rout = {}
for i, r in enumerate(list(rout.keys())):
    total_rout[i] = {}
    total_rout[i][1] = {}
    total_rout[i][1] = rout[r]
    total_rout[i][1]['rout'] = r

total_rout2 = {}
k = 0
for j in range(len(total_rout)):
    last_air = total_rout[j][1]['last_airport']
    for i in range(j, len(total_rout)):
        if last_air == total_rout[i][1]['start_airport']:
            a = total_rout[i][1]['rout'].split(',')
            b = total_rout[j][1]['rout'].split(',')
            if len(np.intersect1d(a, b)) == 0:
                total_rout2[k] = {}
                total_rout2[k][1] = total_rout[j][1]
                total_rout2[k][2] = total_rout[i][1]
                k += 1


total_rout3 = {}
k = 0
for j in range(len(total_rout2)):
    last_air = total_rout2[j][2]['last_airport']
    for i in range(j, len(total_rout2)):
        if last_air == total_rout2[i][2]['start_airport']:
            a = total_rout2[i][2]['rout'].split(',')
            b = total_rout2[j][1]['rout'].split(',')
            c = total_rout2[j][2]['rout'].split(',')
            b = b + c
            if len(np.intersect1d(a, b)) == 0:
                if 'JFK' not in [total_rout2[j][1]['last_airport'], total_rout2[j][2]['last_airport'], total_rout2[i][1]['last_airport']]:
                    continue
                if total_rout2[j][1]['start_airport'] != total_rout2[i][1]['last_airport']:
                    continue
                total_rout3[k]={}
                total_rout3[k][1] = total_rout2[j][1]
                total_rout3[k][2] = total_rout2[j][2]
                total_rout3[k][3] = total_rout2[i][2]
                k += 1
total_rout3[0]
len(total_rout3)# len 1437
np.save('./data/total_rout.npy',total_rout3)
len(total_rout3)
total_rout3[0]

# dictionary to dataframe form
# rout f_day1 f_day2 f_day3 a_day1 a_day2 a_day3 c_day1 c_day2 c_day3

colname = ['rout'] + [ f'day{i+1}_rout' for i in range(3)] + \
          [f'day{i+1}_airport' for i in range(3)] + \
          [f'day{i+1}_flight_hours' for i in range(3)] +\
          [f'day{i+1}_distance' for i in range(3)] +\
          [f'day{i+1}_cost' for i in range(3)]
'''
['rout', 'day1_rout', 'day2_rout', 'day3_rout', 'day1_airport', 'day2_airport', 'day3_airport', 
'day1_flight_hours', 'day2_flight_hours', 'day3_flight_hours', 'day1_distance', 'day2_distance',
 'day3_distance', 'day1_cost', 'day2_cost', 'day3_cost']
'''
result_data = pd.DataFrame([], columns= colname, index= range(len(total_rout3)))
i=0
rout_day=1
for i in range(len(total_rout3)):
    for rout_day in range(1,4):
        result_data.loc[i,'rout'] = i
        rout = total_rout3[i][rout_day]['rout']
        result_data.loc[i, f'day{rout_day}_rout'] = rout

        rout_list = rout.split(',')
        day_flight_hours = 0
        day_distance = 0
        day_cost = 0
        day_departure_cnt = 0
        day_airport = ''
        fnum = int(rout_list[0])

        for fnum in rout_list:
            fligt_num_info = data_dict['flight_number'][int(fnum)]
            day_flight_hours += fligt_num_info['flight_hours'][0]
            day_distance += fligt_num_info['distance'][0]
            day_cost += fligt_num_info['total_cost_A'][0]
            day_airport += fligt_num_info['origin'][0] + ' to ' + fligt_num_info['destination'][0] + ', '
        day_departure_cnt = len(rout_list)

        result_data.loc[i, f'day{rout_day}_rout'] = rout
        result_data.loc[i, f'day{rout_day}_airport'] = rout
        result_data.loc[i, f'day{rout_day}_flight_hours'] = rout
        result_data.loc[i, f'day{rout_day}_distance'] = rout
        result_data.loc[i, f'day{rout_day}_cost'] = rout
