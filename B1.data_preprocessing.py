import os
import pickle
import numpy as np
import pandas as pd


data = pd.read_csv('flight-scheduling.csv')
data.columns

# convert time format HHMM to only minutes
_time = pd.to_datetime(data['Departure Time'], format='%H:%M')
departure_time = _time.dt.hour * 60 + _time.dt.minute
_time = pd.to_datetime(data['Arrival Time'], format='%H:%M')
arrive_time = _time.dt.hour * 60 + _time.dt.minute

data['departure_time'] = departure_time
data['arrive_time'] = arrive_time

data_sub = data[['Flight number', 'Origin', 'Destination', 'departure_time', 'arrive_time', 'Flight Hours', 'Distance']]

colname = np.array(['flight_number', 'origin', 'destination', 'departure_time', 'arrive_time', 'flight_hours', 'distance'])
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

data_dict['flight_number']
data_dict['flight_number'].keys()
data_dict['departure_time']
data_dict['departure_time'].keys()
data_dict['origin']
data_dict['origin'].keys()

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
len(total_rout3)
np.save('./data/total_rout.npy',total_rout3)

total_rout3[0]