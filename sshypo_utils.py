import numpy as np 
import geopy
import geopy.distance
from obspy import read, Trace, Stream

def read_picks(pick_file_name):

    stations = np.array([])
    phases = np.array([])
    return_dict_array = np.array([])

    with open(pick_file_name, 'r') as pick_file:

        picks_data = pick_file.readlines()

        for line in picks_data[1:]:
            station_id = line.split()[4]
            station = station_id.split(".")[1]
            phase = line.split()[8]
            # print (station, phase)
            stations = np.append(stations, station)
            phases = np.append(phases, phase)
    
        stations = np.unique(stations)
        phases = np.unique(phases)

        for station in stations:

            phs = np.array([])
            pk_times = np.array([])

            for line in picks_data[1:]:
                st_id = line.split()[4]
                st = st_id.split(".")[1]
                    
                if st == station:
                    phase = line.split()[8]
                    pick_time = line.split()[1]+"T"+line.split()[2]
                    phs = np.append(phs, phase)
                    pk_times = np.append(pk_times, pick_time)

            p_pick = ""
            s_pick = ""

            for (ph, time) in zip(phs, pk_times):
                if ph == "P":
                    p_pick = time
                
                elif ph == "S":
                    s_pick = time

            # print (p_pick, s_pick)


            pk_dict = {
                    "station": station,
                    "p_pick_time": p_pick,
                    "s_pick_time": s_pick,
            }
            return_dict_array = np.append(return_dict_array, pk_dict)
    # print (return_dict_array)
    return return_dict_array

def calc_rms(arr):

    rms = 0

    for a in arr:

        rms += a**2

    rms = np.sqrt(rms/len(arr))
    return rms


def calc_polarity(arr):

    cum_displacement = 0
    for a in arr:
        cum_displacement += a

    if cum_displacement < 0:
        polarity = -1

    elif cum_displacement > 0:
        polarity = 1

    return polarity

def create_ts_tp_grid(distances, depths, vp, vs, ts_tp_time, t_err):

    grid = np.zeros((len(depths), len(distances)))
    arc_distances = np.array([])
    
    for de, depth in enumerate(depths):
        for di, dist in enumerate(distances):

            R = np.sqrt(dist**2 + depth**2)
            tp = R/vp
            ts = R/vs
            tsp = ts-tp
            grid[de, di] = tsp
            # print ("dd", tsp-ts_tp_time)
            if np.abs(tsp - ts_tp_time) < t_err:
#                 print (np.abs(tsp - ts_tp_time), R)
                arc_distances = np.append(arc_distances, R)
#     print (arc_distances)        
    arc_distance = np.median(arc_distances)

    return grid, arc_distance

def calc_hypo_lat_lon(station_lat, station_lon, radius, azimuth):

    # Define starting point.
    start = geopy.Point(station_lon, station_lat)

    # Define a general distance object, initialized with a distance of 1 km.
    d = geopy.distance.distance(kilometers = radius)

    # Use the `destination` method with a bearing of 0 degrees (which is north)
    # in order to go from point `start` 1 km to north.
    point = d.destination(point=start, bearing=np.deg2rad(180-azimuth))
    point = repr(tuple(point))
    string = "".join(point)
    lon, lat = float(string[1:-1].split(",")[0]), float(string[1:-1].split(",")[1])
    return lon, lat


from geographiclib.constants import Constants
from geographiclib.geodesic import Geodesic

def getEndpoint(lat1, lon1, d, bearing):
    geod = Geodesic(Constants.WGS84_a, Constants.WGS84_f)
    d = geod.Direct(lat1, lon1, bearing, d * 1000)

    # print(getEndpoint(28.455556, -80.527778, 317.662819, 130.224835))
    # print (d['lon2'], d['lat2'])
    return d['lon2'], d['lat2']

def calc_ps_ratio(tr_z, tr_h1, tr_h2, z_twindow, h_twindow, p_arrival, s_arrival):

    print (tr_z, tr_h1, tr_h2)
    z_window = tr_z.slice(starttime=p_arrival-z_twindow, endtime=p_arrival+z_twindow, nearest_sample=True)
    h1_window = tr_h1.slice(starttime=s_arrival, endtime=s_arrival+h_twindow, nearest_sample=True)
    h2_window = tr_h2.slice(starttime=s_arrival, endtime=s_arrival+h_twindow, nearest_sample=True)

    max_z = np.max(z_window.data)
    max_h1 = np.max(h1_window.data)
    max_h2 = np.max(h2_window.data)
    average_hmax = np.mean(np.array([max_h1, max_h2]))

    ps_ratio = max_z/average_hmax
    return ps_ratio

