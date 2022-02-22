import pandas as pd
import joblib
import sklearn
import scipy
import scipy.fft
from scipy import signal
from scipy.stats import skew, kurtosis
from app.utils import *

def filter_data():
    data = pd.read_csv('./csv/accelerometer.csv')
    sos = signal.cheby2(15, 20, 0.17, output='sos', analog=True)
    filtered_x = signal.sosfilt(sos, data['x'])
    filtered_y = signal.sosfilt(sos, data['y'])
    filtered_z = signal.sosfilt(sos, data['z'])

    with open('./csv/accelerometer.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['time', 'pid', 'x', 'y', 'z'])
        for i in range(len(filtered_x)):
            to_write = [data['time'][i], 'MC4267', filtered_x[i], filtered_y[i], filtered_z[i]]
            writer.writerow(to_write)

def process_data():
    data = pd.read_csv('./csv/accelerometer.csv')
    data['window'] = data['time'].apply(get_time_value)
    data['window10'] = data['time'].apply(get_time_ignore_second)
    data['selected_window_10'] = data['window'].apply(get_window_10)

    funcs = [('mean', np.mean), ('variance', np.var), ('median', np.median), ('min', np.min), ('max', np.max),
             ('rms', lambda x: np.sqrt(np.mean(np.square(x)))),
             ('energy_entropy', lambda x: energy_entropy(x, 10)),
             ('energy', energy), ('skew', skew), ('Kurtiosis', kurtosis)]
    fft_funcs = [('variance', np.var), ('spectral_centroid_spread', lambda x: spectral_centroid_spread(x, 40))]
    cols = ['x', 'y', 'z']

    look_up_frames = dict()
    for pid in data.pid.unique():
        response_frame = pd.DataFrame()
        grouped = data[data.pid == pid].groupby('window10')
        for col in cols:
            col_array = grouped[col].apply(np.array)
            col_fft = col_array.apply(scipy.fft.fft)
            for key, func in funcs:
                print(pid, ' :Working on', col, '   ', key)
                response_frame['_'.join(['win_10', col, key])] = col_array.apply(func)
            for key, func in fft_funcs:
                print(pid, ' :Working on', col, '   FFT ', key)
                response_frame['_'.join(['win_10', col, 'FFT', key])] = col_fft.apply(func)
        response_frame.pid = pid
        look_up_frames[pid] = response_frame

        final_frames = dict()
        for pid in data.pid.unique():
            response_frame = pd.DataFrame()
            grouped = data[data.pid == pid].groupby('window')
            for col in cols:
                col_array = grouped[col].apply(np.array)
                col_fft = col_array.apply(scipy.fft.fft)
                for key, func in funcs:
                    print(pid, ' :Working on', col, '   ', key)
                    response_frame['_'.join([col, key])] = col_array.apply(func)
                for key, func in fft_funcs:
                    print(pid, ' :Working on', col, '   FFT ', key)
                    response_frame['_'.join([col, 'FFT', key])] = col_fft.apply(func)
            response_frame['window10'] = grouped['window10'].apply(lambda x: x.unique().tolist()[0])
            response_frame['pid'] = pid
            final_frames[pid] = (response_frame)

            final_frame = pd.DataFrame()
            for key in final_frames:
                val = final_frames[key].copy()
                val.reset_index(drop=True, inplace=True)
                final_frame = pd.concat([final_frame, val])

            good = final_frame.dropna()
            good.to_csv('./csv/intermediate.csv')

            wow = pd.DataFrame()
            for key in final_frames:
                val1 = final_frames[key].copy()
                val1 = val1.reset_index(drop=True)
                val2 = look_up_frames[key].copy().reset_index()
                wow = pd.concat([wow, pd.merge(val1, val2, how='inner', on='window10')])
            wow = wow.dropna()

            wow.to_csv('./csv/final_file.csv')


def create_dataframe():
    frame = pd.read_csv('./csv/final_file.csv')

    frame = frame[[x for x in frame.columns if x != 'Unnamed: 0']]
    frame = frame[[x for x in frame.columns if x != 'pid']]
    frame = frame[[x for x in frame.columns if x != 'window10']]
    frame = frame[[x for x in frame.columns if x != 'win_10_x_FFT_spectral_centroid_spread']]
    frame = frame[[x for x in frame.columns if x != 'win_10_y_FFT_spectral_centroid_spread']]
    frame = frame[[x for x in frame.columns if x != 'win_10_z_FFT_spectral_centroid_spread']]
    frame = frame[[x for x in frame.columns if x != 'x_FFT_spectral_centroid_spread']]
    frame = frame[[x for x in frame.columns if x != 'y_FFT_spectral_centroid_spread']]
    frame = frame[[x for x in frame.columns if x != 'z_FFT_spectral_centroid_spread']]
    frame = frame[[x for x in frame.columns if x != 'key']]

    frame = frame.dropna()
    return frame


def get_prediction():
    process_data()
    frame = create_dataframe()

    # load, no need to initialize the loaded_rf
    loaded_rf = joblib.load("./models/rf_alcohol_detection.joblib")

    pred_cols = list(frame.columns.values)[:]

    # apply the whole pipeline to data
    pred = loaded_rf.predict(frame[pred_cols])
    print(pred)

    print(pred.mean())
    verdict = "Oui" if pred.mean() >= 0.4 else "Non"
    print('Bourré ? -> ' + verdict)

    return verdict
