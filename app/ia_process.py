import pandas as pd
import joblib
import sklearn


def get_prediction():
    frame = pd.read_csv('./csv/data.csv')

    # load, no need to initialize the loaded_rf
    loaded_rf = joblib.load("./models/rf_alcohol_detection.joblib")

    pred_cols = list(frame.columns.values)[:]
    print(pred_cols)

    # apply the whole pipeline to data
    pred = loaded_rf.predict(frame[pred_cols])

    print(pred)
    print('Bourré : ?' + str(pred[0]+1))
    return 'N' + str(pred[0]+1)