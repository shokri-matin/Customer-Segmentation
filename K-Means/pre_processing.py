from sklearn.preprocessing import StandardScaler
import numpy as np

def pre_processing(data):
    data = data.where(np.isnan(data) == False)

    data[data <= 0] = .01
    data_skewless = np.log(data)
    scaler = StandardScaler().fit(data_skewless)
    datamart_normalized = scaler.transform(data_skewless)

    return datamart_normalized