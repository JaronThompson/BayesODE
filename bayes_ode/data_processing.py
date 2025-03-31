import pandas as pd
import numpy as np

# Function to process dataframes
def process_df(df: pd.DataFrame, sys_vars: list[str], inputs: list[str]):
    # return array of eval times T = [N, 1]
    # array of measurements X = [N, 2, len(sys_vars)]
    T = []
    X = []
    U = []

    # loop over each unique condition
    for treatment, comm_data in df.groupby("Treatments"):
        # make sure comm_data is sorted in chronological order
        comm_data.sort_values(by='Time', ascending=True, inplace=True)

        # pull evaluation times
        t_eval = np.array(comm_data['Time'].values, np.float32)

        # pull species data
        data = np.array(comm_data[sys_vars].values, np.float32)

        # pull inputs data 
        data_u = np.array(comm_data[inputs].values, np.float32)[0]

        # append data
        for i, tf in enumerate(t_eval[1:]):
            if not all(np.isnan(data[i + 1])):

                # append time
                T.append(tf)

                # append data
                X.append(np.stack([data[0], data[i + 1]], 0))

                # append inputs
                U.append(data_u)

    # return data
    return np.stack(T), np.stack(X), np.stack(U)