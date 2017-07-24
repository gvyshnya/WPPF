import numpy as np
import pandas as pd
import sklearn.metrics as skm
import math
import WPFDatasetSlicer
import FractalDimensions
import FractalDimensionsSerializer
import Discretizer
import kNN


class Modeller(object):
    """
        This class encapsulates forecasting and cross-validation capabilities needed by fractal dimension-based
        forecasting algorithms used in WPF
    """

    def __init__(self, predictors_df, targets, wpf_dataset_slicer, k1, weights1, k2, weights2):
        """ Constructor for the class

        :param predictors_df - a pandas dataframe with predictors - weather observation data from validation data set,
                             with fields below:
                                 - 'Timestamp'
                                 - 'Temperature, C (observation)'
                                 - 'Wind speed, km/h (observation)'
                                 - 'Wind direction, degrees (observation)'
        :param targets - a vector of target production values from validation set - used to calculate accuracy of
                         the forecast on validation set
        :param wpf_dataset_slicer - an initialized instance of WPFDatasetSlicer class
        :param k1 - an integer representing the number of nearest neighbors to look up in kNN search, phase 1 where
                    searching for closest match(-es) in pre-calculated fractal dimension intervals
                    (on 3h timeframe scale) performed
        :param weights1 - the list of float-value weights of each of each of observation parameters in kNN search,
                          phase 1
                          for weather observations (wind direction, wind speed, tempreture), it will be a 3-element
                          list, for example [1.0,1.0,1.0]
        :param k2 - an integer representing the number of nearest neighbors to look up in kNN search, phase 2 where
                    searching for closest match(-es) in power production observations within the training subset
                    performed
        :param weights2 - the list of float-value weights of each of each of observation parameters in kNN search,
                          phase 2
                          for weather observations (wind direction, wind speed, tempreture), it will be a 3-element
                          list, for example [1.0,1.0,1.0]
        """
        self.predictors_df = predictors_df
        self.targets = targets
        self._predictions = []
        self.wpf_dataset_slicer = wpf_dataset_slicer
        self.k1 = k1
        self.weights1 = weights1
        self.k2 = k2
        self.weights2 = weights2

    def predict(self):
        """ This method will predict production values based on already submitted predictors"""
        debug = 1

        index = 1
        alpha = 0.08
        start_of_cache_date = "2016-01-01 00:00:00"  # TODO: refactor, make more generic in production mode

        current_row = 0
        total_rows = self.predictors_df.shape[0]

        # initialize fractal dimensions cache
        fractal_cache = FractalDimensionsSerializer.FractalDimensionsSerializer()
        fractal_cache.read_fractal_dimensions()

        while current_row < total_rows:
            current_date = self.predictors_df.iloc[current_row]['Timestamp']

            # get the data - pre-calculated & cached fractal dimensions stats for current_date and some interval backward
            # Note: if current_date is not equal to an end date of any interval, all intervals where
            # Timestamp_end < current_date will be returned
            df_data = fractal_cache.get_fractal_dimensions_for_interval(start_of_cache_date, current_date)

            # eliminate everything except the weather observations-related fractal dimension data
            df_data_for_knn = df_data.filter(['D_boxcounting_wind_direction', 'D_infostat_wind_direction',
                'D_variance_wind_direction', 'D_boxcounting_wind_speed', 'D_infostat_wind_speed',
                'D_variance_wind_speed', 'D_boxcounting_temperature', 'D_infostat_temperature',
                'D_variance_temperature'],
                axis=1)

            # get 3-h time frame of observerations since current_date
            three_h_df = self.wpf_dataset_slicer.get_three_hours_of_observations_since_current_date(current_date)

            # calculating fractal observations for the 3-h window since the current date, on a fly
            temp = three_h_df['Temperature, C (observation)'].values
            windSpeed = three_h_df['Wind speed, km/h (observation)'].values
            windDir = three_h_df['Wind direction, degrees (observation)'].values

            if debug == 1: print("Starting discretizing and fractal dimension calculations for last 3-h timeframe... ")

            ################################################################
            # Step 1: Discretization of weather data (temperature)
            #     use heuristic discretization
            # Input: weather data
            # Output: vector of number of discretized
            #     data values (discretization vector)
            #     and find the index of bin
            ################################################################

            discretizer_t2 = Discretizer.Discretizer(temp)
            vector_t2 = discretizer_t2.discretizeWeather()
            if debug == 1: print("vector_t2 (", len(vector_t2), "): ", vector_t2)
            binIndex = discretizer_t2.getBin(vector_t2, temp[index])
            if debug == 1: print("temp[", index, "] = ", temp[index], ", bin = ", binIndex)

            #############################################
            # Step 2: Calculation of fractal dimensions
            # Input: discretization vector, radius
            #     (depends on alpha = 0.08 - default)
            # Output: fractal dimensions (box
            #     counting, information dimension,
            #     variance limit)
            #############################################

            fractal_t2 = FractalDimensions.FractalDimensions(len(vector_t2), alpha)
            rk = fractal_t2.radius(2)
            if debug == 1: print("rk = ", rk)

            Dbc_t2_temp = fractal_t2.Dbc(vector_t2, rk)
            Dinf_t2_temp = fractal_t2.Dinf(vector_t2, rk)
            Dvar_t2_temp = fractal_t2.Dvar(vector_t2, rk)
            if debug == 1: print("Dbc_t2 = ", Dbc_t2_temp, ", Dinf_t2 = ", Dinf_t2_temp, ", Dvar_t2 = ", Dvar_t2_temp)

            ##########################################
            # Step 3: Discretization of weather data
            #     continued (wind speed)
            ##########################################

            discretizer_s2 = Discretizer.Discretizer(windSpeed)
            vector_s2 = discretizer_s2.discretizeWeather()
            if debug == 1: print("vector_s2 (", len(vector_s2), "): ", vector_s2)
            binIndex = discretizer_s2.getBin(vector_s2, windSpeed[index])
            if debug == 1: print("windSpeed[", index, "] = ", windSpeed[index], ", bin = ", binIndex)

            #############################################
            # Step 4: Calculation of fractal dimensions
            #     continued (wind speed)
            #############################################

            fractal_s2 = FractalDimensions.FractalDimensions(len(vector_s2), alpha)
            rk = fractal_s2.radius(2)
            if debug == 1: print("rk = ", rk)

            Dbc_s2 = fractal_s2.Dbc(vector_s2, rk)
            Dinf_s2 = fractal_s2.Dinf(vector_s2, rk)
            Dvar_s2 = fractal_s2.Dvar(vector_s2, rk)
            if debug == 1: print("Dbc_s2 = ", Dbc_s2, ", Dinf_s2 = ", Dinf_s2, ", Dvar_s2 = ", Dvar_s2)

            ##########################################
            # Step 5: Discretization of weather data
            #     continued (wind direction)
            ##########################################

            discretizer_d2 = Discretizer.Discretizer(windDir)
            vector_d2 = discretizer_d2.discretizeWeather()
            if debug == 1: print("vector_d2 (", len(vector_d2), "): ", vector_d2)
            binIndex = discretizer_d2.getBin(vector_d2, windDir[index])
            if debug == 1: print("windDir[", index, "] = ", windDir[index], ", bin = ", binIndex)

            #############################################
            # Step 6: Calculation of fractal dimensions
            #     continued (wind direction)
            #############################################

            fractal_d2 = FractalDimensions.FractalDimensions(len(vector_d2), alpha)
            rk = fractal_d2.radius(2)
            if debug == 1: print("rk = ", rk)

            Dbc_d2 = fractal_d2.Dbc(vector_d2, rk)
            Dinf_d2 = fractal_d2.Dinf(vector_d2, rk)
            Dvar_d2 = fractal_d2.Dvar(vector_d2, rk)
            if debug == 1: print("Dbc_d2 = ", Dbc_d2, ", Dinf_d2 = ", Dinf_d2, ", Dvar_d2 = ", Dvar_d2)

            # construct key object for kNN calculation
            vector_of_keys = [ Dbc_d2, Dinf_d2, Dvar_d2, Dbc_s2, Dinf_s2, Dvar_s2,
                               Dbc_t2_temp, Dinf_t2_temp, Dvar_t2_temp ]

            # kNN search, phase 1:
            # Goal is to find k1 right timeframes in terms of euclidian distances for weather params'
            # fractal dimension values
            knn1 = kNN.kNN(df_data_for_knn, vector_of_keys, weights=self.weights1, k=self.k1)
            fractal_timeframe_ids = knn1.kNearestNeighbor() # get IDs of records in df_data with k1 neighbors

            if debug:
                print("[Modeller.predict]: kNN, phase 1 completed, fractal dimension IDs: ", fractal_timeframe_ids)

            # read IDs of records within fractal dimensions cache, read start  and end dates of each k1-ed interval,
            # construct a data frame with the raw weather observations and production data
            # from the data in training set
            list_of_raw_observation_df = [] # array of dataframes with raw weather observations (in training set)
            list_of_production_df = []      # array of dataframes with power production observations (in training set)

            for i in range(len(fractal_timeframe_ids)):
                current_row_id = int(fractal_timeframe_ids[i])
                interval_start_date = df_data.iloc[current_row_id]['Timestamp_start']
                interval_end_date = df_data.iloc[current_row_id]['Timestamp_end']

                # get weather observations
                observ_df = self.wpf_dataset_slicer.get_observations_for_date_interval(interval_start_date,
                                                                                             interval_end_date)
                # get power production values
                prod_df = self.wpf_dataset_slicer.get_production_values_for_date_interval(interval_start_date,
                                                                                             interval_end_date)
                list_of_raw_observation_df.append(observ_df)
                list_of_production_df.append(prod_df)
                if debug: print("[Modeller.predict]: completed reading raw observations by fractal dimensions, step ", i)
            # combine (bind) several observation DFs into a single one
            # combine (bind) several power production DFs into a single one
            production_df = pd.DataFrame(columns=['Timestamp', 'Temperature, C (observation)',
                        'Wind speed, km/h (observation)', 'Wind direction, degrees (observation)'])
            observations_df = pd.DataFrame(columns=['Timestamp',
                        'Produktion [kW]'])

            for i in range(len(list_of_raw_observation_df)):
                if i == 0:
                    production_df = list_of_production_df[i]
                    observations_df = list_of_raw_observation_df[i]
                else:
                    production_df = pd.concat([production_df, list_of_production_df[i]])
                    observations_df = pd.concat([observations_df, list_of_raw_observation_df[i]])

            # filter out all cols except weather observations themselves - make it ready for kNN step #2
            observations_df = observations_df.filter(['Temperature, C (observation)',
                        'Wind speed, km/h (observation)',
                        'Wind direction, degrees (observation)'],
                       axis=1)

            # prepare key values for kNN, step 2 - these are weather attributes of a current observation
            vector_of_keys = [self.predictors_df.iloc[current_row]['Temperature, C (observation)'],
                              self.predictors_df.iloc[current_row]['Wind speed, km/h (observation)'],
                              self.predictors_df.iloc[current_row]['Wind direction, degrees (observation)']]

            # kNN search, phase 2:
            # Goal is to find k2 power production values (from a training set) in terms of
            # euclidian distances to the raw point in the key points

            knn2 = kNN.kNN(observations_df, vector_of_keys, weights=self.weights2, k=self.k2)
            # get IDs of records in production_df corresponding to k2 neighbors
            power_production_ids = knn2.kNearestNeighbor()

            if debug:
                print("[Modeller.predict]: kNN, phase 2 completed, production observation IDs: ", power_production_ids)

            # calculate forecasted power production value as a mean of all power production data points
            # having indeses in power_production_ids
            power_total = 0
            for i in range(len(power_production_ids)):
                current_row_id = power_production_ids[i]
                current_power_production = production_df.iloc[current_row_id]['Produktion [kW]']
                power_total += current_power_production

            forecast = power_total / len(power_production_ids)
            self._predictions.append(forecast) # save forecasted value

            if debug: print ("Forecast step: ", current_row+1, "Forecasted Power Production: ", forecast)

            current_row = current_row + 1

    def rmse(self):
        """ This method validates RMSE of the predicted values on the validation set"""

        rmse = math.sqrt(skm.mean_squared_error(self.targets, self.predictions))
        return rmse

    def mean_absolute_percentage_error(self):
        y_true = self.targets
        y_pred = self.predictions

        ## Note: does not handle mix 1d representation
        # if _is_1d(y_true):
        #    y_true, y_pred = _check_1d_array(y_true, y_pred)

        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    @property
    def predictions(self):
        return self._predictions
