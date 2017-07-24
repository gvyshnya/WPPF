#!/usr/bin/python
import numpy as np
import pandas as pd
import os
import math
import Discretizer
import FractalDimensions


class FractalDimensionsSerializer(object):
    """ This class manages cache of fractal dimentions calculated for historic observations in past with 3-h frame step
        The cache will serialize as a csv meanwhile (TODO: George V to change it to
    """

    def __init__(self):
        self.df_fractal_dimensions = None 	# Pandas DF with production and weather fractal dimentions
        self.fractal_dimensions_cache_file = "fractal_dimensions.csv"
        self.is_cache_in_memory = 0 # initially, cache is not in memory

    def calculate_fractal_dimensions (self, data_df):
        """ This method calculates and saves (to a csv file) fractal dimentions for weather data
            During the calculation, the original dataframe wiht observations (
                  :param
                    - data_df - a pandas dataframe with power production and weather data, with fields below:
                                 - 'Timestamp'
                                 - 'Temperature, C (observation)'
                                 - 'Wind speed, km/h (observation)'
                                 - 'Wind direction, degrees (observation)'
                                 - 'Produktion [kW]'

        """
        debug = 0

        # Cache dataframe (df)
        #   - Timestamp_start (datetime of the start of the 3-h interval)
        #   - Timestamp_end (datetime of the end of the 3-h interval)
        #   - D_boxcounting_production # Box counting statistic (power production)
        #   - D_infostat_production # Information Statistics (power production)
        #   - D_variance_production # Upper limit of variance of the information statistic (power production)
        #   - D_boxcounting_wind_direction # Box counting statistic (wind direction)
        #   - D_infostat_wind_direction # Information Statistics (wind direction)
        #   - D_variance_wind_direction # Upper limit of variance of the information statistic (wind direction)
        #   - D_boxcounting_wind_speed # Box counting statistic (wind speed)
        #   - D_infostat_wind_speed # Information Statistics (wind speed)
        #   - D_variance_wind_speed # Upper limit of variance of the information statistic (wind speed)
        #   - D_boxcounting_temperature # Box counting statistic (temperature)
        #   - D_infostat_temperature # Information Statistics (temperature)
        #   - D_variance_temperature # Upper limit of variance of the information statistic (temperature)
        col_list = ["Timestamp_start", "Timestamp_end","D_boxcounting_production", 'D_infostat_production',
                'D_variance_production', 'D_boxcounting_wind_direction', 'D_infostat_wind_direction',
                'D_variance_wind_direction', 'D_boxcounting_wind_speed', 'D_infostat_wind_speed',
                'D_variance_wind_speed', 'D_boxcounting_temperature', 'D_infostat_temperature',
                'D_variance_temperature']

        df = pd.DataFrame(columns=col_list)  # create an empty cache df with columns only

        alpha = 0.08
        index = 1

        # step to define the size of the window/frame to calculate fractal dimensions for
        # Note: in the prototype phase where we get 3 observations per hour in the refined training dataset, we will
        # set step = 8 to get 9 observations (3-h frame) to calculate fractal dimentions for - points [0..8],
        # [9..16], [17..24] etc.
        # TODO: it may require better generalization in future
        step = 8

        current_row = 0 # set to 0 to start data_df iteration from the very beginning
        slice_start_row = 0
        total_rows = data_df.shape[0]
        cache_df_row_id = 0  # initial row index in the cache df

        # iterate through the rows of the dataframe in step-size to calculate fractal dimensions for each time frame
        while slice_start_row < total_rows:

            slice_end_row = slice_start_row + step
            if (total_rows - slice_end_row ) < step:
                # at the end of the data frame where less then step remains in the tail, we do the final slice bigger
                slice_end_row = total_rows - 1

            if debug == 1: print ("Slicing the next piece of data: start row = ", slice_start_row,
                                  ", end row: ", slice_end_row)
            df_slice = data_df[slice_start_row:slice_end_row]

            start_time = data_df.iloc[slice_start_row]['Timestamp']  # start time of the slice with observations
            end_time = data_df.iloc[slice_end_row]['Timestamp']      # end time of the slice with observations

            production = df_slice['Produktion [kW]'].values
            temp = df_slice['Temperature, C (observation)'].values
            windSpeed = df_slice['Wind speed, km/h (observation)'].values
            windDir = df_slice['Wind direction, degrees (observation)'].values

            if debug == 1: print("Starting discretizing and fractal dimension calculations ... ")
            ##########################################
            # Step 1: Discretizing of production data
            # For pre-processed production data,
            # use entropy maximized discretization
            # Input: pre-processed data
            # Output: vector of number of discretized
            #     data values (discretization vector)
            ##########################################

            discretizer_p = Discretizer.Discretizer(production)
            vector_p = discretizer_p.discretize()
            if debug == 1: print ("vector_p: ", vector_p)

            ################################################################
            # Step 2: Calculate fractal dimensions for production data
            # Input: discretization vector, radius
            #     (depends on alpha = 0.08 - default)
            # Output: fractal dimensions (box
            #     counting, information dimension,
            #     variance limit)
            ################################################################

            fractal_p = FractalDimensions.FractalDimensions(len(vector_p), 0.08)
            rk = fractal_p.radius(2)
            if debug == 1: print ("rk = ", rk)

            Dbc_p = fractal_p.Dbc(vector_p, rk)
            Dinf_p = fractal_p.Dinf(vector_p, rk)
            Dvar_p = fractal_p.Dvar(vector_p, rk)
            if debug == 1: print ("Dbc_p = ", Dbc_p, ", Dinf_p = ", Dinf_p, ", Dvar_p = ", Dvar_p)

            ################################################################
            # Step 3: Discretization of weather data (temperature)
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
            # Step 4: Calculation of fractal dimensions
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
            # Step 5: Discretization of weather data
            #     continued (wind speed)
            ##########################################

            discretizer_s2 = Discretizer.Discretizer(windSpeed)
            vector_s2 = discretizer_s2.discretizeWeather()
            if debug == 1: print("vector_s2 (", len(vector_s2), "): ", vector_s2)
            binIndex = discretizer_s2.getBin(vector_s2, windSpeed[index])
            if debug == 1: print("windSpeed[", index, "] = ", windSpeed[index], ", bin = ", binIndex)

            #############################################
            # Step 6: Calculation of fractal dimensions
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
            # Step 7: Discretization of weather data
            #     continued (wind direction)
            ##########################################

            discretizer_d2 = Discretizer.Discretizer(windDir)
            vector_d2 = discretizer_d2.discretizeWeather()
            if debug == 1: print("vector_d2 (", len(vector_d2), "): ", vector_d2)
            binIndex = discretizer_d2.getBin(vector_d2, windDir[index])
            if debug == 1: print("windDir[", index, "] = ", windDir[index], ", bin = ", binIndex)

            #############################################
            # Step 8: Calculation of fractal dimensions
            #     continued (wind direction)
            #############################################

            fractal_d2 = FractalDimensions.FractalDimensions(len(vector_d2), alpha)
            rk = fractal_d2.radius(2)
            if debug == 1: print("rk = ", rk)

            Dbc_d2 = fractal_d2.Dbc(vector_d2, rk)
            Dinf_d2 = fractal_d2.Dinf(vector_d2, rk)
            Dvar_d2 = fractal_d2.Dvar(vector_d2, rk)
            if debug == 1: print("Dbc_d2 = ", Dbc_d2, ", Dinf_d2 = ", Dinf_d2, ", Dvar_d2 = ", Dvar_d2)

            # add a new record to the cache df
            # ["Timestamp_start", "Timestamp_end", "D_boxcounting_production", 'D_infostat_production',
            # 'D_variance_production', 'D_boxcounting_wind_direction', 'D_infostat_wind_direction',
            # 'D_variance_wind_direction', 'D_boxcounting_wind_speed', 'D_infostat_wind_speed',
            # 'D_variance_wind_speed', 'D_boxcounting_temperature', 'D_infostat_temperature',
            # 'D_variance_temperature']

            df.loc[cache_df_row_id] = [start_time, end_time, Dbc_p, Dinf_p, Dvar_p, Dbc_d2, Dinf_d2, Dvar_d2,
                                       Dbc_s2, Dinf_s2, Dvar_s2, Dbc_t2_temp, Dinf_t2_temp, Dvar_t2_temp]
            cache_df_row_id = cache_df_row_id + 1

            # go to the next slice
            slice_start_row = slice_end_row + 1

        # re-create the new cache file
        if os.path.exists(self.fractal_dimensions_cache_file):
            os.remove(self.fractal_dimensions_cache_file)
            if debug == 1: print("[FractalDimensionsSerializer.calculate_fractal_dimensions]: Removed the old cache file ...")
        df.to_csv(self.fractal_dimensions_cache_file, index=False) # we do not want row indexes in the cache csv file
        if debug == 1: print(
            "[FractalDimensionsSerializer.calculate_fractal_dimensions]: Created the new cache file ...")
        self.is_cache_in_memory = 0 # reset cache-in-memory flag to trigger reading newest data from the cache file
        # read the new cache data into memory
        self.read_fractal_dimensions()

    def read_fractal_dimensions(self):
        """ this method reads cache data in memory as Pandas DF with cached fractal dimention data for power production
            and weather data
        """
        if os.path.exists(self.fractal_dimensions_cache_file):
            df = pd.read_csv(self.fractal_dimensions_cache_file, encoding='ISO-8859-1', low_memory=True,
                             parse_dates=['Timestamp_start', 'Timestamp_end'])
            self.df_fractal_dimensions = df
            self.is_cache_in_memory = 1 # set flag of a latest cache being in memory, and no
        else:
            print("[FractalDimensionsSerializer.read_fractal_dimensions]: "
                  "Error occured: cache file is missing, please re-create it")
            # TODO: exit with error ...

    def get_fractal_dimensions_for_observation(self, observation_date_time):
        """ This method returns fractal dimensions for both power produaction and all 3 weather parameters
            based on the time of observation. It will return fractal dimentions for the record where the following is
            true:
            Timestamp_star <= observation_date_time <= Timestamp_end

            Returns:
            Cache data result dataframe (df) with columns as follows
        #   - D_boxcounting_production # Box counting statistic (power production)
        #   - D_infostat_production # Information Statistics (power production)
        #   - D_variance_production # Upper limit of variance of the information statistic (power production)
        #   - D_boxcounting_wind_direction # Box counting statistic (wind direction)
        #   - D_infostat_wind_direction # Information Statistics (wind direction)
        #   - D_variance_wind_direction # Upper limit of variance of the information statistic (wind direction)
        #   - D_boxcounting_wind_speed # Box counting statistic (wind speed)
        #   - D_infostat_wind_speed # Information Statistics (wind speed)
        #   - D_variance_wind_speed # Upper limit of variance of the information statistic (wind speed)
        #   - D_boxcounting_temperature # Box counting statistic (temperature)
        #   - D_infostat_temperature # Information Statistics (temperature)
        #   - D_variance_temperature # Upper limit of variance of the information statistic (temperature)
        #
        #   Such a data frame to contain one and only one record
        """

        if self.is_cache_in_memory == 0:
            self.read_fractal_dimensions()

        col_list = ["D_boxcounting_production", 'D_infostat_production',
                'D_variance_production', 'D_boxcounting_wind_direction', 'D_infostat_wind_direction',
                'D_variance_wind_direction', 'D_boxcounting_wind_speed', 'D_infostat_wind_speed',
                'D_variance_wind_speed', 'D_boxcounting_temperature', 'D_infostat_temperature',
                'D_variance_temperature']

        df = pd.DataFrame(columns=col_list)  # create an empty cache df with columns only

        total_cache_rows = self.df_fractal_dimensions.shape[0]
        current_row = 0

        while current_row < total_cache_rows:
            # start of the interval for the current fractal dimensions cache record
            start_time = self.df_fractal_dimensions.iloc[current_row]['Timestamp_start']
            # end of the interval for the current fractal dimensions cache record
            end_time = self.df_fractal_dimensions.iloc[current_row]['Timestamp_end']

            if (observation_date_time >= start_time) and (observation_date_time <= end_time):
                # the right cache record located - retrieving the pre-calculated fractal dimension values
                Dbc_p = self.df_fractal_dimensions.iloc[current_row]['D_boxcounting_production']
                Dinf_p = self.df_fractal_dimensions.iloc[current_row]['D_infostat_production']
                Dvar_p = self.df_fractal_dimensions.iloc[current_row]['D_variance_production']
                Dbc_d2 = self.df_fractal_dimensions.iloc[current_row]['D_boxcounting_wind_direction']
                Dinf_d2 = self.df_fractal_dimensions.iloc[current_row]['D_infostat_wind_direction']
                Dvar_d2 = self.df_fractal_dimensions.iloc[current_row]['D_variance_wind_direction']
                Dbc_s2 = self.df_fractal_dimensions.iloc[current_row]['D_boxcounting_wind_speed']
                Dinf_s2 = self.df_fractal_dimensions.iloc[current_row]['D_infostat_wind_speed']
                Dvar_s2 = self.df_fractal_dimensions.iloc[current_row]['D_variance_wind_speed']
                Dbc_t2_temp = self.df_fractal_dimensions.iloc[current_row]['D_boxcounting_temperature']
                Dinf_t2_temp = self.df_fractal_dimensions.iloc[current_row]['D_infostat_temperature']
                Dvar_t2_temp = self.df_fractal_dimensions.iloc[current_row]['D_variance_temperature']

                df.loc[1] = [Dbc_p, Dinf_p, Dvar_p, Dbc_d2, Dinf_d2, Dvar_d2,
                            Dbc_s2, Dinf_s2, Dvar_s2, Dbc_t2_temp, Dinf_t2_temp, Dvar_t2_temp]
                # force the loop to end
                current_row = total_cache_rows
            else:
                current_row = current_row + 1

        return df

    def get_fractal_dimensions_for_interval(self, interval_start_date, interval_end_date):
        """ This method will return the subset of cached intervals with pre-calculated
            fractal dimension values between interval_start_date and interval_end_date (inclusive)

            :param interval_start_date - start date of the interval (datetime)
            :param interval_end_date - end date of the interval (datetime)

            :return df - a data frame with fractal dimentions for weather data within the interval specified

                the following list will specify the set of columns in df:
                ['Timestamp_start', 'Timestamp_end', 'D_boxcounting_wind_direction', 'D_infostat_wind_direction',
                'D_variance_wind_direction', 'D_boxcounting_wind_speed', 'D_infostat_wind_speed',
                'D_variance_wind_speed', 'D_boxcounting_temperature', 'D_infostat_temperature',
                'D_variance_temperature']
        """
        if self.is_cache_in_memory == 0:
            self.read_fractal_dimensions()

        col_list = ['Timestamp_start', 'Timestamp_end', 'D_boxcounting_wind_direction', 'D_infostat_wind_direction',
                'D_variance_wind_direction', 'D_boxcounting_wind_speed', 'D_infostat_wind_speed',
                'D_variance_wind_speed', 'D_boxcounting_temperature', 'D_infostat_temperature',
                'D_variance_temperature']

        # subset of the records for a particular interval only
        df = self.df_fractal_dimensions[
            (self.df_fractal_dimensions['Timestamp_start'] >= interval_start_date ) &
            (self.df_fractal_dimensions['Timestamp_end'] <= interval_end_date)
        ]
        # filter only fractal dimensions for weather observations
        df = df.filter(col_list, axis=1)

        return df
