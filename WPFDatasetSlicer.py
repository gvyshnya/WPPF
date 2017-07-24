#!/usr/bin/python
import numpy as np
import pandas as pd
import os
import math


class WPFDatasetSlicer(object):
    """
        This class encapsulates auxilary pandas dataframe manipulation routines as follows
        - splitting the entire dataset of observations into training and validation parts, based on the date of the last
          record to be in the training part
        - returning the subset of observations for the last 3-h timeframe windows, based on the date specified
        - returning the subset of weather observations for a time interval specified
        - returning the subset of power production records for a time interval specified
    """

    def __init__(self, data_df):
        """ Constructor for the class

        :param data_df - a pandas dataframe with power production and weather data, with fields below:
                                 - 'Timestamp'
                                 - 'Temperature, C (observation)'
                                 - 'Wind speed, km/h (observation)'
                                 - 'Wind direction, degrees (observation)'
                                 - 'Produktion [kW]'
        """
        self.data_df = data_df
        self._training_df = None
        self._validation_df = None

    def split(self, datetime_value):
        """ This method will split the entire data_df into training and validation parts
            This method should be called just after you iniitalize a WPFDatasetSlicer instace

        :param datetime_value - a datetime value for an obervation to be the first one in the validation set;
                                it therefore assumes observations with earlier Timestamps to be in the training set
        """
        debug = 1
        if debug: print("Split date: '", datetime_value, "'")
        if debug: print("Data at split record: ", self.data_df[self.data_df['Timestamp'] == datetime_value])

        # TODO: check if df does not have the record with datetime_value
        first_validation_df_row_id = self.data_df[self.data_df['Timestamp'] == datetime_value].index.tolist()[0]
        total_rows = self.data_df.shape[0]
        if debug: print("Split on training and validation set by row #", first_validation_df_row_id)

        self._training_df = self.data_df[0:first_validation_df_row_id-1]
        self._validation_df = self.data_df[first_validation_df_row_id:total_rows-1]

    @property
    def validation_df(self):
        return self._validation_df

    @property
    def training_df(self):
        return self._training_df

    def get_three_hours_of_observations_since_current_date(self, current_date):
        """ This method will return last 9 records from one with the Timestamp = current_date

            :param current_date - the datetime stamp of the record to count the last 3-h observations from

            :return Pandas dataframe with weather observations ; its structure to be as follows
                    - 'Timestamp'
                    - 'Temperature, C (observation)'
                    - 'Wind speed, km/h (observation)'
                    - 'Wind direction, degrees (observation)'

            TODO: the current implementation is will have to be refactored in the post-prototype phase as it is
            based on the essential prototype-time convention - there is a dataframe with 3 observations per h
            available
        """

        records_backward = 8 # TODO: change it in post-prototype phase

        row_id = self.data_df[self.data_df['Timestamp'] == current_date].index.tolist()[0]

        slice_first_row_id = row_id - records_backward

        if slice_first_row_id < 0:
            slice_first_row_id = 0

        df = self.data_df[slice_first_row_id:row_id]

        #filter out production data from the result set to return
        df = df.filter(['Timestamp',
                        'Temperature, C (observation)',
                        'Wind speed, km/h (observation)',
                        'Wind direction, degrees (observation)'],
                        axis=1)
        return df

    def get_observations_for_date_interval(self, start_date, end_date):
        """ This method will return obserevation records between two datetime values
          (all rerocds from one with Timestamp = start_date and Timestamp = end_date, edge records inclusive)

                :param start_date - the datetime stamp of the start record in the interval
                :param end_date - the datetime stamp of the end record in the interval

                :return Pandas dataframe with weather observations ; its structure to be as follows
                        - 'Timestamp'
                        - 'Temperature, C (observation)'
                        - 'Wind speed, km/h (observation)'
                        - 'Wind direction, degrees (observation)'
        """

        # TODO: add exception handling when dates do not have records for in data_df
        start_row_id = self.data_df[self.data_df['Timestamp'] == start_date].index.tolist()[0]
        end_row_id = self.data_df[self.data_df['Timestamp'] == end_date].index.tolist()[0]

        df = self.data_df[start_row_id:end_row_id]

        # filter out production data from the result set to return
        df = df.filter(['Timestamp',
                        'Temperature, C (observation)',
                        'Wind speed, km/h (observation)',
                        'Wind direction, degrees (observation)'],
                       axis=1)
        return df

    def get_production_values_for_date_interval(self, start_date, end_date):
        """ This method will return obserevation records between two datetime values
            (all rerocds from one with Timestamp = start_date and Timestamp = end_date, edge records inclusive)

                :param start_date - the datetime stamp of the start record in the interval
                :param end_date - the datetime stamp of the end record in the interval

                :return Pandas dataframe with power production records ; its structure to be as follows
                                - 'Timestamp'
                                - 'Produktion [kW]'
        """
        # TODO: add exception handling when dates do not have records for in data_df
        start_row_id = self.data_df[self.data_df['Timestamp'] == start_date].index.tolist()[0]
        end_row_id = self.data_df[self.data_df['Timestamp'] == end_date].index.tolist()[0]

        df = self.data_df[start_row_id:end_row_id]

        # filter out weather observations data from the result set to return
        df = df.filter(['Timestamp',
                        'Produktion [kW]'],
                       axis=1)
        return df