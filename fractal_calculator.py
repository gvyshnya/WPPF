#!/usr/bin/python
import numpy as np
import pandas as pd
import Preprocessor
import FractalDimensionsSerializer
import FractalDimensions
import Discretizer


#######################################################################
# Auxilary functions needed in pre-processing
#######################################################################
def add_minutes_diff_45or15(datetime_value):
    """Return the artificial diff in min for datetime values with time matching xx:45 and xx:15 mask."""
    x = datetime_value
    result = 0 #default diff
    if x.minute == 45:
        result = 5 # we want 45 -> 50
    else:
        if x.minute == 15:
            result = 5 # we want 45 -> 50
        else:
            result = 0
    return result

################################################
# Main execution loop
################################################

# NOTES:
# - Set do_smoothening to 0 if you do not want to smoothen the training data for Power Production
# - Set do_smoothening to 1, if you like to smoothen power production time series
do_smoothening = 0

############################################
# Step 1: Reading Raw data into memory
############################################

# read raw csv data into Pandas DF
file_name = 'training.csv'
df = pd.read_csv(file_name,
                 encoding = 'ISO-8859-1', low_memory=True, index_col='Time (CET+0100)',
                 parse_dates=['Time (CET+0100)', 'Zeitstempel'])

# filter out the subset of cols needed for the training set
valuable_df = df.filter(['Time (CET+0100)',
                'Temperature, C (observation)',
                'Wind speed, km/h (observation)',
                'Wind direction, degrees (observation)',
                'Zeitstempel', 'Produktion [kW]'],
                        axis=1)


# further split valuable data into DF for independent and dependent variables
# this will be used for future NA elimination and data series timestamp homohenization
observations_df = valuable_df.filter(['Time (CET+0100)',
                'Temperature, C (observation)',
                'Wind speed, km/h (observation)',
                'Wind direction, degrees (observation)'],
                        axis=1)

results_df = valuable_df.filter(['Zeitstempel',
                                 'Produktion [kW]'],
                        axis=1)

# free memory
valuable_df = None
df = None

############################################
# Step 2: Pre-processing
############################################
# Step 2a: Remove records with at least 1 NA from the observations df
# #TODO: In production mode, this can be replaced with a smart imputation, using one of
#        climate/weather data homogenisation algorithms

# Drop the rows where any of the elements are nan
print("Dimentions of the training dataframe before NA elimantion: ", observations_df.shape)
observations_df["TMP"] = observations_df.index.values               # TMP temp col introduced, index is a DateTimeIndex
observations_df = observations_df[observations_df.TMP.notnull()]    # remove all NaT values
observations_df.drop(["TMP"], axis=1, inplace=True)                 # delete TMP again
print("Dimentions of the training dataframe after NA elimantion: ", observations_df.shape)

# Step 2b: replace hh:mm stamps of xx:50 with xx:45, xx:20 with xx:15,
#         to make sure time stamps matches the timestamps in Generated Production Power result variable
#TODO: In production mode, this shall be revisited based on the imputation/homohenization algo chosen
results_df['Zeitstempel'] = results_df['Zeitstempel'] + results_df['Zeitstempel'].apply\
    (lambda x: pd.Timedelta(add_minutes_diff_45or15(x), 'm'))

print("Preprocessing, step 2b, completed")

if do_smoothening == 1:
    # Step 2c: Apply custom pre-processing and smoothening to power production result variable
    Kt = 0.3 # TODO: make it a configurable parameter in production mode
    preprocessor = Preprocessor.Preprocessor(results_df['Produktion [kW]'].values,Kt)
    print("Preprocessing, step 3: Custom production power preprocessor initialized")

    q = 4 # TODO: make it a configurable parameter in production mode
    smooth_data = preprocessor.smoother(results_df['Produktion [kW]'].values, q)

    results_df['Produktion [kW]'].values = smooth_data

    print("Preprocessing, step 2c: Production power preprocessor data smoothened")
else:
    print("Preprocessing, step 2c: Skipped")

# Step 2d: merge observations_df and results_df to create the final training set dataframe
# redo obervations df dynamically to avoid 'Time (CET+0100)' being an index field there

observations_df2 = observations_df.filter([
                'Temperature, C (observation)',
                'Wind speed, km/h (observation)',
                'Wind direction, degrees (observation)'],
                axis=1)
observations_df2['Timestamp'] = observations_df.index.values

# do the actual merge
training_df = pd.merge(observations_df2, results_df, left_on='Timestamp', right_on='Zeitstempel')
# remove 'Zeitstempel' as it is a duplocate colunm in the result training dataframe
training_df = training_df.filter(['Timestamp',
                'Temperature, C (observation)',
                'Wind speed, km/h (observation)',
                'Wind direction, degrees (observation)',
                'Produktion [kW]'],
                axis=1)
print("Preprocessing, step 2d, completed")
print("Final dimentions of the training set: " , training_df.shape)

# free memory
observations_df2 = None
results_df = None

############################################
# Fractal calculations
############################################

fdms = FractalDimensionsSerializer.FractalDimensionsSerializer()

fdms.calculate_fractal_dimensions(training_df)

print ("Finished fractal dimensions calculation for the training set ...")

