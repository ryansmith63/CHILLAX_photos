from PIL import Image, ImageFont, ImageDraw, ImageOps
import os
import numpy as np
import datetime
import pandas as pd
import photo_processing as pp

filepath = 'photos/ForRyan' 

num_slices = 200
slice_length = 450
photo_slices = np.zeros((len(os.listdir(filepath)),slice_length)) 
datetimes = []

for i in range(len(os.listdir(filepath))): 
    #extract date and time of each photo
    filename = os.listdir(filepath)[i]
    file_datetime = datetime.datetime(int(filename[:4]),int(filename[5:7]),int(filename[8:10]),int(filename[11:13]),int(filename[13:15]),int(filename[15:17]))
    datetimes.append(file_datetime)
    
    #load photo in grayscale and profile pixel brightness across ice ring
    photo = np.asarray(Image.open(filepath + '/' + os.listdir(filepath)[i]).convert('L'),dtype=np.float)
    photo_slices[i] = pp.slice_photo(photo)

#use start of profile, in region clear of ice, to find baseline
baseline, std = pp.leading_baseline(photo_slices,300,0,100)
processed_slices = [photo_slices[i]-baseline[i] for i in range(len(baseline))]

rqs = []
for i in range(len(photo_slices)):
    photo_dict = {}
    
    #find "pulses" corresponding to elevated pixel brightness
    start, end = pp.std_dev_pulsefinding(processed_slices[i],std[i],300,60,rising_thresh=3.0,falling_thresh=1.0)

    photo_dict.update({('datetime') : datetimes[i]})
    photo_dict.update({('ice_start') : start})
    photo_dict.update({('ice_end') : end})
    photo_dict.update({('pulse_height') : max(processed_slices[i][start:end])})
    rqs.append(photo_dict)

rqs_df = pd.DataFrame(rqs)
rqs_df.to_pickle('results.txt')