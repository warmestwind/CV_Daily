import numpy as np
import pandas as pd
import os
from glob import glob
import SimpleITK as sitk
import pydicom
from dcm_series_reader import *
import nrrd
from show_multi_series import *
from tqdm import tqdm

df = pd.DataFrame(None, columns=['ID', 'img_shape', 'no_7_shape', 'with_7_shape', 'ratio'])

class dataset_generater:
    def __init__(self, root_addr):
        self.root = root_addr
        self.depth = self.height = self.width = 0
        self.delta_depth = self.delta_height = self.delta_width = 0.0

    def itkdcm_reader(self):
        try:
            series_reader = sitk.ImageSeriesReader()
            filenamesDICOM = series_reader.GetGDCMSeriesFileNames(os.path.join(self.root,
                                                                               trad_list[i],
                                                                               'dicom'))#,
                                                                               #'origindicom'))
            series_reader.SetFileNames(filenamesDICOM)
            series_reader.LoadPrivateTagsOn()
            image = series_reader.Execute()
            self.temp_img = sitk.GetArrayFromImage(image)
            self.temp_shape = self.temp_img.shape
        except:
            print("dicom read error")

    def nrrdmask_reader(self):
        self.temp_mask = np.transpose(nrrd.read(os.path.join(self.root, trad_list[i], 'mask.nrrd'))[0], (2, 1, 0))
        #self.temp_mask = np.transpose(nrrd.read('/home/huiying/WQ/CTAOnline_Offline/cutdicom821/cut/segmask_train/616988liwanzhong_final_label.nrrd')[0], (2, 1, 0))
        # self.temp_mask = np.rot90(self.temp_mask, k=1, axes=(1, 2))

    def find_box3d(self, array):
        #https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
        d = np.any(array, axis=(1, 2))
        h = np.any(array, axis=(0, 2))
        w = np.any(array, axis=(0, 1))
        dmin, dmax = np.where(d)[0][[0, -1]]
        hmin, hmax = np.where(h)[0][[0, -1]]
        wmin, wmax = np.where(w)[0][[0, -1]]
        return dmin, dmax, hmin, hmax, wmin, wmax

    def crop_img_based_mask(self):
        '''
        itk dicom,nrrd mask, Z方向顺序是反的
        对齐nrrd mask和itk dicom,需要对mask tranpose(2,1,0) -> depth, height, width
        '''
        # with 7 box
        dmin, dmax, hmin, hmax, wmin, wmax = self.find_box3d(self.temp_mask)
        # no 7 box
        no_7_mask = self.temp_mask.copy()
        no_7_mask[no_7_mask == 7] = 0
        # print(np.sum(self.temp_mask))
        # print(np.sum(no_7_mask))
        dmin_n, dmax_n, hmin_n, hmax_n, wmin_n, wmax_n = self.find_box3d(no_7_mask)
        # print(dmin, dmax, hmin, hmax, wmin, wmax)
        # print(dmin_n, dmax_n, hmin_n, hmax_n, wmin_n, wmax_n)

        # depth change ratio
        delta_d = (dmax - dmin - dmax_n + dmin_n) / self.temp_img.shape[0]
        delta_h = (hmax - hmin - hmax_n + hmin_n) / self.temp_img.shape[1]
        delta_w = (wmax - wmin - wmax_n + wmin_n) / self.temp_img.shape[2]

        self.depth += self.temp_img.shape[0]
        self.height += self.temp_img.shape[1]
        self.width += self.temp_img.shape[2]

        self.delta_depth += delta_d
        self.delta_height += delta_h
        self.delta_width += delta_w

        return (dmax_n - dmin_n, hmax_n - hmin_n, wmax_n - wmin_n), \
               (dmax - dmin, hmax - hmin, wmax - wmin), \
               (delta_d, delta_h, delta_w)

    def show_series(self):
        multi_slice_viewer(self.temp_img, self.temp_mask)


trad_list = []
#dicom_dirs = ''
for item in os.listdir(dicom_dirs):
    trad_list.append(item)
trad_list.sort()
gen = dataset_generater(dicom_dirs)
l = len(trad_list)
total_no7_shape = [0, 0, 0]
total_with7_shape = [0, 0, 0]
for i in tqdm(range(len(trad_list))):
    #print(trad_list[i])
    gen.itkdcm_reader()
    gen.nrrdmask_reader()
    #dicom_array = gen.temp_img
    #mask = gen.temp_mask
    no_7_shape, with_7_shape, ratio = gen.crop_img_based_mask()
    dict_row = {'ID': trad_list[i], 'img_shape': gen.temp_shape, 'no_7_shape': no_7_shape,
                'with_7_shape':with_7_shape, 'ratio': ratio}
    for i in range(3):
        total_no7_shape[i] += no_7_shape[i]
    for i in range(3):
        total_with7_shape[i] += with_7_shape[i]
    df.loc[df.shape[0]] = dict_row
    #print(dicom_array.shape)
    #print(mask.shape)
    #print(d, h, w)
    #print("------------------------------")

df.to_csv("statistic_label7_result.csv", index=False, sep=',')

print("mean depth:", gen.depth/l)
print("mean height:", gen.height/l)
print("mean width:", gen.width/l)

print("mean delta depth:", gen.depth/l * gen.delta_depth/l)
print("mean delta height:", gen.height/l * gen.delta_height/l)
print("mean delta width:", gen.width/l * gen.delta_width/l)

print("mean delta depth%:", gen.delta_depth/l)
print("mean delta height%:", gen.delta_height/l)
print("mean delta width%:",  gen.delta_width/l)

print("mean no7 shape:", total_no7_shape[0]/l, total_no7_shape[1]/l, total_no7_shape[2]/l)
print("mean with7 shape:", total_with7_shape[0]/l, total_with7_shape[1]/l, total_with7_shape[2]/l)
print("label ratio:", total_no7_shape[0]/total_with7_shape[0],total_no7_shape[1]/total_with7_shape[1],total_no7_shape[2]/total_with7_shape[2])


#gen.show_series()


