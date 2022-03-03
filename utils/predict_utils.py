# -*- coding: UTF-8 -*-
__author__ = "liyi"
__email__ = "liyi_xa@piesat.cn"
__data__ = "2022.3.2"
__description__ = "Script files are auxiliary function files for predicting the big picture "

import os
import numpy as np
import torch
from osgeo import ogr, gdal, gdalconst

shpFileKey = ['.cpg', '.dbf', '.prj', '.shx', '.shp']


class PredictUtils:
    def __init__(self, opt, num_to_color):
        """
        SegPieGeoUtils的初始化函数
        @param opt: 初始化参数,由config.py文件进行定义
        @param num_to_color: 类别标签 dict{"index":color->list}
        """
        super(PredictUtils, self).__init__()
        self.opt = opt
        self.s3Client = None
        self.test_gdal = None
        self.total_output_size = 0
        self.outputFile = None
        self.num_to_color = num_to_color

    @staticmethod
    def read_geo_img(img_path, gdal_=gdal):
        """
        用gdal读取遥感图像
        @param img_path 单图像的路径。
        @return: image_data->np.array((h,w,3)), dataset ->gdal.open
        """
        dataset = gdal_.Open(img_path)
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        band = dataset.RasterCount
        im_data = dataset.ReadAsArray(0, 0, width, height)  # .astype(np.float32)
        if band >= 3:
            im_data = np.transpose(im_data, (1, 2, 0))  # 转换为WHC
            im_data = im_data[:, :, 0:3]  # 只取前三个波段
        elif band == 1:
            im_data = im_data
        return im_data, dataset

    def pred_big_geo_img(self, index, opt, model, img, dataset, img_path, slide_window_size=1024, overlap_rate=0.25):
        """
        函数简介：本函是是将img大图进行滑框取图，然后检测每个小图。最后将小图的结果拼接到大图上。
        @param opt:初始化参数。
        @param img：图像
        @param detector：由CtdetDetector生成的类，包含测试等函数。
        @param slide_window_size：滑框尺寸。
        @param overlap_rate：重叠率。取值在[0, 1]
        @return: 大图的检测框 tensor[num, 6]
        """
        tmpfilename = os.path.basename(img_path)
        filename, extension = os.path.splitext(tmpfilename)

        filename = str(index + 1) + "_" + filename

        self.outputFile = os.path.join(self.opt.save_path, 'picture', filename + "_pred.tif")
        print('outputFile path is [{}]'.format(self.outputFile))

        output_ds_3band = self.write_array_to_tif_init(dataset, self.outputFile, 3)
        output_ds_1band = self.write_array_to_tif_init(dataset, os.path.join(self.opt.save_path, 'picture', filename + '_mask.tif'), 1)

        height, width, band = img.shape
        # 滑框的重叠率
        overlap_pixel = int(slide_window_size * (1 - overlap_rate))

        # ------------------------------------------------------------------#
        #                处理图像各个维度尺寸过小的情况。
        # ------------------------------------------------------------------#
        if height - slide_window_size < 0:  # 判断x是否超边界，为真则表示超边界
            x_idx = [0]
        else:
            x_idx = [x for x in range(0, height - slide_window_size + 1, overlap_pixel)]
            if x_idx[-1] + slide_window_size > height:
                x_idx[-1] = height - slide_window_size
            else:
                x_idx.append(height - slide_window_size)
        if width - slide_window_size < 0:
            y_idx = [0]
        else:
            y_idx = [y for y in range(0, width - slide_window_size + 1, overlap_pixel)]
            if y_idx[-1] + slide_window_size > width:
                y_idx[-1] = width - slide_window_size
            else:
                y_idx.append(width - slide_window_size)
        # ----------------------------------------------------------------------#
        #                判断下x,y的尺寸问题，并且设置裁剪大小，方便后续进行padding。
        # ----------------------------------------------------------------------#
        cut_width = slide_window_size
        cut_height = slide_window_size

        if height - slide_window_size < 0 and width - slide_window_size >= 0:  # x小，y正常
            cut_width = slide_window_size
            cut_height = height
            switch_flag = 1
        elif height - slide_window_size < 0 and width - slide_window_size < 0:  # x小， y小
            cut_width = width
            cut_height = height
            switch_flag = 3
        elif height - slide_window_size >= 0 and width - slide_window_size < 0:  # x正常， y小
            cut_height = slide_window_size
            cut_width = width
            switch_flag = 2
        elif height - slide_window_size >= 0 and width - slide_window_size >= 0:
            switch_flag = 0

        # ----------------------------------------------------------------------#
        #                开始滑框取图，并且获取检测框。
        # ----------------------------------------------------------------------#
        total_progress = len(x_idx) * len(y_idx)
        count = 0
        count_total = 0
        process = "%.1f%%" % (0)
        print("[ROUTINE] [{process}]".format(process=process), flush=True)
        for x_start in x_idx:
            for y_start in y_idx:
                count += 1

                croped_img = img[x_start:x_start + cut_height, y_start:y_start + cut_width]

                if band > 3:
                    croped_img = croped_img[:, :, 0:3]
                # ----------------------------------------------------------------------#
                #                依据switch_flag的设置，进行padding。
                # ----------------------------------------------------------------------#
                temp = np.zeros((slide_window_size, slide_window_size, 3), dtype=np.uint8)
                if switch_flag == 1:
                    # temp = np.zeros((croped_img.shape[0], cut_height, croped_img.shape[2]), dtype=np.uint8) #此为遥感图像
                    temp[0:cut_height, 0:croped_img.shape[1], :] = croped_img
                    croped_img = temp
                elif switch_flag == 2:
                    # temp = np.zeros((cut_size, croped_img.shape[1], croped_img.shape[2]), dtype=np.uint8)
                    temp[0:croped_img.shape[0], 0:cut_width, :] = croped_img
                    croped_img = temp
                elif switch_flag == 3:
                    temp[0:cut_height, 0:cut_width, :] = croped_img
                    croped_img = temp

                # ----------------------------------------------------------------------#
                #                开始检测。
                # ----------------------------------------------------------------------#
                # ks = croped_img.transpose(1, 2, 0).copy()
                croped_img = croped_img / 255.0
                croped_img = croped_img.astype(np.float32)
                croped_img = croped_img.transpose(2, 0, 1)
                croped_img = torch.tensor(croped_img).unsqueeze(0).to(opt.device)
                model.eval()
                with torch.no_grad():
                    output = model(croped_img)
                    pred = np.zeros((slide_window_size, slide_window_size)) + int(output.detach().cpu().numpy().argmax())
                    self.write_array_to_tif_small(output_ds_3band, pred, x_start, y_start, RGB=True)
                    self.write_array_to_tif_small(output_ds_1band, pred, x_start, y_start, RGB=False)
                count_total += 1
                now_progress = int(100 * count_total / total_progress)
                process = "%.1f%%" % (now_progress)
                print("[ROUTINE] [{process}]".format(process=process), flush=True)
                del (croped_img, output)


    @staticmethod
    def write_array_to_tif_init(dataset, output_path, band_size):
        """
         此函数时将小块的预测mask，写入之前创立的tif图层中。
         @param pred_mask-> tensor[h,w,num_class]
         @param outRaster -> gdal.GetDriverByName(format).create
         @param x_start -> 此时小图左上x坐标 -> int
         @param y_start -> 此时小图左上y坐标 -> int
         """
        width = dataset.RasterXSize
        height = dataset.RasterYSize

        geo_trans = dataset.GetGeoTransform()
        projection = dataset.GetProjection()

        format = "GTiff"
        tiff_driver = gdal.GetDriverByName(format)
        output_ds = tiff_driver.Create(output_path, width, height,
                                       band_size, gdalconst.GDT_Byte)
        output_ds.SetGeoTransform(geo_trans)
        output_ds.SetProjection(projection)
        for band_index in range(band_size):
            output_ds.GetRasterBand(band_index + 1).SetNoDataValue(0)
        return output_ds

    def write_array_to_tif_small(self, output_ds, one_channel_mask, x_start, y_start, RGB=False):
        """
        将小块的预测结果写入到tif中
        @param output_ds: 预先设置好参数的tif实例
        @param one_channel_mask: 单通道mask
        @param x_start: 左上起始点的x坐标
        @param y_start: 左上起始点的y坐标
        @param RGB: 判断写入的图像是否是RGB图像,默认是Fasle
        """
        if RGB == True:
            RGB_mask = self.draw_mask(one_channel_mask)
            for band_index in range(3):
                output_ds.GetRasterBand(band_index + 1).WriteArray(RGB_mask[:, :, band_index], y_start, x_start)
        else:
            output_ds.GetRasterBand(1).WriteArray(one_channel_mask, y_start, x_start)

    def draw_mask(self, one_channel_mask):
        """
        将mask生成对应的RGB图像.
        @param one_channel_mask: 类别标签 dict{"index":color->list}
        @return: rgb图像
        """
        rgb_img = np.zeros((one_channel_mask.shape[0], one_channel_mask.shape[1], 3))
        rgb_img[:, :, 0] = self.num_to_color[np.unique(one_channel_mask)[0]][0]
        rgb_img[:, :, 1] = self.num_to_color[np.unique(one_channel_mask)[0]][1]
        rgb_img[:, :, 2] = self.num_to_color[np.unique(one_channel_mask)[0]][2]
        return rgb_img