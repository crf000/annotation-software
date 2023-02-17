import sys
import os
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import cv2
from PIL import Image


from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from UI_main import Ui_MainWindow
from Widget_vtk import widget_vtk

from utils import *

class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.__initData()
        self.__initUI()  # 界面绘制交给InitUi方法

    def __initData(self):
        self.widget_vtk = widget_vtk()
        #--参数--#
        self.__heart = np.zeros((20, 200, 200), dtype=np.uint16)  # 心脏掩码
        self.__heart_segmented = np.zeros((20, 200, 200), dtype=np.uint16)  # 心脏掩码
        self.__heartMark = np.zeros((20, 200, 200), dtype=np.uint16)  # 心脏掩码
        self.__spacing = np.array([1.4, 1.4, 9])  # 心脏掩码分辨率
        self.__spinCenter_vtk = np.array([0, 0, 0])  # 旋转中心

        self.__point_LVA = None
        self.__point_RVA = None
        self.__point_AV = None
        self.__point_PV = None
        self.__point_MV = None
        self.__point_TV = None

        self.plane_len = 220

        # 保证widget_vtkPlane的远近，避免滚轮调整大小
        self.__drawHeartMark()

        # 画坐标轴
        # self.widget_vtk.drawAxis()

    def __initUI(self):
        self.UI = Ui_MainWindow()
        self.UI.setupUi(self)
        self.UI.horizontalLayout_vtkPlane.addWidget(self.widget_vtk)

    #----私有函数-----#
    #--图像导入时初始化参数--#

    def __initParam(self, range_, spacing):
        X_max, X_min, Y_max, Y_min, Z_max, Z_min = range_
        # 初始化参数变量
        self.__spacing = spacing
        self.__spinCenter_vtk = np.array([0, 0, 0])  # 旋转中心

        def rand_():
            x = np.random.randint(X_min, X_max)
            y = np.random.randint(Y_min, Y_max)
            z = np.random.randint(Z_min, Z_max)
            return np.array([x, y, z])

        # 导入已经标记好的数据
        id = self.UI.lineEdit_labelPath.text()[-8: -4]
        record_coords = np.load(os.path.join(r'..\landmarks',
                                             'landmark_%s.npy' % id))
        self.__point_LVA = record_coords[0]
        self.__point_RVA = record_coords[1]
        self.__point_AV = record_coords[2]
        self.__point_MV = record_coords[3]
        self.__point_TV = record_coords[4]
        self.__point_PV = record_coords[5]
        # self.__renderPoints()



        self.__initParamWidget(self.UI.horizontalScrollBar_spinCenterX_vtk, X_max, -X_max, self.__spinCenter_vtk[0], 10)
        self.__initParamWidget(self.UI.horizontalScrollBar_spinCenterY_vtk, Y_max, -Y_max, self.__spinCenter_vtk[1], 10)
        self.__initParamWidget(self.UI.horizontalScrollBar_spinCenterZ_vtk, Z_max, -Z_max, self.__spinCenter_vtk[2], 10)

        # 内部函数 减少重复
        def func(spinBox_X, spinBox_Y, spinBox_Z,
                 horizontalSlider_X, horizontalSlider_Y, horizontalSlider_Z, point):
            self.__initParamWidget(spinBox_X, X_max, X_min, point[0], 1)
            self.__initParamWidget(spinBox_Y, Y_max, Y_min, point[1], 1)
            self.__initParamWidget(spinBox_Z, Z_max, Z_min, point[2], 1)

            self.__initParamWidget(horizontalSlider_X, X_max, X_min, point[0], 10)
            self.__initParamWidget(horizontalSlider_Y, Y_max, Y_min, point[1], 10)
            self.__initParamWidget(horizontalSlider_Z, Z_max, Z_min, point[2], 10)
        # 控件参数初始化
        func(self.UI.spinBox_LVA_X, self.UI.spinBox_LVA_Y, self.UI.spinBox_LVA_Z,
             self.UI.horizontalSlider_LVA_X, self.UI.horizontalSlider_LVA_Y, self.UI.horizontalSlider_LVA_Z,
             self.__point_LVA)
        func(self.UI.spinBox_RVA_X, self.UI.spinBox_RVA_Y, self.UI.spinBox_RVA_Z,
             self.UI.horizontalSlider_RVA_X, self.UI.horizontalSlider_RVA_Y, self.UI.horizontalSlider_RVA_Z,
             self.__point_RVA)
        func(self.UI.spinBox_AV_X, self.UI.spinBox_AV_Y, self.UI.spinBox_AV_Z,
             self.UI.horizontalSlider_AV_X, self.UI.horizontalSlider_AV_Y, self.UI.horizontalSlider_AV_Z,
             self.__point_AV)
        func(self.UI.spinBox_PV_X, self.UI.spinBox_PV_Y, self.UI.spinBox_PV_Z,
             self.UI.horizontalSlider_PV_X, self.UI.horizontalSlider_PV_Y, self.UI.horizontalSlider_PV_Z,
             self.__point_PV)
        func(self.UI.spinBox_MV_X, self.UI.spinBox_MV_Y, self.UI.spinBox_MV_Z,
             self.UI.horizontalSlider_MV_X, self.UI.horizontalSlider_MV_Y, self.UI.horizontalSlider_MV_Z,
             self.__point_MV)
        func(self.UI.spinBox_TV_X, self.UI.spinBox_TV_Y, self.UI.spinBox_TV_Z,
             self.UI.horizontalSlider_TV_X, self.UI.horizontalSlider_TV_Y, self.UI.horizontalSlider_TV_Z,
             self.__point_TV)

    # 一组控件初始化
    def __initParamWidget(self, widget, maxV, minV, value, step):
        # 设置初值时阻断信号
        widget.blockSignals(True)
        # print(maxV, minV, value, step)
        widget.setMaximum(maxV)
        widget.setMinimum(minV)
        widget.setSingleStep(step)
        widget.setValue(value)
        widget.blockSignals(False)

    # 改变一个窗体的值
    def __setParamWidgetValue(self, widget, value):
        # 设置初值时阻断信号
        widget.blockSignals(True)
        widget.setValue(value)
        widget.blockSignals(False)


    # 渲染心脏掩码图像
    def __drawHeartMark(self):
        self.widget_vtk.drawHeartMark(self.__heartMark, self.__spacing,
                                      self.__spinCenter_vtk)



    #-------------槽函数-------------#
    #----数据导入----#
    def on_importImage(self):
        # filePath, _ = QFileDialog.getOpenFileName(self, "选取image文件",
        #                                           r'..\dataset',
        #                                           "npy Files(image_*.npy)")
        filePath = r'D:\坚果云\我的坚果云\实验：心脏定位\实验代码\20220703-数据处理\save\image_1019.npy'
        print(filePath)
        self.UI.lineEdit_imagePath.setText(filePath)
        if filePath == '':
            return
        self.__heart = np.load(filePath).astype(np.float32)


    def on_importLabel(self):
        # filePath, _ = QFileDialog.getOpenFileName(self, "选取image文件",
        #                                           r'..\dataset',
        #                                           "npy Files(label_*.npy)")
        filePath = r'D:\坚果云\我的坚果云\实验：心脏定位\实验代码\20220703-数据处理\save\label_1019.npy'
        print(filePath)
        self.UI.lineEdit_labelPath.setText(filePath)
        if filePath == '':
            return
        self.__heartMark = np.load(filePath)
        self.__heartMark[self.__heartMark==421] = 420
        # 这里因为是没有resample过的，所以这里范围并不准
        print(np.unique(self.__heartMark))
        coords = np.where(self.__heartMark > 0)
        min_x, max_x = coords[2].min(), coords[2].max()
        min_y, max_y = coords[1].min(), coords[1].max()
        min_z, max_z = coords[0].min(), coords[0].max()
        range_ = (max_x, min_x, max_y, min_y, max_z, min_z)
        self.__initParam(range_, (1, 1, 1))
        # vtk绘图
        self.__drawHeartMark()

    #----数据展示和处理----#

    def on_display_SAS(self):
        # The short-axis stack is orthogonal to the axes defined by the LVA and MV
        # 左心底和二尖瓣的连线的垂面，这里取中垂面
        point_LVA = self.__point_LVA[::-1]
        point_MV = self.__point_MV[::-1]
        center = (point_LVA + point_MV) / 2

        # 计算3点构成的平面的法向量
        normal = point_MV - point_LVA
        normal = normal / np.linalg.norm(normal)
        plane_X_norm = np.array([-normal[2] / normal[0], 0, 1])
        plane_X_norm = plane_X_norm / np.linalg.norm(plane_X_norm)
        plane_Y_norm = np.cross(plane_X_norm, normal)
        plane_Y_norm = plane_Y_norm / np.linalg.norm(plane_Y_norm)

        origin = center - self.plane_len / 2 * (plane_X_norm + plane_Y_norm)

        pic = resample_pic_from_volume(self.__heart, origin,
                                       plane_X_norm, plane_Y_norm, self.plane_len)
        pic = Image.fromarray(pic).toqpixmap() \
            .scaled(self.UI.label_pictrue.width(), self.UI.label_pictrue.height())
        self.UI.label_pictrue.setPixmap(pic)


        # 画平面 记得取反
        point1 = origin+self.plane_len*plane_X_norm
        point2 = origin+self.plane_len*plane_Y_norm
        self.widget_vtk.drawPlane(origin[::-1] + self.__spinCenter_vtk,
                                  point1[::-1] + self.__spinCenter_vtk,
                                  point2[::-1] + self.__spinCenter_vtk)
        # self.widget_vtk.drawPoints([self.__point_LVA + self.__spinCenter_vtk,
        #                             self.__point_MV + self.__spinCenter_vtk])

    def on_display_VLA(self):
        # The 2-chamber plane bisects the angle of the 3- and 4- chamber planes
        # 2腔心平分3腔心和4腔心（是一个找平分面的操作）
        # point_LVA = self.__point_LVA[::-1]
        # point_MV = self.__point_MV[::-1]
        # point_AV = self.__point_AV[::-1]
        # point_TV = self.__point_TV[::-1]
        #
        # point_aux = (point_AV+point_TV)/2
        # center = (point_LVA + point_MV + point_aux) / 3

        # 感觉是垂直的
        # point_LVA = self.__point_LVA[::-1]
        # point_MV = self.__point_MV[::-1]
        # point_TV = self.__point_TV[::-1]
        # # todo 还是取心脏截面的重心作为 中心比较好
        # center = (point_LVA + point_MV) / 2
        #
        # # 计算3点构成的平面的法向量
        # vec_aux = np.cross(point_MV - point_LVA, point_TV - point_LVA)
        # normal = np.cross(vec_aux, point_LVA - point_MV)

        point_LVA = self.__point_LVA[::-1]
        point_MV = self.__point_MV[::-1]
        point_AV = self.__point_AV[::-1]
        point_TV = self.__point_TV[::-1]

        point_aux = (point_AV + point_TV) / 2
        vec_aux = np.cross(point_LVA - point_aux, point_LVA - point_MV)  # 平分面法向量
        normal = np.cross(vec_aux, point_LVA - point_MV)

        center = (point_LVA + point_MV) / 2
        normal = normal / np.linalg.norm(normal)
        plane_X_norm = np.array([-normal[2] / normal[0], 0, 1])
        plane_X_norm = plane_X_norm / np.linalg.norm(plane_X_norm)
        plane_Y_norm = np.cross(plane_X_norm, normal)
        plane_Y_norm = plane_Y_norm / np.linalg.norm(plane_Y_norm)

        origin = center - self.plane_len / 2 * (plane_X_norm + plane_Y_norm)

        # 重采样
        pic = resample_pic_from_volume(self.__heart, origin,
                                       plane_X_norm, plane_Y_norm, self.plane_len)
        pic = Image.fromarray(pic).toqpixmap() \
            .scaled(self.UI.label_pictrue.width(), self.UI.label_pictrue.height())
        self.UI.label_pictrue.setPixmap(pic)


        # 画平面 记得取反
        point1 = origin+self.plane_len*plane_X_norm
        point2 = origin+self.plane_len*plane_Y_norm
        self.widget_vtk.drawPlane(origin[::-1] + self.__spinCenter_vtk,
                                  point1[::-1] + self.__spinCenter_vtk,
                                  point2[::-1] + self.__spinCenter_vtk)
        self.widget_vtk.drawPoints([self.__point_LVA + self.__spinCenter_vtk,
                                    self.__point_MV + self.__spinCenter_vtk])


    def on_display_LVOT(self):
        # The 3-chamber plane goes through the AV, MV, and LVA.
        # 3腔心平面通过主动脉瓣、二尖瓣、左心底
        # 这里的坐标需要反一下，因为该类里保存的是（x, y, z），而用来计算图片的数组是（z, y, x）的
        # 平面的边长
        point_LVA = self.__point_LVA[::-1]
        point_MV = self.__point_MV[::-1]
        point_AV = self.__point_AV[::-1]
        center = (point_LVA + point_MV + point_AV) / 3

        # 计算3点构成的平面的法向量
        normal = np.cross(point_MV - point_LVA, point_AV - point_LVA)
        normal = normal / np.linalg.norm(normal)
        plane_X_norm = np.array([-normal[2] / normal[0], 0, 1])
        plane_X_norm = plane_X_norm / np.linalg.norm(plane_X_norm)
        plane_Y_norm = np.cross(plane_X_norm, normal)
        plane_Y_norm = plane_Y_norm / np.linalg.norm(plane_Y_norm)

        origin = center - self.plane_len / 2 * (plane_X_norm + plane_Y_norm)

        pic = resample_pic_from_volume(self.__heart, origin,
                                       plane_X_norm, plane_Y_norm, self.plane_len)
        pic = Image.fromarray(pic).toqpixmap() \
            .scaled(self.UI.label_pictrue.width(), self.UI.label_pictrue.height())
        self.UI.label_pictrue.setPixmap(pic)

        # 画平面 记得取反
        point1 = origin+self.plane_len*plane_X_norm
        point2 = origin+self.plane_len*plane_Y_norm
        self.widget_vtk.drawPlane(origin[::-1] + self.__spinCenter_vtk,
                                  point1[::-1] + self.__spinCenter_vtk,
                                  point2[::-1] + self.__spinCenter_vtk)
        self.widget_vtk.drawPoints([self.__point_LVA + self.__spinCenter_vtk,
                                    self.__point_MV + self.__spinCenter_vtk,
                                    self.__point_AV + self.__spinCenter_vtk])

    def on_display_4CH(self):
        # the 4-chamber plane goes through the TV, MV, and LVA
        # 4腔心平面通过三尖瓣、二尖瓣、左心底
        # (x, y, z) -> (z, y, x) 因为使用numpy算的
        point_LVA = self.__point_LVA[::-1]
        point_MV = self.__point_MV[::-1]
        point_TV = self.__point_TV[::-1]
        center = (point_LVA + point_MV + point_TV) / 3

        # 计算3点构成的平面的法向量
        normal = np.cross(point_MV - point_LVA, point_TV - point_LVA)
        normal = normal / np.linalg.norm(normal)
        plane_X_norm = np.array([-normal[2] / normal[0], 0, 1])
        plane_X_norm = plane_X_norm / np.linalg.norm(plane_X_norm)
        plane_Y_norm = np.cross(plane_X_norm, normal)
        plane_Y_norm = plane_Y_norm / np.linalg.norm(plane_Y_norm)

        origin = center - self.plane_len / 2 * (plane_X_norm + plane_Y_norm)

        pic = resample_pic_from_volume(self.__heart, origin,
                                       plane_X_norm, plane_Y_norm, self.plane_len)
        pic = Image.fromarray(pic).toqpixmap()\
            .scaled(self.UI.label_pictrue.width(), self.UI.label_pictrue.height())
        self.UI.label_pictrue.setPixmap(pic)


        # 画平面 记得取反
        point1 = origin+self.plane_len*plane_X_norm
        point2 = origin+self.plane_len*plane_Y_norm
        self.widget_vtk.drawPlane(origin[::-1] + self.__spinCenter_vtk,
                                  point1[::-1] + self.__spinCenter_vtk,
                                  point2[::-1] + self.__spinCenter_vtk)
        self.widget_vtk.drawPoints([self.__point_LVA + self.__spinCenter_vtk,
                                    self.__point_MV + self.__spinCenter_vtk,
                                    self.__point_TV + self.__spinCenter_vtk])

    def on_display_RVOT(self):
        # the RVOT plane goes through the TV, PV, and RVA
        # RVOT平面通过三尖瓣、肺动脉瓣、右心底
        # (x, y, z) -> (z, y, x) 因为使用numpy算的
        point_RVA = self.__point_RVA[::-1]
        point_PV = self.__point_PV[::-1]
        point_TV = self.__point_TV[::-1]
        center = (point_RVA + point_PV + point_TV) / 3

        # 计算3点构成的平面的法向量
        normal = np.cross(point_PV - point_RVA, point_TV - point_RVA)
        normal = normal / np.linalg.norm(normal)
        plane_X_norm = np.array([-normal[2] / normal[0], 0, 1])
        plane_X_norm = plane_X_norm / np.linalg.norm(plane_X_norm)
        plane_Y_norm = np.cross(plane_X_norm, normal)
        plane_Y_norm = plane_Y_norm / np.linalg.norm(plane_Y_norm)

        origin = center - self.plane_len / 2 * (plane_X_norm + plane_Y_norm)

        pic = resample_pic_from_volume(self.__heart, origin,
                                       plane_X_norm, plane_Y_norm, self.plane_len)
        pic = Image.fromarray(pic).toqpixmap() \
            .scaled(self.UI.label_pictrue.width(), self.UI.label_pictrue.height())
        self.UI.label_pictrue.setPixmap(pic)

    def on_saveCoords(self):
        saved_data = np.array([
            self.__point_LVA,
            self.__point_RVA,
            self.__point_AV,
            self.__point_MV,
            self.__point_TV,
            self.__point_PV,
        ])
        saveFile = r'D:\坚果云\我的坚果云\实验：心脏定位\论文代码\20221008-数据处理\landmarks'
        idx = self.UI.lineEdit_labelPath.text()[-8: -4]
        np.save(os.path.join(saveFile, 'landmark_' + idx), saved_data)
        print('landmark info: ', saved_data)
        QMessageBox.about(self, '提示', '保存成功！')

    #----定位面操作部分----#
    # vtk1 旋转中心移动
    def on_change_spinCenterX_vtk(self, value):
        self.__spinCenter_vtk[0] = value
        self.__drawHeartMark()
        # self.__renderPoints()

    def on_change_spinCenterY_vtk(self, value):
        self.__spinCenter_vtk[1] = value
        self.__drawHeartMark()
        # self.__renderPoints()

    def on_change_spinCenterZ_vtk(self, value):
        self.__spinCenter_vtk[2] = value
        self.__drawHeartMark()
        # self.__renderPoints()

    # 左心室心底
    def on_change_LVA_X(self, value):
        self.__point_LVA[0] = value
        self.__setParamWidgetValue(self.UI.spinBox_LVA_X, value)
        self.__setParamWidgetValue(self.UI.horizontalSlider_LVA_X, value)
        point_LVA = self.__point_LVA + self.__spinCenter_vtk
        self.widget_vtk.drawPoint_LVA(point_LVA)

    def on_change_LVA_Y(self, value):
        self.__point_LVA[1] = value
        self.__setParamWidgetValue(self.UI.spinBox_LVA_Y, value)
        self.__setParamWidgetValue(self.UI.horizontalSlider_LVA_Y, value)
        point_LVA = self.__point_LVA + self.__spinCenter_vtk
        self.widget_vtk.drawPoint_LVA(point_LVA)

    def on_change_LVA_Z(self, value):
        self.__point_LVA[2] = value
        self.__setParamWidgetValue(self.UI.spinBox_LVA_Z, value)
        self.__setParamWidgetValue(self.UI.horizontalSlider_LVA_Z, value)
        point_LVA = self.__point_LVA + self.__spinCenter_vtk
        self.widget_vtk.drawPoint_LVA(point_LVA)

    # 右动脉瓣
    def on_change_RVA_X(self, value):
        self.__point_RVA[0] = value
        self.__setParamWidgetValue(self.UI.spinBox_RVA_X, value)
        self.__setParamWidgetValue(self.UI.horizontalSlider_RVA_X, value)
        point_RVA = self.__point_RVA + self.__spinCenter_vtk
        self.widget_vtk.drawPoint_RVA(point_RVA)

    def on_change_RVA_Y(self, value):
        self.__point_RVA[1] = value
        self.__setParamWidgetValue(self.UI.spinBox_RVA_Y, value)
        self.__setParamWidgetValue(self.UI.horizontalSlider_RVA_Y, value)
        point_RVA = self.__point_RVA + self.__spinCenter_vtk
        self.widget_vtk.drawPoint_RVA(point_RVA)

    def on_change_RVA_Z(self, value):
        self.__point_RVA[2] = value
        self.__setParamWidgetValue(self.UI.spinBox_RVA_Z, value)
        self.__setParamWidgetValue(self.UI.horizontalSlider_RVA_Z, value)
        point_RVA = self.__point_RVA + self.__spinCenter_vtk
        self.widget_vtk.drawPoint_RVA(point_RVA)

    # 主动脉瓣
    def on_change_AV_X(self, value):
        self.__point_AV[0] = value
        self.__setParamWidgetValue(self.UI.spinBox_AV_X, value)
        self.__setParamWidgetValue(self.UI.horizontalSlider_AV_X, value)
        point_AV = self.__point_AV + self.__spinCenter_vtk
        self.widget_vtk.drawPoint_AV(point_AV)

    def on_change_AV_Y(self, value):
        self.__point_AV[1] = value
        self.__setParamWidgetValue(self.UI.spinBox_AV_Y, value)
        self.__setParamWidgetValue(self.UI.horizontalSlider_AV_Y, value)
        point_AV = self.__point_AV + self.__spinCenter_vtk
        self.widget_vtk.drawPoint_AV(point_AV)

    def on_change_AV_Z(self, value):
        self.__point_AV[2] = value
        self.__setParamWidgetValue(self.UI.spinBox_AV_Z, value)
        self.__setParamWidgetValue(self.UI.horizontalSlider_AV_Z, value)
        point_AV = self.__point_AV + self.__spinCenter_vtk
        self.widget_vtk.drawPoint_AV(point_AV)

    # 肺动脉瓣
    def on_change_PV_X(self, value):
        self.__point_PV[0] = value
        self.__setParamWidgetValue(self.UI.spinBox_PV_X, value)
        self.__setParamWidgetValue(self.UI.horizontalSlider_PV_X, value)
        point_PV = self.__point_PV + self.__spinCenter_vtk
        self.widget_vtk.drawPoint_PV(point_PV)

    def on_change_PV_Y(self, value):
        self.__point_PV[1] = value
        self.__setParamWidgetValue(self.UI.spinBox_PV_Y, value)
        self.__setParamWidgetValue(self.UI.horizontalSlider_PV_Y, value)
        point_PV = self.__point_PV + self.__spinCenter_vtk
        self.widget_vtk.drawPoint_PV(point_PV)

    def on_change_PV_Z(self, value):
        self.__point_PV[2] = value
        self.__setParamWidgetValue(self.UI.spinBox_PV_Z, value)
        self.__setParamWidgetValue(self.UI.horizontalSlider_PV_Z, value)
        point_PV = self.__point_PV + self.__spinCenter_vtk
        self.widget_vtk.drawPoint_PV(point_PV)

    # 二尖瓣
    def on_change_MV_X(self, value):
        self.__point_MV[0] = value
        self.__setParamWidgetValue(self.UI.spinBox_MV_X, value)
        self.__setParamWidgetValue(self.UI.horizontalSlider_MV_X, value)
        point_MV = self.__point_MV + self.__spinCenter_vtk
        self.widget_vtk.drawPoint_MV(point_MV)

    def on_change_MV_Y(self, value):
        self.__point_MV[1] = value
        self.__setParamWidgetValue(self.UI.spinBox_MV_Y, value)
        self.__setParamWidgetValue(self.UI.horizontalSlider_MV_Y, value)
        point_MV = self.__point_MV + self.__spinCenter_vtk
        self.widget_vtk.drawPoint_MV(point_MV)

    def on_change_MV_Z(self, value):
        self.__point_MV[2] = value
        self.__setParamWidgetValue(self.UI.spinBox_MV_Z, value)
        self.__setParamWidgetValue(self.UI.horizontalSlider_MV_Z, value)
        point_MV = self.__point_MV + self.__spinCenter_vtk
        self.widget_vtk.drawPoint_MV(point_MV)

    # 三尖瓣
    def on_change_TV_X(self, value):
        self.__point_TV[0] = value
        self.__setParamWidgetValue(self.UI.spinBox_TV_X, value)
        self.__setParamWidgetValue(self.UI.horizontalSlider_TV_X, value)
        point_TV = self.__point_TV + self.__spinCenter_vtk
        self.widget_vtk.drawPoint_TV(point_TV)

    def on_change_TV_Y(self, value):
        self.__point_TV[1] = value
        self.__setParamWidgetValue(self.UI.spinBox_TV_Y, value)
        self.__setParamWidgetValue(self.UI.horizontalSlider_TV_Y, value)
        point_TV = self.__point_TV + self.__spinCenter_vtk
        self.widget_vtk.drawPoint_TV(point_TV)

    def on_change_TV_Z(self, value):
        self.__point_TV[2] = value
        self.__setParamWidgetValue(self.UI.spinBox_TV_Z, value)
        self.__setParamWidgetValue(self.UI.horizontalSlider_TV_Z, value)
        point_TV = self.__point_TV + self.__spinCenter_vtk
        self.widget_vtk.drawPoint_TV(point_TV)

    def __renderPoints(self):
        point_LVA = self.__point_LVA + self.__spinCenter_vtk
        self.widget_vtk.drawPoint_LVA(point_LVA)
        point_RVA = self.__point_RVA + self.__spinCenter_vtk
        self.widget_vtk.drawPoint_RVA(point_RVA)
        point_AV = self.__point_AV + self.__spinCenter_vtk
        self.widget_vtk.drawPoint_AV(point_AV)
        point_PV = self.__point_PV + self.__spinCenter_vtk
        self.widget_vtk.drawPoint_PV(point_PV)
        point_MV = self.__point_MV + self.__spinCenter_vtk
        self.widget_vtk.drawPoint_MV(point_MV)
        point_TV = self.__point_TV + self.__spinCenter_vtk
        self.widget_vtk.drawPoint_TV(point_TV)


if __name__ == '__main__':
    # 创建应用程序和对象
    app = QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())