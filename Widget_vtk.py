import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import vtk
from vtk.util.vtkImageImportFromArray import vtkImageImportFromArray
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import numpy as np
import SimpleITK as sitk


class widget_vtk(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(600, 600)
        self.gridlayout = QGridLayout(self)
        self.gridlayout.setContentsMargins(0, 0, 0, 0)
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.vtkWidget.resize(600, 600)

        self.gridlayout.addWidget(self.vtkWidget, 0, 0, 1, 1)
        self.rendererWin = self.vtkWidget.GetRenderWindow()

        self.interactor = self.rendererWin.GetInteractor()  # 获取渲染窗口的
        self.interactor.Initialize()
        self.interactor.Start()

        self.renderer = vtk.vtkRenderer()
        self.rendererWin.AddRenderer(self.renderer)  # 把渲染器添加窗口上

        # 背景颜色
        colors = vtk.vtkNamedColors()
        colors.SetColor('BkgColor', [126, 151, 177, 255])
        self.renderer.SetBackground(colors.GetColor3d('BkgColor'))

        # 私有变量
        self.volume_heartMark = None  # 心脏掩码
        self.actor_outline = None
        self.actor_point_LVA = None
        self.actor_point_RVA = None
        self.actor_point_AV = None
        self.actor_point_TV = None
        self.actor_point_MV = None
        self.actor_point_PV = None

        self.planes = {

        }
        self.actor_SAS_plane = None
        self.actor_VLA_plane = None
        self.actor_LVOT_plane = None
        self.actor_4CH_plane = None
        self.actor_RVOT_plane = None
        self.actor_plane = None

    # 添加 坐标轴
    def drawAxis(self):
        axesActor = vtk.vtkAxesActor()
        self.axesWidget = vtk.vtkOrientationMarkerWidget()
        self.axesWidget.SetOrientationMarker(axesActor)
        self.axesWidget.SetInteractor(self.interactor)
        self.axesWidget.On()
        self.axesWidget.SetInteractive(0)

    def drawHeartMark(self, heartMark, spacing, spinCenter):
        vtk_arr = vtkImageImportFromArray()  # 创建一个空的vtk类-----vtkImageImportFromArray
        vtk_arr.SetArray(heartMark)  # 把arr塞到vtkImageImportFromArray（arr）
        vtk_arr.SetDataSpacing(tuple(spacing))  # 设置spacing

        # 坐标偏移
        vtk_arr.SetDataOrigin(tuple(spinCenter))
        vtk_arr.Update()

        # 不透明度传输函数---放在tfun
        tfun = vtk.vtkPiecewiseFunction()  # 不透明度传输函数---放在tfun
        tfun.AddPoint(0, 0)

        for value in [205, 420, 500, 550, 600, 820, 850]:
            tfun.AddPoint(value, 0.06)

        # 颜色传输函数---放在ctfun
        colors = np.load('colors.npy')
        colors = colors/255 * 2
        # 颜色传输函数---放在ctfun
        ctfun = vtk.vtkColorTransferFunction()  # 颜色传输函数---放在ctfun
        for value, color in zip([205, 420, 500, 550, 600, 820, 850], colors):
            ctfun.AddRGBPoint(value, color[0], color[1], color[2])

        # 绘制心脏
        volumeMapper = vtk.vtkGPUVolumeRayCastMapper()  # 映射器volumnMapper使用vtk的管线投影算法
        volumeMapper.SetInputData(vtk_arr.GetOutput())  # 向映射器中输入数据：shifter(预处理之后的数据)
        volumeProperty = vtk.vtkVolumeProperty()  # 创建vtk属性存放器,向属性存放器中存放颜色和透明度
        volumeProperty.SetColor(ctfun)
        volumeProperty.SetScalarOpacity(tfun)
        volumeProperty.SetInterpolationTypeToLinear()  # ???
        volumeProperty.ShadeOn()

        # 创建演员
        volume = vtk.vtkVolume()  # 演员
        volume.SetMapper(volumeMapper)
        volume.SetProperty(volumeProperty)

        self.renderer.RemoveVolume(self.volume_heartMark)
        self.volume_heartMark = volume
        self.renderer.AddVolume(self.volume_heartMark)

        # 绘制轮廓
        # outline = vtk.vtkOutlineFilter()
        # outline.SetInputConnection(vtk_arr.GetOutputPort())
        #
        # outlineMapper = vtk.vtkPolyDataMapper()
        # outlineMapper.SetInputConnection(outline.GetOutputPort())
        #
        # outlineActor = vtk.vtkActor()
        # outlineActor.SetMapper(outlineMapper)
        #
        # self.renderer.RemoveActor(self.actor_outline)
        # self.actor_outline = outlineActor
        # self.renderer.AddActor(self.actor_outline)

        self.rendererWin.Render()

    # 心尖
    def drawPoint_LVA(self, center):
        sphereSource = vtk.vtkSphereSource()
        sphereSource.SetCenter(center)
        sphereSource.SetRadius(2.0)
        sphereSource.Update()

        # Create a mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphereSource.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        colors = vtk.vtkNamedColors()
        actor.GetProperty().SetColor(colors.GetColor3d("red"))

        self.renderer.RemoveActor(self.actor_point_LVA)
        self.actor_point_LVA = actor
        self.renderer.AddActor(self.actor_point_LVA)

        self.rendererWin.Render()

    def drawPoint_RVA(self, center):
        sphereSource = vtk.vtkSphereSource()
        sphereSource.SetCenter(center)
        sphereSource.SetRadius(2.0)
        sphereSource.Update()

        # Create a mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphereSource.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        colors = vtk.vtkNamedColors()
        actor.GetProperty().SetColor(colors.GetColor3d("orange"))

        self.renderer.RemoveActor(self.actor_point_RVA)
        self.actor_point_RVA = actor
        self.renderer.AddActor(self.actor_point_RVA)

        self.rendererWin.Render()

    def drawPoint_AV(self, center):
        sphereSource = vtk.vtkSphereSource()
        sphereSource.SetCenter(center)
        sphereSource.SetRadius(2.0)
        sphereSource.Update()

        # Create a mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphereSource.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        colors = vtk.vtkNamedColors()
        actor.GetProperty().SetColor(colors.GetColor3d("yellow"))

        self.renderer.RemoveActor(self.actor_point_AV)
        self.actor_point_AV = actor
        self.renderer.AddActor(self.actor_point_AV)

        self.rendererWin.Render()

    def drawPoint_PV(self, center):
        sphereSource = vtk.vtkSphereSource()
        sphereSource.SetCenter(center)
        sphereSource.SetRadius(2.0)
        sphereSource.Update()

        # Create a mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphereSource.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        colors = vtk.vtkNamedColors()
        actor.GetProperty().SetColor(colors.GetColor3d("green"))

        self.renderer.RemoveActor(self.actor_point_PV)
        self.actor_point_PV = actor
        self.renderer.AddActor(self.actor_point_PV)

        self.rendererWin.Render()

    def drawPoint_MV(self, center):
        sphereSource = vtk.vtkSphereSource()
        sphereSource.SetCenter(center)
        sphereSource.SetRadius(2.0)
        sphereSource.Update()

        # Create a mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphereSource.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        colors = vtk.vtkNamedColors()
        actor.GetProperty().SetColor(colors.GetColor3d("blue"))

        self.renderer.RemoveActor(self.actor_point_MV)
        self.actor_point_MV = actor
        self.renderer.AddActor(self.actor_point_MV)

        self.rendererWin.Render()

    def drawPoint_TV(self, center):
        sphereSource = vtk.vtkSphereSource()
        sphereSource.SetCenter(center)
        sphereSource.SetRadius(2.0)
        sphereSource.Update()

        # Create a mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphereSource.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        colors = vtk.vtkNamedColors()
        actor.GetProperty().SetColor(colors.GetColor3d("purple"))

        self.renderer.RemoveActor(self.actor_point_TV)
        self.actor_point_TV = actor
        self.renderer.AddActor(self.actor_point_TV)

        self.rendererWin.Render()

    def drawPlane(self, origin, point1, point2):
        # print('drawPlane', origin, point1, point2)
        planeSource = vtk.vtkPlaneSource()
        planeSource.SetOrigin(tuple(origin)) # 起始点
        planeSource.SetPoint1(tuple(point1)) # 三个点中的终点1
        planeSource.SetPoint2(tuple(point2)) # 三个点中的终点2
        planeSource.Update()

        # Create a mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(planeSource.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        colors = vtk.vtkNamedColors()
        actor.GetProperty().SetColor(colors.GetColor3d('gray'))
        actor.GetProperty().SetOpacity(0.7)

        self.renderer.RemoveActor(self.actor_plane)
        self.actor_plane = actor
        self.renderer.AddActor(self.actor_plane)

        self.rendererWin.Render()

    def drawPoints(self, points):
        for point in points:
            self.drawPoint(point)

    def drawPoint(self, center):
        sphereSource = vtk.vtkSphereSource()
        sphereSource.SetCenter(center)
        sphereSource.SetRadius(4.0)
        sphereSource.Update()

        # Create a mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphereSource.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        colors = vtk.vtkNamedColors()
        actor.GetProperty().SetColor(colors.GetColor3d("red"))

        self.renderer.AddActor(actor)

        self.rendererWin.Render()

    def drawOutline(self):
        self.renderer.AddActor(self.actor_outline)
        self.rendererWin.Render()

    def removeOutline(self):
        self.renderer.RemoveActor(self.actor_outline)
        self.rendererWin.Render()

