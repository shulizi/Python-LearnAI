# -*- coding: utf-8 -*-
import wx
import numpy as np
import matplotlib
import roc_auc
import log_regres

# matplotlib采用WXAgg为后台,将matplotlib嵌入wxPython中
matplotlib.use("WXAgg")

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
from matplotlib.ticker import MultipleLocator

import pylab
from matplotlib import pyplot


class MPL_Panel_base(wx.Panel):
    ''''' #MPL_Panel_base面板,可以继承或者创建实例'''

    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent, id=-1)

        self.Figure = matplotlib.figure.Figure(figsize=(4, 3))
        self.axes = self.Figure.add_axes([0.1, 0.1, 0.8, 0.8])
        self.FigureCanvas = FigureCanvas(self, -1, self.Figure)

        self.NavigationToolbar = NavigationToolbar(self.FigureCanvas)

        self.StaticText = wx.StaticText(self, -1, label='')

        self.SubBoxSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.SubBoxSizer.Add(self.NavigationToolbar, proportion=0, border=2, flag=wx.ALL | wx.EXPAND)
        self.SubBoxSizer.Add(self.StaticText, proportion=-1, border=2, flag=wx.ALL | wx.EXPAND)

        self.TopBoxSizer = wx.BoxSizer(wx.VERTICAL)
        self.TopBoxSizer.Add(self.SubBoxSizer, proportion=-1, border=2, flag=wx.ALL | wx.EXPAND)
        self.TopBoxSizer.Add(self.FigureCanvas, proportion=-10, border=2, flag=wx.ALL | wx.EXPAND)

        self.SetSizer(self.TopBoxSizer)


    def UpdatePlot(self):
        '''''#修改图形的任何属性后都必须使用self.UpdatePlot()更新GUI界面 '''
        self.FigureCanvas.draw()

    def plot(self, *args, **kwargs):
        '''''#最常用的绘图命令plot '''
        self.axes.plot(*args, **kwargs)
        self.UpdatePlot()

    def grid(self, flag=True):
        ''''' ##显示网格  '''
        if flag:
            self.axes.grid()
        else:
            self.axes.grid(False)

    def title_MPL(self, TitleString="wxMatPlotLib Example In wxPython"):
        ''''' # 给图像添加一个标题   '''
        self.axes.set_title(TitleString)

    def xlabel(self, XabelString="X"):
        ''''' # Add xlabel to the plotting    '''
        self.axes.set_xlabel(XabelString)

    def ylabel(self, YabelString="Y"):
        ''''' # Add ylabel to the plotting '''
        self.axes.set_ylabel(YabelString)

    def xticker(self, major_ticker=1.0, minor_ticker=0.1):
        ''''' # 设置X轴的刻度大小 '''
        self.axes.xaxis.set_major_locator(MultipleLocator(major_ticker))
        self.axes.xaxis.set_minor_locator(MultipleLocator(minor_ticker))

    def yticker(self, major_ticker=1.0, minor_ticker=0.1):
        ''''' # 设置Y轴的刻度大小 '''
        self.axes.yaxis.set_major_locator(MultipleLocator(major_ticker))
        self.axes.yaxis.set_minor_locator(MultipleLocator(minor_ticker))

    def savefig(self, *args, **kwargs):
        ''' #保存图形到文件 '''
        self.Figure.savefig(*args, **kwargs)

    def clean(self):
        ''' # 再次画图前,必须调用该命令清空原来的图形  '''
        self.axes.clear()
        self.Figure.set_canvas(self.FigureCanvas)
        self.UpdatePlot()

    def ShowHelpString(self, HelpString="Show Help String"):
        ''''' #可以用它来显示一些帮助信息,如鼠标位置等 '''
        self.StaticText.SetLabel(HelpString)

class MPL2_Frame(wx.Frame):
    """MPL2_Frame可以继承,并可修改,或者直接使用"""

    def __init__(self, title="Logistic Regression", size=(1000, 500)):
        wx.Frame.__init__(self, parent=None, title=title, size=size)

        self.BoxSizer = wx.BoxSizer(wx.HORIZONTAL)

        self.MPL1 = MPL_Panel_base(self)
        self.BoxSizer.Add(self.MPL1, proportion=-1, border=2, flag=wx.ALL | wx.EXPAND)

        self.MPL2 = MPL_Panel_base(self)
        self.BoxSizer.Add(self.MPL2, proportion=-1, border=2, flag=wx.ALL | wx.EXPAND)

        self.RightPanel = wx.Panel(self, -1)
        self.BoxSizer.Add(self.RightPanel, proportion=0, border=2, flag=wx.ALL | wx.EXPAND)

        self.SetSizer(self.BoxSizer)

        # 创建FlexGridSizer
        self.FlexGridSizer = wx.FlexGridSizer(rows=9, cols=1, vgap=5, hgap=5)
        self.FlexGridSizer.SetFlexibleDirection(wx.BOTH)


        self.Button1 = wx.Button(self.RightPanel, -1, "Probability\nDensity", size=(100, 40), pos=(10, 10))
        self.Button1.Bind(wx.EVT_BUTTON, self.Button1Event)


        self.Button2 = wx.Button(self.RightPanel, -1, "ROC", size=(100, 40), pos=(10, 10))
        self.Button2.Bind(wx.EVT_BUTTON, self.Button2Event)
        

        self.Button3 = wx.Button(self.RightPanel, -1, "Data", size=(100, 40), pos=(10, 10))
        self.Button3.Bind(wx.EVT_BUTTON, self.Button3Event)


        # 加入Sizer中
        self.FlexGridSizer.Add(self.Button1, proportion=0, border=5, flag=wx.ALL | wx.EXPAND)
        self.FlexGridSizer.Add(self.Button2, proportion=0, border=5, flag=wx.ALL | wx.EXPAND)
        self.FlexGridSizer.Add(self.Button3, proportion=0, border=5, flag=wx.ALL | wx.EXPAND)
 
        
        self.RightPanel.SetSizer(self.FlexGridSizer)

        # 状态栏
        self.StatusBar()

        # MPL2_Frame界面居中显示
        self.Centre(wx.BOTH)



        # 按钮事件,用于测试

    def Button1Event(self, event):
        
        self.MPL1.clean()  # 必须清理图形,才能显示下一幅图
        x1,y1,x2,y2,decisionx,decisiony = log_regres.get_x_y()
        self.MPL1.plot(x1, y1, '*g')
        self.MPL1.plot(x2, y2, '*r')
        self.MPL1.plot(decisionx, decisiony, 'y')
        self.MPL1.ShowHelpString('k=%s'%(decisiony[0]/decisionx[0]))
        self.MPL1.xticker(5, 1)
        self.MPL1.yticker(1, 0.5)
        self.MPL1.title_MPL("Probability Density")
        self.MPL1.grid()
        self.MPL1.UpdatePlot()  # 必须刷新才能显示

        

        

    def Button2Event(self, event):
        auc=roc_auc.auc_calculate()
        self.MPL2.clean()
        x,y = roc_auc.get_x_y()
        self.MPL2.plot(x, y, ':^b')
        self.MPL2.ShowHelpString('AUC={:.6f}'.format(auc))
        self.MPL2.xticker(0.2, 0.02)
        self.MPL2.yticker(0.2, 0.02)
        self.MPL2.title_MPL("ROC")
        self.MPL2.grid()
        self.MPL2.UpdatePlot()

    def Button3Event(self, event):
        f = open('txt_set.txt','r')
        data=''
        for i in f.readlines():
            data=data+'（'+str(i).strip() +'）'
        dlg = wx.MessageDialog(self,data,'Data', wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()

        # 自动创建状态栏

    def StatusBar(self):
        self.statusbar = self.CreateStatusBar()
        self.statusbar.SetFieldsCount(3)
        self.statusbar.SetStatusWidths([-2, -2, -1])


if __name__ == '__main__':
    app = wx.App()
    frame = MPL2_Frame()
    frame.Center()
    frame.Show()
    app.MainLoop()
