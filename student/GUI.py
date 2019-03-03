#!/usr/bin/python
# -*- coding: UTF-8 -*-
import wx
import wx.grid
import time
import MySQLdb
import Load
import Manage
import SelectClass
class GradeFrame(wx.Frame):
    def __init__(self,student):
        

        wx.Frame.__init__(self,parent=None,title="学生信息单",pos=(100,100),size=(500,500))
        
        self.panel=wx.Panel(self)
        studentMessage = student.getStudentMessage()
        
        wx.StaticText(self.panel,-1,studentMessage[0],pos=(50,10))
        wx.StaticText(self.panel,-1,studentMessage[1],pos=(100,10))
        
        gradeLabel=wx.StaticText(self.panel,-1,"学生成绩单",pos=(200,10))
        gradeLabel.SetFont(wx.Font(18, wx.ROMAN, wx.ITALIC, wx.NORMAL))

        wx.StaticText(self.panel,-1,"================================================",pos=(50,30))
        t = str(time.localtime(time.time()).tm_year)+\
            '.'+str(time.localtime(time.time()).tm_mon)+\
            '.'+str(time.localtime(time.time()).tm_mday)

        

        
        self.gradeLabel=wx.StaticText(self.panel,-1,t,pos=(400,10))
        grid = wx.grid.Grid(self.panel,id=-1,pos=(50,50),size=(400,200))
       
        grid.CreateGrid(10, 5)
        grid.HideRowLabels()
        grid.HideColLabels()
        grid.SetColSize(0, 69)
        grid.SetColSize(1, 100)
        grid.SetColSize(2, 69)
        grid.SetGridLineColour(wx.Colour(230,230,230))
        grid.SetDefaultCellBackgroundColour(wx.Colour(230,230,230))
        grid.EnableEditing(False)
        classesGrade = student.getCompleteStudentGrade()
        grid.SetCellValue(0, 0, '课程号')
        grid.SetCellValue(0, 1, '课程名')
        grid.SetCellValue(0, 2, '成绩')
        grid.SetCellValue(0, 3, '学分')
        grid.SetCellValue(0, 4, '教师')
        avgGrade = 0
        for i in range(len(classesGrade)):
            for j in range(len(classesGrade[i])):
                grid.SetCellValue(i+1, j, classesGrade[i][j])
            avgGrade += int(classesGrade[i][2])    
        avgLabel = wx.StaticText(self.panel,-1,"平均成绩："+str(avgGrade/len(classesGrade)),pos=(200,300))
        avgLabel.SetFont(wx.Font(30, wx.ROMAN, wx.ITALIC, wx.NORMAL))
    def onCloseMenuSelect(self,event):
        answer = wx.MessageBox("确认退出?", "确认",wx.YES_NO | wx.CANCEL, self)
        if answer == wx.YES:
            self.Close()
class StudentsFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self,parent=None,title="学生信息",pos=(100,100),size=(550,350))

        self.panel=wx.Panel(self)
        
        self.studentsNumberLabel=wx.StaticText(self.panel,-1,"记录总数：",pos=(20,10))
        self.studentsNumberLabel.SetFont(wx.Font(18, wx.ROMAN, wx.ITALIC, wx.NORMAL))
        
        self.manage = Manage.Manage()
        rowLength = len(self.manage.getStudentsMessage())
        
        self.grid = wx.grid.Grid(self.panel,id=-1,pos=(20,80),size=(410,220))
        self.grid.CreateGrid(rowLength, 7)
        self.grid.SetRowLabelSize(30)
        self.grid.SetColLabelValue(0,'学号')
        self.grid.SetColSize(0, 40)
        self.grid.SetColLabelValue(1,'姓名')
        self.grid.SetColLabelValue(2,'性别')
        self.grid.SetColSize(2, 40)
        self.grid.SetColLabelValue(3,'年龄')
        self.grid.SetColSize(3, 40)
        self.grid.SetColLabelValue(4,'所在系')
        self.grid.SetColSize(4, 100)
        self.grid.SetColLabelValue(5,'登录名')
        self.grid.SetColSize(5, 40)
        self.grid.SetColLabelValue(6,'密码')
        self.grid.SetColSize(6, 40)
        self.grid.EnableEditing(False)
        self.setGridReadOnly()
        self.updateStudents()
        

        self.studentsNumberLabel.SetLabel(self.studentsNumberLabel.GetLabel()+str(self.grid.GetNumberRows()))
        
        self.addButton=wx.Button(self.panel,-1,"新增",pos=(450,100),size=(80,20))
        self.Bind(wx.EVT_BUTTON,self.onAdd,self.addButton)
    
        self.saveButton=wx.Button(self.panel,-1,"保存",pos=(450,150),size=(80,20))
        self.Bind(wx.EVT_BUTTON,self.onSave,self.saveButton)

        self.deleteButton=wx.Button(self.panel,-1,"删除",pos=(450,200),size=(80,20))
        self.Bind(wx.EVT_BUTTON,self.onDelete,self.deleteButton)

        self.closeButton=wx.Button(self.panel,-1,"退出",pos=(450,250),size=(80,20))
        self.Bind(wx.EVT_BUTTON,self.onClose,self.closeButton)
        self.saveButton.Disable()

    def onAdd(self,event):
        self.grid.AppendRows()
        self.grid.SelectRow(self.grid.GetNumberRows()-1)
        self.grid.MovePageDown()

        for j in range(7):
            self.grid.SetReadOnly(self.grid.GetNumberRows()-1,j,False)
        self.grid.SetCellValue(self.grid.GetNumberRows()-1,2,'男')
        self.grid.SetCellValue(self.grid.GetNumberRows()-1,3,'20')
        self.grid.SetCellValue(self.grid.GetNumberRows()-1,4,'计算机软件')
        self.grid.SetCellValue(self.grid.GetNumberRows()-1,6,'123456')
                
        self.addButton.Disable()
        self.deleteButton.Disable()
        self.closeButton.Disable()
        self.saveButton.Enable()
    def onSave(self,event):
        
        studentList = []
        for j in range(7):
            self.grid.SetReadOnly(self.grid.GetNumberRows()-1,j,False)
            studentList.append(self.grid.GetCellValue(self.grid.GetNumberRows()-1,j))
        result =  self.manage.addStudent(studentList)
        
        if result == 1:
            wx.MessageBox("添加成功！", "成功",wx.OK, self)
            self.studentsNumberLabel.SetLabel("记录总数："+str(self.grid.GetNumberRows()))
            self.addButton.Enable()
            self.deleteButton.Enable()
            self.closeButton.Enable()
            self.saveButton.Disable()
            self.setGridReadOnly()
        elif result == -1:
            answer = wx.MessageBox("学生信息没有填写完整，继续填写？", "提示",wx.CANCEL|wx.YES_NO, self)
            
            if answer == wx.CANCEL:
                pass
            elif answer == wx.YES:
                pass
            else:
                self.addButton.Enable()
                self.deleteButton.Enable()
                self.closeButton.Enable()
                self.saveButton.Disable()
                self.setGridReadOnly()
                self.grid.DeleteRows(self.grid.GetNumberRows()-1)
        else:
            answer = wx.MessageBox("该学生已存在，继续填写？", "提示",wx.CANCEL|wx.YES_NO, self)
            
            if answer == wx.CANCEL:
                pass
            elif answer == wx.YES:
                pass
            else:
                self.addButton.Enable()
                self.deleteButton.Enable()
                self.closeButton.Enable()
                self.saveButton.Disable()
                self.setGridReadOnly()
                self.grid.DeleteRows(self.grid.GetNumberRows()-1)
                
        
    def onDelete(self,event):
        deleteList = self.grid.GetSelectedRows()
        if deleteList == []:
            wx.MessageBox("你没选择任何一行，请点击左侧序号选择一行", "提示",wx.OK, self)
        else:
            answer = wx.MessageBox("确定要删除"+str(len(deleteList))+"条学生信息？", "提示",wx.YES_NO, self)
            if answer == wx.YES:
                for i in range(len(deleteList)):
                    self.manage.deleteStudent(self.grid.GetCellValue(deleteList[i],0))
                for j in range(len(deleteList)-1,-1,-1):
                    self.grid.DeleteRows(deleteList[j])
                self.studentsNumberLabel.SetLabel("记录总数："+str(self.grid.GetNumberRows()))
                wx.MessageBox("删除成功！", "成功",wx.OK, self)
    def updateStudents(self):
        students = self.manage.getStudentsMessage()
        
        self.grid.ClearGrid()
        for i in range(len(students)):
            for j in range(len(students[i])):
                self.grid.SetCellValue(i, j, students[i][j])
    def setGridReadOnly(self):
        self.grid.EnableEditing(True)
        for i in range(self.grid.GetNumberRows()):
            for j in range(7):
                self.grid.SetReadOnly(i,j)
                
    def onClose(self,event):
        answer = wx.MessageBox("确认退出?", "确认",wx.YES_NO | wx.CANCEL, self)
        if answer == wx.YES:
            self.Close()
class ClassesFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self,parent=None,title="课程信息",pos=(100,100),size=(550,350))

        self.panel=wx.Panel(self)
        
        self.classesNumberLabel=wx.StaticText(self.panel,-1,"记录总数：",pos=(20,10))
        self.classesNumberLabel.SetFont(wx.Font(18, wx.ROMAN, wx.ITALIC, wx.NORMAL))
        
        self.manage = Manage.Manage()
        rowLength = len(self.manage.getClassesMessage())
        
        self.grid = wx.grid.Grid(self.panel,id=-1,pos=(20,80),size=(410,220))
        self.grid.CreateGrid(rowLength, 5)
        self.grid.SetRowLabelSize(30)
        self.grid.SetColLabelValue(0,'课程号')
        self.grid.SetColSize(0, 40)
        self.grid.SetColLabelValue(1,'课程名')
        self.grid.SetColSize(1, 100)
        self.grid.SetColLabelValue(2,'学分数')
        self.grid.SetColSize(2, 40)
        self.grid.SetColLabelValue(3,'所在系')
        self.grid.SetColSize(3, 100)
        self.grid.SetColLabelValue(4,'任课教师')
        self.grid.SetColSize(4, 100)
        self.grid.EnableEditing(False)
        self.setGridReadOnly()
        self.updateClasses()
        

        self.classesNumberLabel.SetLabel("记录总数："+str(self.grid.GetNumberRows()))
        
        self.addButton=wx.Button(self.panel,-1,"新增",pos=(450,100),size=(80,20))
        self.Bind(wx.EVT_BUTTON,self.onAdd,self.addButton)
    
        self.saveButton=wx.Button(self.panel,-1,"保存",pos=(450,150),size=(80,20))
        self.Bind(wx.EVT_BUTTON,self.onSave,self.saveButton)

        self.deleteButton=wx.Button(self.panel,-1,"删除",pos=(450,200),size=(80,20))
        self.Bind(wx.EVT_BUTTON,self.onDelete,self.deleteButton)

        self.closeButton=wx.Button(self.panel,-1,"退出",pos=(450,250),size=(80,20))
        self.Bind(wx.EVT_BUTTON,self.onClose,self.closeButton)
        self.saveButton.Disable()

    def onAdd(self,event):
        self.grid.AppendRows()
        self.grid.SelectRow(self.grid.GetNumberRows()-1)
        self.grid.MovePageDown()

        for j in range(5):
            self.grid.SetReadOnly(self.grid.GetNumberRows()-1,j,False)
        self.grid.SetCellValue(self.grid.GetNumberRows()-1,2,'4')
        self.grid.SetCellValue(self.grid.GetNumberRows()-1,3,'计算机软件')
                
        self.addButton.Disable()
        self.deleteButton.Disable()
        self.closeButton.Disable()
        self.saveButton.Enable()
    def onSave(self,event):
        
        classesList = []
        for j in range(5):
            self.grid.SetReadOnly(self.grid.GetNumberRows()-1,j,False)
            classesList.append(self.grid.GetCellValue(self.grid.GetNumberRows()-1,j))
        result =  self.manage.addClass(classesList)
        if result == 1:
            wx.MessageBox("添加成功！", "成功",wx.OK, self)
            self.classesNumberLabel.SetLabel("记录总数："+str(self.grid.GetNumberRows()))
            self.addButton.Enable()
            self.deleteButton.Enable()
            self.closeButton.Enable()
            self.saveButton.Disable()
            self.setGridReadOnly()
        elif result == -1:
            answer = wx.MessageBox("课程信息没有填写完整，继续填写？", "提示",wx.CANCEL|wx.YES_NO, self)
            
            if answer == wx.CANCEL:
                pass
            elif answer == wx.YES:
                pass
            else:
                self.addButton.Enable()
                self.deleteButton.Enable()
                self.closeButton.Enable()
                self.saveButton.Disable()
                self.setGridReadOnly()
                self.grid.DeleteRows(self.grid.GetNumberRows()-1)
        else:
            answer = wx.MessageBox("该课程已存在，继续填写？", "提示",wx.CANCEL|wx.YES_NO, self)
            
            if answer == wx.CANCEL:
                pass
            elif answer == wx.YES:
                pass
            else:
                self.addButton.Enable()
                self.deleteButton.Enable()
                self.closeButton.Enable()
                self.saveButton.Disable()
                self.setGridReadOnly()
                self.grid.DeleteRows(self.grid.GetNumberRows()-1)
                
        
    def onDelete(self,event):
        deleteList = self.grid.GetSelectedRows()
        if deleteList == []:
            wx.MessageBox("你没选择任何一行，请点击左侧序号选择一行", "提示",wx.OK, self)
        else:
            answer = wx.MessageBox("确定要删除"+str(len(deleteList))+"条课程信息？", "提示",wx.YES_NO, self)
            if answer == wx.YES:
                for i in range(len(deleteList)):
                    self.manage.deleteClass(self.grid.GetCellValue(deleteList[i],0))
                for j in range(len(deleteList)-1,-1,-1):
                    self.grid.DeleteRows(deleteList[j])
                self.classesNumberLabel.SetLabel("记录总数："+str(self.grid.GetNumberRows()))
                wx.MessageBox("删除成功！", "成功",wx.OK, self)
    def updateClasses(self):
        classes = self.manage.getClassesMessage()
        
        self.grid.ClearGrid()
        for i in range(len(classes)):
            for j in range(len(classes[i])):
                self.grid.SetCellValue(i, j, classes[i][j])
    def setGridReadOnly(self):
        self.grid.EnableEditing(True)
        for i in range(self.grid.GetNumberRows()):
            for j in range(7):
                self.grid.SetReadOnly(i,j)
                
    def onClose(self,event):
        answer = wx.MessageBox("确认退出?", "确认",wx.YES_NO | wx.CANCEL, self)
        if answer == wx.YES:
            self.Close() 
class ManageFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self,parent=None,title="管理界面",pos=(100,100),size=(550,400))

        menuBar = wx.MenuBar()
        openMenu = wx.Menu()
        closeMenu = wx.Menu()
        
        self.studentsMenuSelect = openMenu.Append(-1, u'学生表\tCtrl+S')
        self.Bind(wx.EVT_MENU, self.onStudentsMenuSelect, self.studentsMenuSelect)
        self.classesMenuSelect = openMenu.Append(-1, u'课程表\tCtrl+C')
        self.Bind(wx.EVT_MENU, self.onClassesMenuSelect, self.classesMenuSelect)
        
        self.closeMenuSelect = closeMenu.Append(-1, u'关闭\tCtrl+Q')
        self.Bind(wx.EVT_MENU, self.onCloseMenuSelect, self.closeMenuSelect)
        
        menuBar.Append(openMenu, u'维护')
        menuBar.Append(closeMenu, u'关闭')
        
        self.SetMenuBar(menuBar)
        
        self.panel=wx.Panel(self)
        self.classLabel=wx.StaticText(self.panel,-1,"课程:",pos=(20,10))
        self.classLabel.SetFont(wx.Font(18, wx.ROMAN, wx.ITALIC, wx.NORMAL))
        self.teacherLabel=wx.StaticText(self.panel,-1,"任课教师：",pos=(200,10))
        self.teacherLabel.SetFont(wx.Font(18, wx.ROMAN, wx.ITALIC, wx.NORMAL))
        mentionLabel=wx.StaticText(self.panel,-1,"请选择课程名：",pos=(20,100))
        studentsLabel=wx.StaticText(self.panel,-1,"已选修此课程的学生：",pos=(200,100))

        self.manage = Manage.Manage()
        self.classesList,self.teacheresList=self.manage.getClasses()
        self.classesComboBox = wx.ComboBox(self.panel,-1,value='\t',choices=self.classesList,pos=(20,130))
        self.Bind(wx.EVT_COMBOBOX,self.onClassSelect,self.classesComboBox)
        
        
        
        self.grid = wx.grid.Grid(self.panel,id=-1,pos=(200,130),size=(180,200))
        self.grid.CreateGrid(10, 2)
        self.grid.HideRowLabels()
        self.grid.SetColLabelValue(0,'学号')
        self.grid.SetColLabelValue(1,'成绩')
        self.grid.EnableEditing(False)
        self.setGridReadOnly()
        
        

         
        self.searchButton=wx.Button(self.panel,-1,"查询",pos=(400,150),size=(80,20))
        self.Bind(wx.EVT_BUTTON,self.onSearch,self.searchButton)
    
        self.inputGradeButton=wx.Button(self.panel,-1,"输入成绩",pos=(400,200),size=(80,20))
        self.Bind(wx.EVT_BUTTON,self.onInputGrade,self.inputGradeButton)

        self.gradeSpreadButton=wx.Button(self.panel,-1,"成绩分布",pos=(400,250),size=(80,20))
        self.Bind(wx.EVT_BUTTON,self.onGradeSpread,self.gradeSpreadButton)

        self.closeButton=wx.Button(self.panel,-1,"退出",pos=(400,300),size=(80,20))
        self.Bind(wx.EVT_BUTTON,self.close,self.closeButton)

        self.inputGradeButton.Disable()
        self.gradeSpreadButton.Disable() 
    def onStudentsMenuSelect(self,event):
        frame = StudentsFrame()
        frame.Show(True)
            
    def onClassesMenuSelect(self,event):
        frame = ClassesFrame()
        frame.Show(True)
    def onCloseMenuSelect(self,event):
        self.close(event)
        
    def onClassSelect(self,event):
        self.classLabel.SetLabel(u"课程："+event.GetString())
        self.teacherLabel.SetLabel(u"任课教师："+ self.teacheresList[event.GetSelection()])

    def onSearch(self,event):
        if self.classesComboBox.GetStringSelection() == '':
            wx.MessageBox("你还没有选择课程！", "注意",wx.OK, self)
        else:
            self.inputGradeButton.Enable()
            self.gradeSpreadButton.Enable() 
            self.setGridReadOnly()
            self.updateStudents()
            for i in range(10):
                if self.grid.GetCellValue(i,1)!='':
                    self.grid.SetReadOnly(i,1,False)
        
        
    def updateStudents(self):
        classStudents = self.manage.getClassStudents(self.classesComboBox.GetStringSelection())
        
        self.grid.ClearGrid()
        for i in range(len(classStudents)):
            for j in range(len(classStudents[i])):
                self.grid.SetCellValue(i, j, classStudents[i][j])
    def setGridReadOnly(self):
        for i in range(10):
            self.grid.SetReadOnly(i,0)
            self.grid.SetReadOnly(i,1)
    def onInputGrade(self,event):
        
        if self.inputGradeButton.GetLabel() == u"输入成绩":
            
            self.searchButton.Disable() 
            self.gradeSpreadButton.Disable()
            self.closeButton.Disable()
            
            self.grid.EnableEditing(True)
            self.inputGradeButton.SetLabel("保存")
        else:
            self.searchButton.Enable() 
            self.gradeSpreadButton.Enable()
            self.closeButton.Enable()
            
            self.grid.EnableEditing(False)
            self.inputGradeButton.SetLabel("输入成绩")
            studentsGrade = []
            studentsNo = []
            classNo = self.manage.getClassNo(self.classesComboBox.GetStringSelection())
            result = 0
            for i in range(10):
                if self.grid.GetCellValue(i,1) != u'':
                    result += self.manage.updateGrade(self.grid.GetCellValue(i,0),classNo,self.grid.GetCellValue(i,1))
            self.updateStudents()
            if result == 0:
                wx.MessageBox("未做任何更新！", "完成",wx.OK, self)
            elif result > 0:
                wx.MessageBox("更新了"+str(result)+"个学生的信息", "成功",wx.OK, self)
            
        
    def onGradeSpread(self):
        pass
    def close(self,event):
        answer = wx.MessageBox("确认退出?", "确认",wx.YES_NO | wx.CANCEL, self)
        if answer == wx.YES:
            self.Close() 
class MainFrame(wx.Frame):
    def __init__(self,studentID):
        self.studentID = studentID
        self.student = SelectClass.Student(studentID)
        wx.Frame.__init__(self,parent=None,title="学生选课",pos=(100,100),size=(1000,600))
 

        menuBar = wx.MenuBar()
        openMenu = wx.Menu()
        closeMenu = wx.Menu()
        
        self.gradeMenuSelect = openMenu.Append(-1, u'学生成绩单\tCtrl+A')
        self.Bind(wx.EVT_MENU, self.onGradeMenuSelect, self.gradeMenuSelect)
        self.helpMenuSelect = closeMenu.Append(-1, u'帮助\tCtrl+H')
        self.Bind(wx.EVT_MENU, self.onHelpMenuSelect, self.helpMenuSelect)
        self.closeMenuSelect = closeMenu.Append(-1, u'关闭\tCtrl+Q')
        self.Bind(wx.EVT_MENU, self.onCloseMenuSelect, self.closeMenuSelect)
        
        menuBar.Append(openMenu, u'打开')
        menuBar.Append(closeMenu, u'关闭')
        
        self.SetMenuBar(menuBar)
        
        self.panel=wx.Panel(self)
        self.studentLabel=wx.StaticText(self.panel,-1,"学生详细信息:",pos=(20,10))
        self.studentLabel.SetForegroundColour(wx.Colour(200, 200, 200))

        
        grid = wx.grid.Grid(self.panel,id=-1,pos=(20,30),size=(300,200))
        grid.CreateGrid(1, 5)
        grid.HideRowLabels()
        grid.SetColLabelValue(0,'学号')
        grid.SetColLabelValue(1,'姓名')
        grid.SetColLabelValue(2,'年龄')
        grid.SetColLabelValue(3,'性别')
        grid.SetColLabelValue(4,'学院')
                
        
        studentMessage = self.student.getStudentMessage()
        for i in range(len(studentMessage)):
            grid.SetCellValue(0, i, studentMessage[i])

        self.classLabel=wx.StaticText(self.panel,-1,"可选课程:",pos=(400,10))
        self.classLabel.SetForegroundColour(wx.Colour(200, 200, 200))

        
        self.validGrid = wx.grid.Grid(self.panel,id=-1,pos=(400,30),size=(300,200))
        self.validGrid.CreateGrid(10, 5)
        self.validGrid.HideRowLabels()
        self.validGrid.SetColLabelValue(0,'课程号')
        self.validGrid.SetColLabelValue(1,'课程名')
        self.validGrid.SetColLabelValue(2,'学分')
        self.validGrid.SetColLabelValue(3,'开课系')
        self.validGrid.SetColLabelValue(4,'任课教师')
        
        self.updateValidClasses()
            

        self.gradeLabel=wx.StaticText(self.panel,-1,"已修课程成绩:",pos=(20,300))
        self.gradeLabel.SetForegroundColour(wx.Colour(200, 200, 200))
        grid = wx.grid.Grid(self.panel,id=-1,pos=(20,330),size=(300,200))
        grid.CreateGrid(10, 3)
        grid.HideRowLabels()
        grid.SetColSize(1, 150)
        grid.SetColLabelValue(0,'课程号')
        grid.SetColLabelValue(1,'课程名')
        grid.SetColLabelValue(2,'成绩')
        
        classesGrade = self.student.getStudentGrade()
        for i in range(len(classesGrade)):
            for j in range(len(classesGrade[i])):
                grid.SetCellValue(i, j, classesGrade[i][j])
        

        self.selectedClassLabel=wx.StaticText(self.panel,-1,"已选课程:",pos=(400,300))
        self.selectedClassLabel.SetForegroundColour(wx.Colour(200, 200, 200))

        self.selectedGrid = wx.grid.Grid(self.panel,id=-1,pos=(400,330),size=(300,200))
        self.selectedGrid.CreateGrid(10, 5)
        self.selectedGrid.HideRowLabels()
        self.selectedGrid.SetColLabelValue(0,'课程号')
        self.selectedGrid.SetColLabelValue(1,'课程名')
        self.selectedGrid.SetColLabelValue(2,'学分')
        self.selectedGrid.SetColLabelValue(3,'开课系')
        self.selectedGrid.SetColLabelValue(4,'任课教师')
        
        self.updateSelectedClasses()

        


        self.classNumberLabel=wx.StaticText(self.panel,-1,"请输入课程号：",pos=(750,10))
        self.classNumberLabel.SetForegroundColour(wx.Colour(200, 200, 200))
        self.classNumberText=wx.TextCtrl(self.panel,-1,pos=(750,30),size=(200,20))


        
        self.selectClassButton=wx.Button(self.panel,-1,"选课",pos=(750,100),size=(200,20))
        self.Bind(wx.EVT_BUTTON,self.selectClass,self.selectClassButton)
    
        self.deleteClassButton=wx.Button(self.panel,-1,"退课",pos=(750,200),size=(200,20))
        self.Bind(wx.EVT_BUTTON,self.deleteClass,self.deleteClassButton)

        self.closeButton=wx.Button(self.panel,-1,"关闭",pos=(750,300),size=(200,20))
        self.Bind(wx.EVT_BUTTON,self.close,self.closeButton)
    def onGradeMenuSelect(self,event):
        frame = GradeFrame(self.student)
        frame.Show(True)

    def onHelpMenuSelect(self,event):
        pass
    def onCloseMenuSelect(self,event):
        self.close(event)
    def updateSelectedClasses(self):
        self.selectedGrid.ClearGrid()
        selectedClasses = self.student.getSelectedClasses()
        for i in range(len(selectedClasses)):
            for j in range(len(selectedClasses[i])):
                self.selectedGrid.SetCellValue(i, j, selectedClasses[i][j])
    def updateValidClasses(self):
        self.validGrid.ClearGrid()
        validClasses = self.student.getValidClasses()
        for i in range(len(validClasses)):
            for j in range(len(validClasses[i])):
                self.validGrid.SetCellValue(i, j, validClasses[i][j])

    def selectClass(self,event):
        classNumber=self.classNumberText.GetValue()
        if classNumber=='':
            wx.MessageBox("你还没有输入课程号。", "注意！",wx.OK, self)
        else:

            result = self.student.selectClass(classNumber)
            self.updateSelectedClasses()
            self.updateValidClasses()
            
            if result == -1:
                wx.MessageBox("此课程已选！", "失败",wx.OK, self)
            elif result == -2:
                wx.MessageBox("没有这门课！", "失败",wx.OK, self)
            elif result == 1:
                wx.MessageBox("选课成功！", "成功",wx.OK, self)  
    def deleteClass(self,event):
        classNumber=self.classNumberText.GetValue()
        if classNumber=='':
            wx.MessageBox("你还没有输入课程号。", "注意！",wx.OK, self)
        else:

            result = self.student.deleteClass(classNumber)
            self.updateSelectedClasses()
            self.updateValidClasses()
            
            if result == -1:
                wx.MessageBox("此课程已登分！", "失败",wx.OK, self)
            elif result == -2:
                wx.MessageBox("此课程不存在！", "失败",wx.OK, self)
            elif result == 1:
                wx.MessageBox("退课成功！", "成功",wx.OK, self)
    def close(self,event):
        answer = wx.MessageBox("确认退出?", "确认",wx.YES_NO | wx.CANCEL, self)
        if answer == wx.YES:
            self.Close() 
class LoadFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self,parent=None,title="登陆界面",pos=(100,100),size=(400,200))
        self.panel=wx.Panel(self)
        self.panel.SetBackgroundColour(wx.Colour(100, 100, 100))

        self.titleLabel=wx.StaticText(self.panel,-1,"请输入你的用户名和密码：",pos=(20,10))
        self.titleLabel.SetForegroundColour(wx.Colour(100, 200, 200))

        self.usernameLabel=wx.StaticText(self.panel,-1,"用户名",pos=(20,50))
        self.usernameLabel.SetForegroundColour(wx.Colour(200, 200, 200))       
        self.usernameText=wx.TextCtrl(self.panel,-1,pos=(100,50),size=(200,20))
        
        self.passwordLabel=wx.StaticText(self.panel,-1,"密码",pos=(20,80))
        self.passwordLabel.SetForegroundColour(wx.Colour(200, 200, 200))
        self.passwordText=wx.TextCtrl(self.panel,-1,pos=(100,80),size=(200,20))

        
        self.loadButton=wx.Button(self.panel,-1,"登陆",pos=(100,120))
        self.Bind(wx.EVT_BUTTON,self.load,self.loadButton)

        self.cancelButton=wx.Button(self.panel,-1,"取消",pos=(200,120))
        self.Bind(wx.EVT_BUTTON,self.cancel,self.cancelButton)
        
    def load(self,event):
        username=self.usernameText.GetValue()
        password=self.passwordText.GetValue()
        if username=='':
            wx.MessageBox("请输入你的用户名。", "注意！",wx.OK, self)
        elif password=='':
            wx.MessageBox("请输入你的密码。", "注意！",wx.OK, self)
        else:
            result = Load.load(username,password)
            if result == -1:
                wx.MessageBox("用户名或密码错误。", "错误！",wx.OK, self)
            elif result == 1:
                frame = ManageFrame()
                frame.Show(True)
                self.Close()
            else :
                frame = MainFrame(result)
                frame.Show(True)
                self.Close() 
         
    def cancel(self,event):
        answer = wx.MessageBox("确认退出?", "确认", wx.YES_NO | wx.CANCEL, self)
        if answer == wx.YES:
            self.Close()                 
  
app=wx.App()
frame = LoadFrame()
frame.Show(True)
app.MainLoop()
