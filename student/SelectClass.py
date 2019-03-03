#!/usr/bin/python
# -*- coding: UTF-8 -*-
import wx
import re
import MySQLdb

class Student:
    def __init__(self,studentNo):
        self.studentNo = studentNo
    def getStudentMessage(self):
        db = MySQLdb.connect("localhost", "lee", "lee", "powerbuilder", charset='utf8' )
        cursor = db.cursor()

        sql = "Select sno,sname,age,sex,sdept from S where sno = '"+self.studentNo+"'"
        
        cursor.execute(sql)
        
        data = list(cursor.fetchone())
        
        data = map(unicode,data)

        db.close()
        return data
    def getStudentGrade(self):
        db = MySQLdb.connect("localhost", "lee", "lee", "powerbuilder", charset='utf8' )
        cursor = db.cursor()
        
        sql = "Select distinct C.cno,cname,grade from C,SC where C.cno = SC.cno and SC.sno = '"\
              +self.studentNo+"' and grade is not null"
        
        cursor.execute(sql)
        data = cursor.fetchall()
        dataList = []
        for i in range(len(data)):
            dataList.append(list(data[i]))
            for j in range(len(data[i])):
                if isinstance(data[i][j],long):
                    dataList[i][j] = str(data[i][j])
        db.close()           
        return dataList
    def getCompleteStudentGrade(self):
        db = MySQLdb.connect("localhost", "lee", "lee", "powerbuilder", charset='utf8' )
        cursor = db.cursor()
        
        sql = "Select distinct C.cno,cname,grade,credit,tname from C,SC where C.cno = SC.cno and SC.sno = '"\
              +self.studentNo+"' and grade is not null"
        
        cursor.execute(sql)
        data = cursor.fetchall()
        dataList = []
        for i in range(len(data)):
            dataList.append(list(data[i]))
            for j in range(len(data[i])):
                if isinstance(data[i][j],long):
                    dataList[i][j] = str(data[i][j])
        db.close()           
        return dataList
    def getValidClasses(self):
        db = MySQLdb.connect("localhost", "lee", "lee", "powerbuilder", charset='utf8' )
        cursor = db.cursor()
        
        sql = "Select cno,cname,credit,cdept,tname from C where cno not in(select cno from SC where SC.sno =  '"\
              +self.studentNo+"')"
        
        cursor.execute(sql)
        data = cursor.fetchall()
        dataList = []


        for i in range(len(data)):
            dataList.append(list(data[i]))
            for j in range(len(data[i])):
                if isinstance(data[i][j],long):
                    dataList[i][j] = str(data[i][j])

        db.close()           
        return dataList
    def getSelectedClasses(self):
        db = MySQLdb.connect("localhost", "lee", "lee", "powerbuilder", charset='utf8' )
        cursor = db.cursor()
        
        sql = "Select distinct C.cno,cname,credit,cdept,tname from C,SC where C.cno = SC.cno and SC.sno = '"\
              +self.studentNo+"' and grade is null"
        cursor.execute(sql)
        data = cursor.fetchall()
        dataList = []
        for i in range(len(data)):
            dataList.append(list(data[i]))
            for j in range(len(data[i])):
                if isinstance(data[i][j],long):
                    dataList[i][j] = str(data[i][j])
        db.close()           
        return dataList
    def selectClass(self,classNo):
        db = MySQLdb.connect("localhost", "lee", "lee", "powerbuilder", charset='utf8' )
        cursor = db.cursor()
        result = 1
        try:
            sql = "insert into SC (sno,cno) values ('"\
                  +self.studentNo+"', '"+classNo+"')"
            cursor.execute(sql)
            db.commit()
        except MySQLdb.IntegrityError,error:
            db.rollback()
            if error[0] == 1062:
                result = -1
                
            elif error[0] == 1216:
                result = -2
        
        db.close()
        return result
    def deleteClass(self,classNo):
        db = MySQLdb.connect("localhost", "lee", "lee", "powerbuilder", charset='utf8' )
        cursor = db.cursor()
        result = 1
        try:
            sql = "delete from SC where sno = '"\
                  +self.studentNo+"' and cno = '"+classNo+"' and grade is null"
            if cursor.execute(sql) == 0:
                sql = "Select distinct C.cno,cname,credit,cdept,tname from C,SC where C.cno = SC.cno and SC.sno = '"\
                    +self.studentNo+"' and C.cno = '"+classNo+"'"
                if cursor.execute(sql):
                    result = -1
                else:
                    result = -2
            
            db.commit()
        except:
            
            db.rollback()
            
        
        db.close()
        return result
