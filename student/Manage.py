#!/usr/bin/python
# -*- coding: UTF-8 -*-
import wx
import re
import MySQLdb

class Manage:
    
    def addStudent(self,studentList):
        db = MySQLdb.connect("localhost", "lee", "lee", "powerbuilder", charset='utf8' )
        cursor = db.cursor()
        
        
        for i in range(len(studentList)):
            if studentList[i]==u"":
                return -1
        result = 0
        try:
            sql = "insert into  S values ('"+studentList[0]+"','"+studentList[1]+"', '"+studentList[2]+"','"\
                  +studentList[3]+"','"+studentList[4]+"', '"+studentList[5]+"','"+studentList[6]+"')"
            result = cursor.execute(sql)
            db.commit()
        except MySQLdb.IntegrityError,error:
            db.rollback()
        
        db.close()
        return result
    def deleteStudent(self,studentNo):
        db = MySQLdb.connect("localhost", "lee", "lee", "powerbuilder", charset='utf8' )
        cursor = db.cursor()
        
        
        result = 0
        try:
            sql = "delete from S where sno = '"+studentNo+"'"
            result = cursor.execute(sql)
            db.commit()
        except MySQLdb.IntegrityError,error:
            db.rollback()
        
        db.close()
        return result
    def getStudentsMessage(self):
        db = MySQLdb.connect("localhost", "lee", "lee", "powerbuilder", charset='utf8' )
        cursor = db.cursor()
        
        sql = "Select * from S"
        
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
    def addClass(self,studentList):
        db = MySQLdb.connect("localhost", "lee", "lee", "powerbuilder", charset='utf8' )
        cursor = db.cursor()
        
        
        for i in range(len(studentList)):
            if studentList[i]==u"":
                return -1
        result = 0
        try:
            sql = "insert into  C values ('"+studentList[0]+"','"+studentList[1]+"', '"+studentList[2]+"','"\
                  +studentList[3]+"','"+studentList[4]+"')"
            result = cursor.execute(sql)
            db.commit()
        except MySQLdb.IntegrityError,error:
            db.rollback()
        
        db.close()
        return result
    def deleteClass(self,studentNo):
        db = MySQLdb.connect("localhost", "lee", "lee", "powerbuilder", charset='utf8' )
        cursor = db.cursor()
        
        
        result = 0
        try:
            sql = "delete from C where cno = '"+studentNo+"'"
            result = cursor.execute(sql)
            db.commit()
        except MySQLdb.IntegrityError,error:
            db.rollback()
        
        db.close()
        return result
    def getClassesMessage(self):
        db = MySQLdb.connect("localhost", "lee", "lee", "powerbuilder", charset='utf8' )
        cursor = db.cursor()
        
        sql = "Select * from C"
        
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
    def getClasses(self):
        db = MySQLdb.connect("localhost", "lee", "lee", "powerbuilder", charset='utf8' )
        cursor = db.cursor()

        sql = "Select cname,tname from C"
        
        cursor.execute(sql)
        
        data = cursor.fetchall()
        classesList = []
        teacheresList = []
        for i in range(len(data)):
            classesList.append(data[i][0])
            teacheresList.append(data[i][1])
            
        db.close()
        return classesList,teacheresList
    def getClassNo(self,className):
        db = MySQLdb.connect("localhost", "lee", "lee", "powerbuilder", charset='utf8' )
        cursor = db.cursor()
        
        sql = "Select C.cno from SC,C where SC.cno = C.cno and C.cname = '"+className+"'"
        
        cursor.execute(sql)
        data = cursor.fetchone()[0]
        db.close()
        return data
    def getClassStudents(self,className):
        db = MySQLdb.connect("localhost", "lee", "lee", "powerbuilder", charset='utf8' )
        cursor = db.cursor()
        
        sql = "Select S.sno,grade from S,SC,C where S.sno = SC.sno and SC.cno = C.cno and C.cname = '"+className+"' and grade is not null"
        
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
    def updateGrade(self,studentNo,classNo,grade):
        db = MySQLdb.connect("localhost", "lee", "lee", "powerbuilder", charset='utf8' )
        cursor = db.cursor()
        
        result = 0
        try:
            sql = "update SC set grade = '"+grade+"' where sno = '"+studentNo+"' and cno = '"+classNo+"'"
            result = cursor.execute(sql)
            print result
            db.commit()
        except MySQLdb.IntegrityError,error:
            db.rollback()
        
        db.close()
        return result

