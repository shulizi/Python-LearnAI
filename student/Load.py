#!/usr/bin/python
# -*- coding: UTF-8 -*-
import wx
import threading
import MySQLdb
def load(username,password):
    db = MySQLdb.connect("localhost", "lee", "lee", "powerbuilder", charset='utf8' )
    cursor = db.cursor()
    sql = "Select sno from S where logn = '"+username+"' and pswd = '"+password+"'"        
    cursor.execute(sql)
        
    data = cursor.fetchone()
    db.close()
    if data!=None:
        print data[0]
        return data[0]
    if username == "system" and password == "system" :
        return 1
    else:
        return -1
    
