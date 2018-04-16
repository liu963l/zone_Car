# -*- coding: utf-8 -*-
"""
Created on Tue Apr 03 21:20:36 2018

@author: Liu-pc
"""



#什么情况下会

import pandas as pd
import numpy as np

data = pd.read_csv("C:\Users\Liu-pc\Desktop\\973.csv")

data = data[data['LOC_TREND'] == 1]    


 
def od_matrix1(data,station_num = 51):
    #计算出od矩阵
    l = []
    for i in range( station_num): 
        for j in range(station_num):
            l.append(len(data[(data['ON_STATIONID']==i)&(data['OFF_STATIONID']==j)]))
    l = np.array(l).reshape(station_num,station_num)
    return l

od = od_matrix1(data, 51)  

def od_matrix(data,station_num = 51):
    #计算出od矩阵
    l = []
    for i in range(station_num):
        l = l + [0]*( i + 1 )
        for j in range(i + 1,station_num):
            l.append(len(data[(data['ON_STATIONID']==i)&(data['OFF_STATIONID']==j)]))
    l = np.array(l).reshape(station_num,station_num)
    return l
od = od_matrix(data, 51)


def shuttle(od_matrix, num, direct = 1):
    #给定区间长度，计算合适的区间位置
    l = []
    #dric为方向，num为区间长度
    if direct == 1:
        for i in range(len(od_matrix) - num):
            a = float(sum(od_matrix[i:i+num,i:i+num]))/sum(od_matrix[i:i+num+1,:]) 
            if a>=float(num/len(od_matrix)):
            #if a>=0.5:
                l.append((i,a,sum(od_matrix[i:i+num,i:i+num])))
                print i
            else:
                continue
    elif direct == 2:
        for i in range(len(od_matrix) - num):
            a = float(sum(od_matrix[i:i+num,i:i+num]))/sum(od_matrix[:,i:i+  num]) 
            
            if a>=float(num/len(od_matrix)):
            #if a>=0.5:
                l.append((i,a,sum(od_matrix[i:i+num,i:i+num])))
                print i
            else:
                continue
    a = map(lambda x: x[2], l)
    m = a.index(max(a))
    return l[m]
                   

a = shuttle(od, 10,1)



#################获取合理区间集合，上下行的区间最好在一个区间


import pandas as pd
rou = pd.read_csv('F:\\1zonel vehicle\\rou.csv')
del rou['station'] 
rou = rou/60.0

lon_lat = pd.read_csv('F:\\1zonel vehicle\\distance.csv')

def calcu_distance(lon1,lat1,lon2,lat2):
      dx = lon1 - lon2
      dy = lat1 - lat2
      b = (lat1 + lat2) / 2.0;
      Lx = (dx/57.2958) * 6371004.0* math.cos(b/57.2958)
      Ly = 6371004.0 * (dy/57.2958)
      return math.sqrt(Lx * Lx + Ly * Ly)
  
distance = []
for i in range(len(lon_lat)-1):
    distance.append(calcu_distance(lon_lat['lon'][i],lon_lat['lat'][i],lon_lat['lon'][i +1],lon_lat['lat'][i+1 ]))


speed = pd.read_csv('F:\\1zonel vehicle\\speed.csv')

'''
data = pd.read_csv("C:\Users\Liu-pc\Desktop\\973.csv")
#data = pd.read_csv("H:\data\\973_data.csv",encoding = 'gbk')
#data = data[data['LINE_CODE'] == 973]
data['ON_STATIONID'] = data['ON_STATIONID'].replace(0,1)
data = data[data['LOC_TREND'] == 1]  
data = data[data['ON_STATIONNAME'] != 'error']
data['ON_STATION_TIME']  = pd.to_datetime(data['ON_STATION_TIME'])
data['on_hour'] = map(lambda x :x.hour, data['ON_STATION_TIME'] )
data_hour_Count = data.groupby(['on_hour', 'ON_STATIONID'])['ON_STATIONID'].count()
'''



leave_time = pd.DataFrame({'stiation':range(1,50)})  #50
arr_time = pd.DataFrame({'stiation':range(1,51)})    #51
stop_t = pd.DataFrame({'stiation':range(1,50)})     #50
up_people = pd.DataFrame({'stiation':range(1,50)})  #50



leave = []
arr = [0]
trav = 0
st = []
for i in range(49):
    n = int(arr[-1]/600) +1
    n = str('speed_%s') %n
    ss = 10
    l = arr[i]  + ss
    trav = distance[i]/speed.loc[i,n]
    arr.append(l + trav)
    leave.append(l)
    st.append(ss)
    print l
    
leave_time['banci1'] = leave
arr_time ['banci1'] = arr
stop_t['banci1'] = st
#####第二班
leave = []
arr = [400]
trav = 0
st = []
people_list = []
for i in range(49):
    n = int(arr[-1]/600) +1
    n = str('speed_%s') %n
    ss = 10
    people = 5
    l = arr[i]  + ss
    trav = distance[i]/speed.loc[i,n]
    arr.append(l + trav)
    leave.append(l)
    st.append(ss)
    people_list.append(people)
    print l    
arr_time ['banci2'] = arr
leave_time['banci2'] = leave
stop_t['banci2'] = st
up_people['banci2'] =  people_list

#stop_t = pd.DataFrame()
def leave_normal(n_ban,station, leave_time):
    #l = range(800,4400,400) + [4400, 4500,4800, 5200] + range(5600, 8000,400) 
    l = [800,1200,1600,2000,2400,2800]
    #l = range(800,7000,400)
    for i in range(n_ban):
        #people = (l[i] - l[i-1])
        left = []   #区间车的剩余人数留给下一班车乘坐
        if (l[i] in [1000,1400]):
            #leave = list(leave_time['banci%s' %(i+2)][0:6])
        #head = []
            leave = []    #离开时间
            arr = [l[i]]  #到达时间
            st = []      #停车时间
            #arr_time ['banci%s' %(i+3)] = arr
            trav = 0    #行程时间
            people_list = []  #上车人数
            for j in range(station-1):
                n = int(arr[-1]/600) + 1
                n1 = str('speed_%s') %n
                s = str('time_%s') %n
                b = str('banci%s') %(i+2)
                #b1 = str('banci%s') %(i+1)
           # print n
                if (j<=11):
                    stop_time = ((arr[j] - arr_time[b][j])* rou[s][j]*2/60*0.5) + 10
                    left.append((arr[j] - arr_time[b][j])* rou[s][j]/60*0.5)
                    people = round((stop_time-10)/2.0)
                    print arr[j] - arr_time[b][j]
                    le = arr[j] +stop_time
                    trav = distance[j]/speed.loc[j,n1]
                    arr.append(le + trav)
                    leave.append(le)
                    st.append(stop_time)
                    people_list.append(people)
                else:
                    arr.append(arr_time[b][j+1])
                    leave.append(leave_time[b][j])
                    st.append(0)
                    people_list.append(0)
                    left.append(0)
            print left
                    
        else:
            
            #leave = [l[i]]
            #leave_time['banci%s' %(i+3)] = l[i]
            leave = []
            arr = [l[i]]
            st = []
            #arr_time ['banci%s' %(i+3)] = arr
            trav = 0
            people_list = []
            for j in range(station-1):
                n = int(arr[-1]/600) + 1
                n1 = str('speed_%s') %n
                s = str('time_%s') %n
                b = str('banci%s') %(i+2)
               # b1 = str('banci%s') %(i+1)
                #b2 = str('banci%s') %(i+3)
                if left != []:
                    stop_time = ((arr[j] - arr_time[b][j]) * rou[s][j]/60 + left[j])*2 + 10
                    people = round((stop_time-10)/2.0)
                    print arr[j] - arr_time[b][j]
                    le = arr[j] +stop_time
                    trav = distance[j]/speed.loc[j,n1]
                    arr.append(le + trav)
                    leave.append(le)
                    st.append(stop_time)
                    people_list.append(people)
                else:
                    stop_time = ((arr[j] - arr_time[b][j]) * rou[s][j]/60 )*2 + 10
                    people = round((stop_time-10)/2.0)
                    print arr[j] - arr_time[b][j]
                    le = arr[j] +stop_time
                    trav = distance[j]/speed.loc[j,n1]
                    arr.append(le + trav)
                    leave.append(le)
                    st.append(stop_time)
                    people_list.append(people)
                
                #head =  leave[-1] - leave_time[b][j]
            #if head< 0.7*600:
                #print j
                
            #print head
            
        print i
        leave_time['banci%s' %(i+3)] = leave
        arr_time ['banci%s' %(i+3)] = arr
        stop_t ['banci%s' %(i+3)] =  st
        up_people['banci%s' %(i+3)] =  people_list
        
        
    return leave_time

'''
设置仿真数据应改为前几站人数较多，控制区间为前几站，这样可以直接应用上述函数，不需要循环给定l列表

if people > **:
    记录着个点车辆离开时间，并且发出区间车
'''
leave_time = leave_normal(6,50,leave_time)





def head1(data):
    dic = pd.DataFrame()
    h = []
    for j in range(6):
        n = j+1
        b = 'banci%i' %(n)
        b1 = 'banci%i' %(n+1)
        for k in range(50):
            r = data[b1][k] - data[b][k]
            h.append(r)
        dic[b] = h
        h = []
    return dic

a = head1(arr_time)  #极端
a1 = head1(leave_time)



###############################提取站间速度， 用off——time############################
data = pd.read_csv("C:\Users\Liu-pc\Desktop\\973.csv")
data['OFF_STATION_TIME'] = pd.to_datetime(data['OFF_STATION_TIME'])
data['off_hour'] = map(lambda x: x.hour, data['OFF_STATION_TIME'])
data_7 = data[data['off_hour'] ==7]

banci = data_7[['VEHICLE_CODE', u'BANCIID']].drop_duplicates()
banci['id'] = range(len(banci))

data_7 = pd.merge(data_7, banci, on = ['VEHICLE_CODE', u'BANCIID'], how = 'inner')

off_time = data_7.groupby(['id','OFF_STATIONID'])['OFF_STATION_TIME'].apply(lambda i:i.iloc[1] if len(i)>1 else np.nan)
off_time = pd.DataFrame(off_time)

off_time['station'] = map(lambda x: x[1], off_time.index)
off_time['id'] = map(lambda x: x[0], off_time.index)

a = off_time.pivot(index = 'station', columns= 'id', values= 'OFF_STATION_TIME')
#下车站点从1开始，得到的最好是1：50这50个站的时间



#下车时间矩阵（每班车的在每一站的下车时间）
#填充
def fill_index(data,banci, station_num):
    index1 = data.index
    print index1
    station = range(1,station_num+1)
    cha =set(station) - set(index1)
    print cha
    a1 = {}
    for i in range(banci):
        l = []
        for j in station:
            #print i
            if j in index1:
                print j
                l.append(data.loc[j,i])
            elif j in cha:
                l.append(data.loc[1,1])
        #print i
        a1[i] = l
    return pd.DataFrame(a1)

a1 = fill_index(a, 20, 50)  #20 班车 50站



    




def gap_time(x):
    l = []
    for i in range(1,len(x)):
        l.append(x[i] - x[i-1])
    return l




###############################################
data = pd.read_csv("H:\data\\LINASHIDI_CXB_N_20160412.csv",encoding = 'gbk')
data = data[data['LINE_CODE'] ==674]
data = data[data['LOC_TREND'] ==1]
data['ON_STATION_TIME'] = pd.to_datetime(data['ON_STATION_TIME'])
data['on_hour'] = map(lambda x: x.hour, data['ON_STATION_TIME'] )
data = data[data['on_hour'] == 7]

on = data.groupby('ON_STATIONID')['ON_STATIONID'].count()
on =pd.DataFrame(on)
on['id'] = on.index
off = data.groupby('OFF_STATIONID')['OFF_STATIONID'].count()
off =pd.DataFrame(off)
off['id'] = off.index
on_off = pd.merge(off,on, on = 'id', how = 'outer')



data_hour = data.groupby('on_hour')

data = data[data['on_hour'] ==7]


def od_matrix1(data,station_num = 23):
    #计算出od矩阵
    l = []
    for i in range( station_num): 
        for j in range(station_num):
            l.append(len(data[(data['ON_STATIONID']==i)&(data['OFF_STATIONID']==j)]))
    l = np.array(l).reshape(station_num,station_num)
    return l


od_dic = data_hour.apply(od_matrix1)
#寻找异常矩阵

od_hour_sum = data_hour['on_hour'].agg('count')
#异常矩阵是7小时， 看7小时的上车量

data_7 = data[(data['on_hour'] == 7) & (data['ON_STATIONID'] == 0|1)]

#data_7 = dict(list(data_7.groupby(['VEHICLE_CODE','BANCIID'])))
#看7小时内发了多少班(10 班)

data_7 = data_7[['VEHICLE_CODE','BANCIID', 'on_hour']].drop_duplicates()
data_7 = pd.merge(data, data_7, left_on = ['VEHICLE_CODE','BANCIID'], right_on = ['VEHICLE_CODE','BANCIID'],how = 'inner')   
#提取了这七班车的数据    


data_7['ON_STATION_TIME'] = pd.to_datetime(data_7['ON_STATION_TIME'])

def huatu(data):
    l = []
    for i in range(23):
        l.append(data[data['ON_STATIONID'] ==i]['ON_STATION_TIME'].unique())
    return l


#a = huatu(data_7[(data_7['VEHICLE_CODE'] == 59696)&(data_7['BANCIID'] == 1)])

a = data_7.groupby(['VEHICLE_CODE','BANCIID']).apply(huatu)


data_7_plot = pd.DataFrame()
for i in range(len(a)):
    data_7_plot=pd.concat([data_7_plot,pd.DataFrame(a[i])],axis=1)

data_7_plot =  data_7_plot.fillna(method = 'ffill')


    


#################
data_71 = data_7[map(lambda x,y : x!=y,data_7['ON_STATIONID'],data_7['OFF_STATIONID']) ]











