import sys
sys.path.append('..')
import numpy as np
import pandas as pd
from dataprocess import FeatureSelect as fs

#get the wind direct, old version
def windDriectold(pooling_mat):

    direct_mat = np.zeros((15, 4, 6, 6))
    xx=np.arange(15)
    yy=np.arange(15)
    windDrirct=np.zeros(4)
    sumx=0
    sumy=0

    for i in range(15):
        wind_mat = pooling_mat[i, 3]  #forth level
        raw, column = wind_mat.shape
        positon = np.argmax(wind_mat)
        m, n = divmod(positon, column)#position of the max value point
        xx[i]=m
        yy[i]=n

    for i in range(0,3):
        sumx+=xx[i]
        sumy+=yy[i]

    sumx=sumx/3
    sumy=sumy/3

    for i in range(3,15):
        if(xx[i]>sumx): #north wind
            windDrirct[2]+=1
        if(xx[i]<sumx):
            windDrirct[3]+=1
        if(yy[i]>sumy):#west wind
            windDrirct[0]+=1
        if (yy[i] <sumy):
            windDrirct[1] += 1

    direct=np.argmax(windDrirct)

    for i in range(15):

        for j in range(4):

            temp_mat = pooling_mat[i,j]

            if(direct == 0):
                if(windDrirct[2]>6):#west north
                    direct_mat1=temp_mat[1:7,0:6]
                elif(windDrirct[3]>6):#west south
                    direct_mat1=temp_mat[3:9,0:6]
                else:
                    direct_mat1=temp_mat[2:8,0:6]

            elif(direct ==1):
                if (windDrirct[2] > 6):  # east
                    direct_mat1 = temp_mat[1:7,4:10]
                elif (windDrirct[3] > 6):  #east south
                    direct_mat1 = temp_mat[3:9,4:10]
                else:
                    direct_mat1 = temp_mat[2:8,4:10]

            elif(direct==2):
                if (windDrirct[0] > 6):  # west north
                    direct_mat1 = temp_mat[0:6,1:7]
                elif (windDrirct[1] > 6):  #east north
                    direct_mat1 = temp_mat[0:6,3:9]
                else:
                    direct_mat1 = temp_mat[0:6,2:8]

            elif (direct == 3):
                if (windDrirct[0] > 6):  # west south
                    direct_mat1 = temp_mat[4:10,1:7]
                elif (windDrirct[1] > 6):  #east south
                    direct_mat1 = temp_mat[4:10,3:9]
                else:
                    direct_mat1 = temp_mat[4:10,2:8]
            else:
                direct_mat1 = temp_mat[2:8, 2:8]

            direct_mat[i][j]=direct_mat1

    return direct_mat.reshape(2160)

# get wind direct, new version
def windDriect1ave(pooling_mat):

    direct_mat = np.zeros((15, 4, 6, 6))
    xx=np.arange(15)
    yy=np.arange(15)
    windDrirct=np.zeros(4)

    for i in range(15):
        wind_mat = pooling_mat[i, 3]
        raw, column = wind_mat.shape
        sumx=0
        sumy=0
        temp=wind_mat.reshape(1,100)
        paramsort = np.argsort(-temp)

        for j in range(5):
            sumx+=paramsort[0][j]//column
            sumy+=paramsort[0][j]%column

        xx[i]=sumx//5
        yy[i]=sumy//5

    for i in range(1,15):
        if (xx[i] > xx[0]):
            windDrirct[2] += 1
        if (xx[i] < xx[0]):
            windDrirct[3] += 1
        if (yy[i] > yy[0]):
            windDrirct[0] += 1
        if (yy[i] < yy[0]):
            windDrirct[1] += 1

    direct=np.argmax(windDrirct)

    for i in range(15):

        for j in range(4):

            temp_mat = pooling_mat[i,j]

            if(direct == 0):
                if(windDrirct[2]>7):
                    direct_mat1 = temp_mat[1:7,0:6]
                elif(windDrirct[3]>7):
                    direct_mat1 = temp_mat[3:9,0:6]
                else:
                    direct_mat1 = temp_mat[2:8,0:6]

            elif(direct == 1):
                if (windDrirct[2] > 7):
                    direct_mat1 = temp_mat[1:7,4:10]
                elif (windDrirct[3] > 7):
                    direct_mat1 = temp_mat[3:9,4:10]
                else:
                    direct_mat1 = temp_mat[2:8,4:10]

            elif(direct==2):
                if (windDrirct[0] > 7):
                    direct_mat1 = temp_mat[0:6,1:7]
                elif (windDrirct[1] > 7):
                    direct_mat1 = temp_mat[0:6,3:9]
                else:
                    direct_mat1 = temp_mat[0:6,2:8]

            elif (direct == 3):
                if (windDrirct[0] > 7):
                    direct_mat1 = temp_mat[4:10,1:7]
                elif (windDrirct[1] > 7):
                    direct_mat1 = temp_mat[4:10,3:9]
                else:
                    direct_mat1 = temp_mat[4:10,2:8]

            else:
                direct_mat1 = temp_mat[2:8, 2:8]

            direct_mat[i][j]=direct_mat1

    return direct_mat

#extend data in eight directions
def extendData(pooling_mat):

    return_value=np.zeros((8, 15, 4, 6, 6))

    for i in range(15):

        for j in range(4):

            temp_mat = pooling_mat[i,j]

            #topdown
            temp_mat1 = np.flipud(temp_mat)
            #leftright
            temp_mat2 = np.fliplr(temp_mat)
            #topdown-leftright
            temp_mat3 = np.fliplr(temp_mat1)
            #turn 90
            temp_mat4 = np.rot90(temp_mat)
            #turn 270
            temp_mat5 = np.rot90(temp_mat, 3)
            #turn 90-topdown
            temp_mat6 = np.flipud(temp_mat4)
            #tun 90-leftright
            temp_mat7 = np.fliplr(temp_mat4)

            return_value[0][i][j] = temp_mat
            return_value[1][i][j] = temp_mat1
            return_value[2][i][j] = temp_mat2
            return_value[3][i][j] = temp_mat3
            return_value[4][i][j] = temp_mat4
            return_value[5][i][j] = temp_mat5
            return_value[6][i][j] = temp_mat6
            return_value[7][i][j] = temp_mat7

    return return_value

# convolution 5*5
def train_convolution(line,data_type):

    cate = line[0].split(',')
    id_label = [cate[0]]

    if data_type == 'train':
        id_label.append(float(cate[1]))

    record = [int(cate[2])]
    length = len(line)

    for i in range(1,length):
        record.append(int(line[i]))

    mat = np.array(record).reshape(15,4,101,101)
    con_mat = np.zeros((15,4,20,20))

    for i in range(15):

        for j in range(4):

            temp_mat = mat[i,j]
            temp_mat = np.delete(temp_mat,0,axis=0)
            temp_mat = np.delete(temp_mat,0,axis=1)

            for m in range(20):

                for n in range(20):

                    avg_mat = temp_mat[m*5:m*5+5,n*5:n*5+5]
                    con_mat[i,j,m,n] = np.average(avg_mat)

    return id_label, con_mat

# max pooling
def max_pooling(con_mat):

    pooling_mat = np.zeros((15,4,10,10))#10,10

    for i in range(15):
        for j in range(4):

            temp_mat = con_mat[i, j]

            for m in range(10):#10
                for n in range(10):#10
                    max_mat = temp_mat[2*m:2*m+2,n*2:n*2+2]
                    pooling_mat[i,j,m,n] = np.max(max_mat)

    return pooling_mat

# process wind data
def dataprocess(filename, data_type,windversion):

    header_list = ['id']
    for i in range(432):#6000
        feature = 'thxy_' + str(i+1)
        header_list.append(feature)

    if data_type == 'train':
        header_list += ['label']

    df = pd.DataFrame(columns=header_list)

    with open(filename) as fr:

        if data_type == 'train':
            sample_num = 10000
        elif data_type == 'testB':
            sample_num = 2000

        for i in range(1, sample_num + 1):

            line = fr.readline().strip().split(' ')
            id_label, con_mat = train_convolution(line, data_type)
            pooling_mat = max_pooling(con_mat)

            if windversion == 'new' and data_type == 'train':

                eightValue = extendData(windDriect1ave(pooling_mat))

                for j in range(8):
                    value = eightValue[j].reshape((15, 4, 36))
                    value = fs.slice_h(value, time=15, m=6, n=6, h=1, asd=1)
                    value = fs.slice_t(value, time_sum=15, time_slice=12, m=6, n=6, h=1)
                    simp = list(value)
                    temp = [id_label[0]]
                    temp += simp

                    if data_type == 'train':
                        temp += [id_label[1]]

                    print(temp)
                    df_temp = pd.DataFrame([temp], columns=header_list)
                    df = df.append(df_temp, ignore_index=True)

            else:

                if windversion =='old':
                    value = windDriectold(pooling_mat)
                else:#new test
                    value = windDriect1ave(pooling_mat).reshape(2160)

                value = value.reshape((15, 4, 36))
                value = fs.slice_h(value, time=15, m=6, n=6, h=1, asd=1)
                value = fs.slice_t(value, time_sum=15, time_slice=12, m=6, n=6, h=1)
                simp = list(value)
                temp = [id_label[0]]
                temp += simp

                if data_type == 'train':
                    temp += [id_label[1]]

                print(temp)
                df_temp = pd.DataFrame([temp], columns=header_list)
                df = df.append(df_temp, ignore_index=True)

#        print(df.head())

        if windversion == 'old':            
            df.to_csv('/home/Team4/Team4/dataset/'+data_type+'_'+windversion+'_wind_4240.csv',index=False,float_format='%.3f')
        else:
            df.to_csv('/home/Team4/Team4/dataset/'+data_type+'_'+windversion+'_wind_1ave_8extend.csv',index=False,float_format='%.3f')

    return df

#the path of train set & the path of testB set
trainfile = 'data_new/CIKM2017_train/train.txt'
testBfile = 'data_new/CIKM2017_testB/testB.txt'

#produces the train set of 'old' wind
#dataprocess(trainfile, data_type='train',windversion='old')
#proceces the testB set of 'old' wind
#dataprocess(testBfile, data_type='testB',windversion='old')

#produces the extended train set
#dataprocess(trainfile, data_type='train',windversion='new')
#produces the extended testB set
#dataprocess(testBfile, data_type='testB',windversion='new')
