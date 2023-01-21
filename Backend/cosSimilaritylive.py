from numpy import dot
from numpy.linalg import norm
import csv

def cosSim():  
    with open('D:/ProgramData/AC/EmotionDating/Backend/emotionvalues.csv', 'r') as vec:
            reader = list(csv.reader(vec))
            for index, rows in enumerate(reader):
                if index == 33:
                    v1_1 = rows
                if index == 34:
                    v1_2 = rows
                if index == 35:
                    v1_3 = rows
                
        # person 1
    with open('D:/ProgramData/AC/EmotionDating/Backend/user_study_vecs.csv', 'r') as vec:
            reader = list(csv.reader(vec))
            for index, rows in enumerate(reader):
                if index == 3:
                    v2_1 = rows  
                if index == 4:
                    v2_2 = rows  
                if index == 5:
                    v2_3 = rows    
        # person 2
                if index == 39:
                    v3_1 = rows  
                if index == 40:
                    v3_2 = rows  
                if index == 41:
                    v3_3 = rows     
        # person 3
                if index == 56:
                    v4_1 = rows  
                if index == 57:
                    v4_2 = rows  
                if index == 58:
                    v4_3 = rows        

    a_1 = [float(x) for x in v1_1]   
    a_2 = [float(x) for x in v1_2]
    a_3 = [float(x) for x in v1_3]

    b_1 = [float(x) for x in v2_1]   
    b_2 = [float(x) for x in v2_2]
    b_3 = [float(x) for x in v2_3]

    c_1 = [float(x) for x in v3_1]   
    c_2 = [float(x) for x in v3_2]
    c_3 = [float(x) for x in v3_3]

    d_1 = [float(x) for x in v4_1]   
    d_2 = [float(x) for x in v4_2]
    d_3 = [float(x) for x in v4_3]


    cos_sim_1 = dot(a_1, b_1)/(norm(a_1)*norm(b_1))
    cos_sim_2 = dot(a_1, c_1)/(norm(a_1)*norm(c_1))
    cos_sim_3 = dot(a_1, d_1)/(norm(a_1)*norm(d_1))
    ave1 = (cos_sim_1 + cos_sim_2 + cos_sim_3)/3

    cos_sim_4 = dot(a_2, b_2)/(norm(a_2)*norm(b_2))
    cos_sim_5 = dot(a_2, c_2)/(norm(a_2)*norm(c_2))
    cos_sim_6 = dot(a_2, d_2)/(norm(a_2)*norm(d_2))
    ave2 = (cos_sim_4 + cos_sim_5 + cos_sim_6)/3

    cos_sim_7 = dot(a_3, b_3)/(norm(a_3)*norm(b_3))
    cos_sim_8 = dot(a_3, c_3)/(norm(a_3)*norm(c_3))
    cos_sim_9 = dot(a_3, d_3)/(norm(a_3)*norm(d_3))
    ave3 = (cos_sim_7 + cos_sim_8 + cos_sim_9)/3
        # l.append(cos_sim)
    # return ave1, ave2, ave3
    print(ave1, ave2, ave3)