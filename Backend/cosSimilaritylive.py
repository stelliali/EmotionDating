from numpy import dot
from numpy.linalg import norm
import csv
import requests

def cosSim():  
    with open('D:/ProgramData/AC/EmotionDating/Backend/emotionvalues.csv', 'r') as vec:
            reader = list(csv.reader(vec))
            for index, rows in enumerate(reader):
                if index == 1:
                    v1_1 = rows
                if index == 2:
                    v1_2 = rows
                if index == 3:
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
                if index == 45:
                    v3_1 = rows  
                if index == 46:
                    v3_2 = rows  
                if index == 47:
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

    ave1 = str(int(round(ave1 * 100, 0))) + "%"
    ave2 = str(int(round(ave2 * 100, 0))) + "%"
    ave3 = str(int(round(ave3 * 100, 0))) + "%"
    # return ave1, ave2, ave3
    print(ave1, ave2, ave3)


    results = {
        "Jonas": ave1,
        "Max": ave2,
        "Philipp": ave3
    }
    r = requests.post("http://localhost:5000/upload_results", json=results)
    print(r.status_code)
    with open('D:/ProgramData/AC/EmotionDating/Backend/result-fe/ave1.txt','a') as f1:
        f1.write(ave1)
    with open('D:/ProgramData/AC/EmotionDating/Backend/result-fe/ave2.txt','a') as f2:
        f2.write(ave2)
    with open('D:/ProgramData/AC/EmotionDating/Backend/result-fe/ave3.txt','a') as f3:
        f3.write(ave3)