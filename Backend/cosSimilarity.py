from numpy import dot
from numpy.linalg import norm

import csv

l = []
totalAve = [0.882167609706316, 0.7004884563419405, 0.5309750422771653, 0.641727334981321, 0.7834747699543595, 0.4750877451627905, 0.6342156695914987, 0.819629974402912, 0.5998219025088113, 0.7944509942098629]
def cos(i):
    with open('D:/ProgramData/AC/EmotionDating/Backend/user_study_vecs.csv', 'r') as vec:
        reader = list(csv.reader(vec))
        for index, rows in enumerate(reader):
            if index == i:
                v1 = rows
            if index == i+17:
                v2 = rows
                

    a = [float(x) for x in v1]   
    b = [float(x) for x in v2]         
    cos_sim = dot(a, b)/(norm(a)*norm(b))
    # l.append(cos_sim)
    return cos_sim

    # with open('D:/ProgramData/AC/EmotionDating/Backend/user_study_re.csv',  'w', encoding='UTF8') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(l)

if __name__ == "__main__":
    # n = 9*35
    # for i in range(n+3, n+18):
    #     a = cos(i)
    #     l.append(a)
    #     ave = sum(l)/len(l)
    #     print(l)
    #     print(ave)
        
    totalave = sum(totalAve)/len(totalAve)
    print(totalave)
    # 0.6862039499136978
