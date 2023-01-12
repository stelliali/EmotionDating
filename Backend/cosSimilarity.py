from numpy import dot
from numpy.linalg import norm

List1 = [0.53,0.0,0.04,0.31,0.08,0.02,0.01]
List2 = [0.09,0.0,0.07,0.12,0.59,0.0,0.12]
result = dot(List1, List2)/(norm(List1)*norm(List2))
print(result)