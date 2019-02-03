import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures  
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
data = pd.read_csv('datap.csv')
data.drop('id',axis = 1,inplace=True)


#dic = {};

#for i in range (200):
    
pf = PolynomialFeatures(degree = 10)  

x = data.iloc[:,:1]
y = data.iloc[:,1:2]

x2 = pf.fit_transform(x)

model = reg.fit(x2, y)
yfit = model.predict(x2)

plt.scatter(x, y)
plt.plot(x, yfit);
    
 #   dic[i] = reg.score(x2,y)
 #  print (reg.score(x2,y))
    
#df = pd.DataFrame()
#df['key'] = dic.keys()
#df['value'] = dic.values()
#
#better_degree = df.sort_values('value',ascending=False)