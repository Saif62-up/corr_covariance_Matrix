# corr_covariance_Matrix
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns

df = pd.read_csv("/kaggle/input/datasets/spscientist/students-performance-in-exams/StudentsPerformance.csv")
df.head() 

data = df[["math score", "reading score", "writing score"]]
print(data)
correlation = data.corr()
print(correlation) 

sns.heatmap(correlation, annot=True, cmap="coolwarm", vmin=-1 ,vmax=1) #display all the values inside the heatmap 
plt.title("Correlation Matrix")
plt.show() 

sns.pairplot(df[['math score', 'reading score', 'writing score']])
plt.show() 

df["gender_num"] = df["gender"].map({
    "female": 0, 
    "male": 1
})
sns.pairplot(df[['gender_num','math score', 'reading score', 'writing score']], hue ="gender_num")
plt.show()

df_num = df[["gender_num","math score", "reading score", "writing score"]]

sns.heatmap(
    df_num.corr(),
    annot=True,
    cmap="Blues"
)

plt.title("Heatmap of Four Features")
plt.show() 








