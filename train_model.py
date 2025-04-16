import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib   # Modeli kaydetmek için kullanılır


# Veriyi oku
data = pd.read_csv('Salary_Data.csv')  

# Değişkenleri ayır
x = data[['YearsExperience']] #2D
y = data['Salary']            #1D

# Eğitim ve test seti
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

# Modeli oluştur ve eğit
model = LinearRegression()
model.fit(x_train,y_train)

# Modeli kaydet
joblib.dump(model,'linear_model.pkl')  # .pkl dosyasını oluşturur

print("Model başarıyla eğitildi ve linear_model.pkl olarak kaydedildi.")


