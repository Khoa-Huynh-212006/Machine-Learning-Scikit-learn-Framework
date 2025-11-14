import pandas as pd 
import numpy as np

medical_charges_url = 'https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv'
from urllib.request import urlretrieve
urlretrieve(medical_charges_url, 'medical.csv')

medical_df = pd.read_csv('medical.csv')

smoker_code = {'no':0, 'yes': 1 }
medical_df['smoker_code'] = medical_df.smoker.map(smoker_code)
sex_code ={'male':1, 'female':0}
medical_df['sex_code'] = medical_df.sex.map(sex_code)

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder (sparse_output=False, handle_unknown= 'ignore') 

categorical_cols = medical_df.select_dtypes('object').columns.to_list()
if 'sex' in categorical_cols:
    categorical_cols.remove('sex')
if 'smoker' in categorical_cols:
    categorical_cols.remove('smoker')
numeric_cols = medical_df.select_dtypes(include=np.number).columns.tolist()
if 'charges' in numeric_cols:
    numeric_cols.remove('charges')
encoder.fit(medical_df[categorical_cols])

encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
medical_df[encoded_cols] = encoder.transform(medical_df[categorical_cols].fillna('Unknown'))

from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(medical_df, test_size= 0.2, random_state= 42)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
input_cols = numeric_cols + encoded_cols
inputs = train_df[input_cols]
targets = train_df['charges']

model.fit(inputs, targets)
predictions = model.predict(inputs)
predictions

def rmse(targets, predictions):
    return np.sqrt(np.mean(np.square(targets-predictions)))
rmse (targets,  predictions)

inputs_test = test_df[input_cols]
targets_test = test_df['charges']
predictions_test = model.predict(inputs_test)
predictions_test
rmse_test = rmse(targets_test, predictions_test)

weights_df =pd.DataFrame({
    'feature': np.append(input_cols , 1),
    'weight': np.append(model.coef_, model.intercept_)
})
weights_df

from sklearn.metrics import mean_squared_error

def try_model_interactive():
    print("Nhập thông tin người dùng để dự đoán chi phí bảo hiểm y tế:")

    try:
        age = int(input("Tuổi (age): "))
        bmi = float(input("BMI (Chỉ số khối cơ thể): "))
        children = int(input("Số lượng con (children): "))
        smoker_code = int(input("Hút thuốc? (có: 1, không: 0): "))
        sex_code = int(input("Giới tính (nam: 1, nữ: 0): "))

        print("Khu vực sinh sống (chọn đúng 1):")
        region_northeast = int(input("Northeast? (có: 1, không: 0): "))
        region_northwest = int(input("Northwest? (có: 1, không: 0): "))
        region_southeast = int(input("Southeast? (có: 1, không: 0): "))
        region_southwest = int(input("Southwest? (có: 1, không: 0): "))

        input_list = [age, bmi, children, smoker_code, sex_code,region_northeast, region_northwest, region_southeast, region_southwest]

        if sum([region_northeast, region_northwest, region_southeast, region_southwest]) != 1:
            print("Lỗi: Chỉ được chọn đúng một khu vực.")
            return

        input_array = np.array([input_list])
        predicted_charge = model.predict(input_array)[0]
        print("Dự đoán chi phí (charges):", round(predicted_charge, 2))

        predictions_test = model.predict(test_df[input_cols])
        targets_test = test_df['charges']
        rmse_test = rmse(targets_test, predictions_test)
        print("Sai số trung bình căn (RMSE - test):", round(rmse_test, 2))

        return predicted_charge, rmse_test

    except Exception as e:
        print("Lỗi khi nhập dữ liệu:", str(e))

try_model_interactive()

