import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

client = Groq(api_key=os.getenv('groq_APIkey'))


def train_and_save():
    print('Melatih model')
    df = pd.read_csv("data/Teen_Mental_Health_Dataset.csv")
    leGender = LabelEncoder() # untuk mengubah data gender menjadi angka
    df['gender'] = leGender.fit_transform(df['gender']) # mengubah data gender
    df['social_interaction_level'] = df['social_interaction_level'].map({'low':1,',medium':2,'high':3})

    df = pd.get_dummies(df,columns=['platform_usage'])

    x= df.drop('depression_label',axis = 1)
    y = df['depression_label']

    x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=.2,random_state = 42)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    model = RandomForestClassifier(n_estimators = 100, random_state = 42, max_depth=15, n_jobs=-1)
    model.fit(x_train_scaled, y_train)
    print(model.score(scaler.transform(x_test),y_test))
    joblib.dump(model,'model_mental.pkl')
    joblib.dump(scaler,'scaler_mental.pkl')
    joblib.dump(leGender,'leGender.pkl')
    joblib.dump(x.columns.tolist(),'feature_columns.pkl')

    print('Train selesai dan model disimpan')

def predictWithLLM():
    model = joblib.load('model_mental.pkl')
    scaler = joblib.load('scaler_mental.pkl')
    leGender = joblib.load('leGender.pkl')
    # print(featursName := joblib.load('feature_columns.pkl'))

    print("Data prediksi".center(50,'-'))
    age = int(input('Masukkan umur : '))
    gender = input('Gender (male/female) : ').lower()
    sosmedHours = float(input('Jam Sosmed perhari : '))
    platfrom = input('Platform (Instagram/Tiktok/Both) : ')
    sleep = float(input('Jam Tidur : '))
    screenTime = float(input('Screem Time sebelum tidur : '))
    gpa = float(input('IPK/performa akademik : '))
    phys = float(input('Jam olahraga/Hari : '))
    social = input('Level interaksi (Low/Medium/High) : ').lower()
    stress= int(input('Skala Stress (1-10) : '))
    anxiety =int(input ('Skala Cemas (1-10) : '))
    addict = int(input('Skala Kecanduan Sosmed (1-10) : '))

    genderEncoded = leGender.transform([gender])[0]
    socialEncoded = {'low':1,'medium':2,'high':3}.get(social,2)
    p_both,p_insta,p_tik = 0,0,0
    if platfrom.lower() == 'both': p_both = 1
    elif platfrom.lower() == 'instagram': p_insta = 1
    elif platfrom.lower() == 'tiktok': p_tik = 1

    featursName = joblib.load('feature_columns.pkl')
    input_dict = {
    'age': age,
    'gender': genderEncoded,
    'daily_social_media_hours': sosmedHours,
    'sleep_hours': sleep,
    'screen_time_before_sleep': screenTime,
    'academic_performance': gpa,
    'physical_activity': phys,
    'social_interaction_level': socialEncoded,
    'stress_level': stress,
    'anxiety_level': anxiety,
    'addiction_level': addict,
    'platform_usage_Both': p_both,
    'platform_usage_Instagram': p_insta,
    'platform_usage_TikTok': p_tik
    }
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[featursName]  
    scaleData = scaler.transform(input_df)
    pred = model.predict(scaleData)[0]

    textPred = 'Beresiko Depresi' if pred == 1 else 'Kondisi Normal'

    print(f'\n[HASIL PREDIKSI DATA MINING] : {textPred}')
    print('meminta penjelasan ke LLM...')


    prompt = f'''
kamu adalah seorang konselor kesehatan mental AI
berdasarkan data mining, mahasiswa ini memiliki status kesehatan mental : {textPred}
Data : umur {age},Tidur {sleep} jam, Screen Time {screenTime},Stress level {stress}/10.
Tolong jelaskan secara singkat kenapa gaya hidup ini berpengaruh pada kesehatan mentalnya dan berikan 5 saran ramah. Gunakan bahasa indonesia yang santai dan mudah dimengerti oleh mahasiswa
'''
    
    chat_response = client.chat.completions.create(
        messages = [{'role':'user','content':prompt}],
        model = 'llama-3.3-70b-versatile',
    )

    print('\n[ANALISIS FROM LLM]')
    print(chat_response.choices[0].message.content)


if __name__ == '__main__':
    try:
        joblib.load('model_mental.pkl')
    except: 
        train_and_save()
    predictWithLLM()    

