import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


dataset = pd.read_csv(r'C:\Users\Admin\Desktop\Road_Accident.csv')
dataset=dataset.drop(columns=["Educational_level","Vehicle_driver_relation"])
dataset
categorical_cols = ['Age_band_of_driver', 'Sex_of_driver', 
                    'Driving_experience', 'Lanes_or_Medians', 'Types_of_Junction', 'Road_surface_type', 
                    'Light_conditions', 'Weather_conditions', 'Type_of_collision', 'Vehicle_movement', 
                    'Pedestrian_movement', 'Cause_of_accident']
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    dataset[col] = label_encoders[col].fit_transform(dataset[col])

X = dataset.drop('Accident_severity',axis = 1)
y = dataset['Accident_severity']
    
def main():
    st.title('Road Accident Prediiction and Prevention')
    st.subheader('Enter the details :')

    age=st.selectbox("**Age_band_of_driver :**",['18-30','31-50','over 51','under 18','unknown'],index=1)

    sex=st.selectbox("**Sex_of_driver :**",['Male','Female','Unknown'],index=1)

    driving_exp = st.selectbox("**Driving_Experience :**",['1-2yr' ,'Above 10yr' ,'5-10yr' ,'2-5yr', 'Unknown' ,'No Licence' ,'Below 1yr','unknown'],index =1)

    lanes = st.selectbox("**Lanes :**",['Unknown' ,'Undivided Two way' ,'other' ,'Double carriageway (median)','One way' ,'Two-way (divided with solid lines road marking)','Two-way (divided with broken lines road marking)'],index = 1)

    jun_type = st.selectbox("**Type_of_Junction :**",['No junction' ,'Y Shape' ,'Crossing', 'O Shape', 'Other' ,'Unknown' ,'T Shape','X Shape'],index = 1)
    
    road_type=st.selectbox("**Road_surface_type :**",['Asphalt roads','Asphalt roads with some distress','Earth roads','Gravel roads','Others','Unknown'],index=1)

    light_con = st.selectbox("**Light_condition :**",['Daylight', 'Darkness - lights lit', 'Darkness - no lighting','Darkness - lights unlit'], index = 1)

    weather_con= st.selectbox("**Weather_condition :**",['Cloudy','Fog or mist','Normal','other','Raining','Raining and windy','Snow','Unknown','Windy'],index=1)

    type_of_col=st.selectbox("**Type_of_Collision :**",['Collision with animals','Collision with pedestrians','Collision with roadside objects','Collision with roadside-parked vehicles','Fall from vehicles','Others','Rollover','UnKnown','Vechicle with vechile collision','With Train'],index=1)

    veh_mov = st.selectbox("**Vehicle_Movement :**",['Going straight','U-Turn', 'Moving Backward' ,'Turnover', 'Waiting to go''Getting off', 'Reversing' ,'Unknown' ,'Parked' ,'Stopping' ,'Overtaking','Other', 'Entering a junction'], index =1)

    ped_mov = st.selectbox("**Pedestrian_Movement :**",['Not a Pedestrian' ,"Crossing from driver's nearside",'Crossing from nearside - masked by parked or statioNot a Pedestrianry vehicle', 'Unknown or other','Crossing from offside - masked by  parked or statioNot a Pedestrianry vehicle','In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing)','Walking along in carriageway, back to traffic','Walking along in carriageway, facing traffic','In carriageway, statioNot a Pedestrianry - not crossing  (standing or playing) - masked by parked or statioNot a Pedestrianry vehicle'], index =1)

    cause = st.selectbox("**Cause_of_accident :**",['Moving Backward' ,'Overtaking' ,'Changing lane to the left',
 'Changing lane to the right', 'Overloading', 'Other',
 'No priority to vehicle' ,'No priority to pedestrian' ,'No distancing',
 'Getting off the vehicle improperly' ,'Improper parking' 'Overspeed',
 'Driving carelessly' ,'Driving at high speed' ,'Driving to the left',
 'Unknown' ,'Overturning' ,'Turnover' ,'Driving under the influence of drugs',
 'Drunk driving'], index =1)
    a = st.button("Predict")
    if a ==True:
        arr = np.array([age,sex,driving_exp,lanes,jun_type,road_type,light_con,weather_con,type_of_col,veh_mov,ped_mov, cause])
        df = pd.DataFrame(arr)
        data = df.T
        data = data.astype(int)
        cat_col = ['age','sex','driving_exp','lanes','jun_type','road_type','light_con','weather_con','type_of_col','veh_mov','ped_mov', 'cause']
        label_encoders = {}
        for col in cat_col:
            label_encoders[col] = LabelEncoder()
            data[col] = label_encoders[col].fit_transform(data[col])

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
        random_forest = RandomForestClassifier(random_state=42)
        random_forest.fit(X_train, y_train)
        var = random_forest.predict(data)
        if var == 1| var == 0:
             st.write(f"The severity is {var}.The Cause of Accident includes Improper Parking and Turning over ")

        else:
             st.write(f"The Severity is {var}. The Cause of Accident includes Changing the Lanes and Rash Driving")

if __name__== '__main__':
    main()

    
