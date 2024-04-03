import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

model = RandomForestClassifier()

dataset = pd.read_csv(r'C:\Users\Admin\Desktop\Road_Accident.csv')
dataset2=dataset.drop(columns=["Educational_level","Vehicle_driver_relation"])
df = pd.DataFrame(dataset2)
categorical_cols = ['Age_band_of_driver', 'Driving_experience','Sex_of_driver','Lanes_or_Medians', 'Types_of_Junction', 

                        'Light_conditions','Pedestrian_movement','Cause_of_accident','Type_of_collision','Road_surface_type','Weather_conditions','Vehicle_movement']
label_encoders = {}
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])
X=df.drop(columns='Accident_severity',axis=1)
y=df['Accident_severity']

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0) 
from sklearn.ensemble import RandomForestClassifier
model.fit(X_train, y_train) 
 
y_pred = model.predict(X_test)
model.fit(X_train, y_train)

  
def preprocess_input(data):
    
    label_encoders = {}  
    categorical_cols = ['Age_band_of_driver', 'Driving_experience','Sex_of_driver','Lanes_or_Medians', 'Types_of_Junction', 

                        'Light_conditions','Pedestrian_movement','Cause_of_accident','Type_of_collision','Road_surface_type','Weather_conditions','Vehicle_movement']
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])
    
    return data


def predict_severity(input_data):

    input_data = input_data[['Age_band_of_driver', 'Sex_of_driver', 'Driving_experience', 'Lanes_or_Medians', 'Types_of_Junction', 'Road_surface_type', 
                             'Light_conditions', 'Weather_conditions', 'Type_of_collision', 'Vehicle_movement', 'Pedestrian_movement', 'Cause_of_accident']]
    
    input_data = preprocess_input(input_data)
    prediction = model.predict(input_data)
    return prediction

def main():
    st.title("Accident Severity Prediction")
    st.write("Enter the details below to predict the severity of an accident.")


    Age_band_of_driver = st.selectbox("Age Band of Driver", ['18-30', '31-50', 'Under 18', 'Over 51', 'Unknown'])
    Driving_experience = st.selectbox("Driving_experience", ['1-2yr','Above 10yr','5-10yr','2-5yr','Unknown','No Licence','Below 1yr', 'unknown'])
    Sex_of_driver = st.selectbox("Sex_of_driver", ['Male','Female','Unknown'])
    Lanes_or_Medians= st.selectbox("Lanes_or_Medians", ['Unknown','Undivided Two way','other','Double carriageway (median)', 'One way','Two-way (divided with solid lines road marking)', 'Two-way (divided with broken lines road marking)'])
    Types_of_Junction = st.selectbox("Types_of_Junction", ['No junction','Y Shape','Crossing','O Shape','Other','Unknown','T Shape', 'X Shape'])
    Light_conditions = st.selectbox("Light_conditions", ['Daylight','Darkness - lights lit','Darkness - no lighting', 'Darkness - lights unlit'])
    Pedestrian_movement = st.selectbox("Pedestrian_movement", ['Not a Pedestrian', "Crossing from driver's nearside", 'Crossing from nearside - masked by parked or stationary vehicle', 'Unknown or other', 'Crossing from offside - masked by parked or stationary vehicle', 'In carriageway, stationary - not crossing  (standing or playing)', 'Walking along in carriageway, back to traffic', 'Walking along in carriageway, facing traffic', 'In carriageway, stationary - not crossing  (standing or playing) - masked by parked or stationary vehicle'])
    Cause_of_accident = st.selectbox('Cause_of_accident', ['Moving Backward', 'Overtaking', 'Changing lane to the left', 'Changing lane to the right', 'Overloading', 'Other', 'No priority to vehicle', 'No priority to pedestrian', 'No distancing', 'Getting off the vehicle improperly', 'Improper parking', 'Overspeed', 'Driving carelessly', 'Driving at high speed', 'Driving to the left', 'Unknown', 'Overturning', 'Turnover', 'Driving under the influence of drugs', 'Drunk driving'])
    Type_of_collision = st.selectbox('Type_of_collision', ['Collision with roadside-parked vehicles',
       'Vehicle with vehicle collision',
       'Collision with roadside objects', 'Collision with animals',
       'Other', 'Rollover', 'Fall from vehicles',
       'Collision with pedestrians', 'With Train', 'Unknown'])
    Road_surface_type = st.selectbox('Road_surface_type', ['Asphalt roads','Earth roads','Unknown', 'Asphalt roads with some distress','Gravel roads','Other'])
    Weather_conditions = st.selectbox("Weather_conditions", ['Normal','Raining','Raining and Windy','Cloudy','Other','Windy','Snow', 'Unknown','Fog or mist'])
    Vehicle_movement = st.selectbox("Vehicle_movement", ['Going straight','U-Turn','Moving Backward','Turnover','Waiting to go', 'Getting off','Reversing','Unknown','Parked','Stopping','Overtaking', 'Other','Entering a junction'])
    
    input_data = {
        'Age_band_of_driver': [Age_band_of_driver],
        'Driving_experience':[Driving_experience],
        'Sex_of_driver' :[Sex_of_driver],
        'Lanes_or_Medians':[Lanes_or_Medians],
        'Types_of_Junction':[Types_of_Junction],
        'Light_conditions':[Light_conditions],
        'Pedestrian_movement':[Pedestrian_movement],
        'Cause_of_accident':[Cause_of_accident],
        'Type_of_collision':[Type_of_collision],
        'Road_surface_type':[Road_surface_type],
        'Weather_conditions':[Weather_conditions],
        'Vehicle_movement':[Vehicle_movement]
       
    }

    input_df = pd.DataFrame(input_data)
    if st.button("Predict"):
        prediction = predict_severity(input_df)
        st.success(f"Predicted Severity: {prediction}")

if __name__ == "__main__":
    main()
