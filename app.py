import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

@st.cache_data
def load_data():
    df = pd.read_csv("healthy_diet_macros_by_height_weight.csv")
    # Calculate water intake column
    df["Water_L"] = df["Weight_kg"] * 0.035
    return df

@st.cache_data
def train_model(df):
    X = df[["Height_cm", "Weight_kg"]]
    y = df[["Protein_g", "Carbs_g", "Fats_g", "Fiber_g", "Water_L"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def predict_macros(model, height, weight):
    sample_input = pd.DataFrame([[height, weight]], columns=["Height_cm", "Weight_kg"])
    prediction = model.predict(sample_input)[0]
    protein, carbs, fats, fiber, water = prediction
    calories = (protein * 4) + (carbs * 4) + (fats * 9)
    return protein, carbs, fats, fiber, water, calories

def main():
    st.title("🍎 Healthy Diet Macronutrients Predictor")

    df = load_data()
    model = train_model(df)

    st.sidebar.header("Enter Your Details")
    height = st.sidebar.number_input("Height (cm)", min_value=50, max_value=250, value=170)
    weight = st.sidebar.number_input("Weight (kg)", min_value=20, max_value=200, value=70)

    if st.sidebar.button("Predict Macros"):
        protein, carbs, fats, fiber, water, calories = predict_macros(model, height, weight)
        
        st.subheader("Predicted Macronutrients & Water Intake")
        st.write(f"**Protein (g):** {protein:.2f}")
        st.write(f"**Carbs (g):** {carbs:.2f}")
        st.write(f"**Fats (g):** {fats:.2f}")
        st.write(f"**Fiber (g):** {fiber:.2f}")
        st.write(f"**Water (L):** {water:.2f}")
        st.write(f"**Estimated Calories (kcal):** {calories:.2f}")

if __name__ == "__main__":
    main()
