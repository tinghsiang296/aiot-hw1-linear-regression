
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- App Configuration ---
st.set_page_config(
    page_title="Interactive Regression Explorer",
    page_icon="📈",
    layout="wide"
)

# --- Main Application ---
st.title("📈 Interactive Linear Regression Explorer")
st.markdown("""
Use the sliders in the sidebar to generate data with different characteristics. 
The application will instantly train a linear regression model on the new data and show you how the model performs.
""")

# --- Sidebar for User Inputs ---
st.sidebar.header("Data Generation Parameters")

param_a = st.sidebar.slider(
    'Slope (a)', 
    min_value=50, max_value=500, value=150, step=10,
    help="Determines how much the price increases for each additional square foot of area."
)
param_b = st.sidebar.slider(
    'Intercept (b)', 
    min_value=0, max_value=100000, value=50000, step=5000,
    help="The base price of a house, regardless of its area."
)
param_noise = st.sidebar.slider(
    'Noise Level', 
    min_value=0, max_value=150000, value=40000, step=1000,
    help="The amount of random variation in the data. Higher noise makes the relationship less clear."
)
param_num_points = st.sidebar.slider(
    'Number of Data Points', 
    min_value=50, max_value=1000, value=250, step=50,
    help="The size of the dataset to generate."
)

# --- 1. Data Generation (with Caching) ---
@st.cache_data
def generate_data(a, b, noise, num_points):
    """Generates a DataFrame with house area and price based on user parameters."""
    np.random.seed(42) # for reproducibility
    area = np.random.rand(num_points) * 3000 + 500  # Area between 500 and 3500 sq ft
    price_noise = np.random.randn(num_points) * noise
    price = (a * area + b) + price_noise
    df = pd.DataFrame({'area': area, 'price': price})
    # Ensure price is not negative
    df['price'] = df['price'].clip(lower=0)
    return df

df = generate_data(param_a, param_b, param_noise, param_num_points)

# --- Main Panel ---
col1, col2 = st.columns((1, 1))

with col1:
    st.header("Data & Model Visualization")
    st.markdown("The chart below shows the generated data and the fitted regression line.")
    
    # --- Modeling and Visualization ---
    X = df[['area']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, color='#0072B2', alpha=0.5, label='Generated Data')
    ax.plot(X_test, y_pred, color='#D55E00', linewidth=3, label='Fitted Regression Line')
    ax.set_title('Price vs. Area', fontsize=16)
    ax.set_xlabel('Area (sq ft)', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

with col2:
    st.header("Model Evaluation")
    st.markdown("Here's how the model performed on the test data.")
    
    # Evaluate model
    intercept = model.intercept_
    slope = model.coef_[0]
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    st.subheader("Learned Linear Relationship")
    st.latex(f"Price = {slope:.2f} \times Area + {intercept:,.2f}")
    
    st.subheader("Performance Metrics")
    st.metric(label="R-squared (R²)", value=f"{r2:.4f}")
    st.metric(label="Root Mean Squared Error (RMSE)", value=f"${rmse:,.2f}")

    with st.expander("About These Metrics"):
        st.markdown("""
        - **R-squared (R²)**: Represents the proportion of the variance in the dependent variable that is predictable from the independent variable(s). A value of 1.0 means the model perfectly explains the data.
        - **RMSE**: Measures the average difference between the values predicted by the model and the actual values. A lower RMSE is better.
        """)

# --- Data Inspector ---
with st.expander("Inspect the Generated Data"):
    st.dataframe(df)
