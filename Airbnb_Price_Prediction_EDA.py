# ==============================================================================
# STEP 1: IMPORT PRODUCTION LIBRARIES & CONFIGURE RUNTIME ENVIRONMENT
# ==============================================================================
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

# Set plotting cosmetics
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ==============================================================================
# STEP 2: LOAD AND INGEST DATASET SAFELY
# ==============================================================================
data_path = Path(__file__).resolve().parents[1] / "data" / "AB_NYC_2019.csv"
df = pd.read_csv(data_path)
print("[+] Raw Database Records Ingested:", df.shape)

# ==============================================================================
# STEP 3: RIGOROUS DATA CLEANING & OUTLIER HANDLING
# ==============================================================================
# Checking for missing values and handling systematically
print("\n--- Missing Records Before Cleaning ---")
print(df.isnull().sum())

# Impute missing values for reviews per month safely to 0
df['reviews_per_month'] = df['reviews_per_month'].fillna(0.0)

# Drop structural tracking elements that do not contribute variance to price prediction
df.drop(['id', 'name', 'host_name', 'last_review'], axis=1, inplace=True, errors='ignore')

# Identify price variations, boundaries and eliminate pricing outliers
# Filtering out extreme zero-values and structural 99th-percentile spikes
df = df[(df['price'] >= 10) & (df['price'] <= df['price'].quantile(0.99))]

# Address structural skewness of Target Variable by applying log-transformation
df['log_price'] = np.log1p(df['price'])

# ==============================================================================
# STEP 4: EXPLORATORY DATA ANALYSIS (EDA) VISUALIZATIONS
# ==============================================================================
# 1. Price Distribution across locations (Neighbourhood Groups)
plt.figure()
sns.boxplot(x='neighbourhood_group', y='price', data=df, palette='Set2')
plt.title('Price Variation Across NYC Neighbourhood Groups')
plt.xlabel('Neighbourhood Group')
plt.ylabel('Price ($)')
plt.show()

# 2. Price Variations by Room Type
plt.figure()
sns.violinplot(x='room_type', y='price', data=df, palette='Pastel1')
plt.title('Impact of Room Type on Booking Pricing Tiers')
plt.xlabel('Room Type')
plt.ylabel('Price ($)')
plt.show()

# 3. Spatial Price Map Correlation Distribution
plt.figure()
sns.scatterplot(x='longitude', y='latitude', hue='price', data=df, palette='viridis', alpha=0.5)
plt.title('Spatial Heatmap Layer of Booking Pricing Density Across Coordinates')
plt.show()

# ==============================================================================
# STEP 5: REGRESSION MODELLING & PIPELINE ENGINE 
# ==============================================================================
X = df.drop(['price', 'log_price'], axis=1)
y = df['log_price']

num_features = ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 
                'reviews_per_month', 'calculated_host_listings_count', 'availability_365']
cat_features = ['neighbourhood_group', 'room_type']

# Establish clean Column Transformer preprocessing frameworks
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train multiple regression models concurrently inside standard Pipelines
models = {
    "Linear Regression": LinearRegression(),
    "XGBoost Regressor": xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
}

print("\n--- Model Performance Pipeline Execution ---")
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    
    # Revert target prediction transformations from log-space back to raw dollar indices
    actual_y = np.expm1(y_test)
    actual_preds = np.expm1(preds)
    
    print(f"\n[{name} Evaluation Scores]")
    print(f" -> Mean Absolute Error (MAE): ${mean_absolute_error(actual_y, actual_preds):.2f}")
    print(f" -> Root Mean Squared Error (RMSE): ${np.sqrt(mean_squared_error(actual_y, actual_preds)):.2f}")
    print(f" -> R2 Score Metrics: {r2_score(y_test, preds):.4f}")
