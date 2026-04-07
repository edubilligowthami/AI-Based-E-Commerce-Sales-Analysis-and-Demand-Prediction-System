import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ==============================
# STEP 1: LOAD DATA
# ==============================
df = pd.read_csv("ecommerce_data.csv")

# ==============================
# STEP 2: DATA PROCESSING
# ==============================
df["Revenue"] = df["Price"] * df["Quantity"]

print("Data Loaded:\n", df.head())

# ==============================
# STEP 3: BASIC ANALYSIS
# ==============================
print("\nTotal Revenue:", df["Revenue"].sum())
print("\nSales by Product:\n", df.groupby("Product")["Revenue"].sum())
print("\nSales by Category:\n", df.groupby("Category")["Revenue"].sum())

# ==============================
# STEP 4: CUSTOMER SEGMENTATION
# ==============================
customer_spend = df.groupby("Customer_ID")["Revenue"].sum()

def segment(x):
    if x > 50000:
        return "High Value"
    elif x > 20000:
        return "Medium Value"
    else:
        return "Low Value"

segments = customer_spend.apply(segment)
print("\nCustomer Segments:\n", segments)

# ==============================
# STEP 5: AI PREPARATION
# ==============================
df["Order_Date"] = pd.to_datetime(df["Order_Date"], dayfirst=True)
df["Day"] = df["Order_Date"].dt.day
df["Product_Code"] = df["Product"].astype("category").cat.codes

X = df[["Day", "Product_Code", "Price"]]
y = df["Quantity"]

# ==============================
# STEP 6: TRAIN MODEL
# ==============================
model = LinearRegression()
model.fit(X, y)

# ==============================
# STEP 7: MODEL ACCURACY
# ==============================
y_pred = model.predict(X)
print("\nModel Accuracy (R2 Score):", r2_score(y, y_pred))

# ==============================
# STEP 8: FUTURE PREDICTION
# ==============================
future = pd.DataFrame({
    "Day": [25, 26, 27, 28, 29],
    "Product_Code": [0, 1, 2, 0, 1],
    "Price": [50000, 20000, 3000, 48000, 21000]
})

future["Predicted_Sales"] = model.predict(future)

print("\nFuture Predictions:\n", future)

# ==============================
# STEP 9: FEATURE IMPORTANCE
# ==============================
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.coef_
})

print("\nFeature Importance:\n", importance)

# ==============================
# STEP 10: VISUALIZATION
# ==============================

# Category Sales Graph
df.groupby("Category")["Revenue"].sum().plot(kind='bar')
plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Revenue")
plt.show()

# Prediction Graph
plt.plot(future["Day"], future["Predicted_Sales"])
plt.title("Predicted Sales Trend")
plt.xlabel("Day")
plt.ylabel("Predicted Quantity")
plt.show()