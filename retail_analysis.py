import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ==========================================
# 1. GENERATE DOMAIN-SPECIFIC DATA (Retail Sales)
# ==========================================
# Simulating 365 days of retail data
np.random.seed(42)
dates = pd.date_range(start='2025-01-01', periods=365)
categories = np.random.choice(['Electronics', 'Clothing', 'Home & Garden'], 365)
marketing_spend = np.random.uniform(100, 1000, 365)
discount_percent = np.random.uniform(0, 30, 365)

# Sales are driven by marketing, discounts, and some random daily variance
sales = 500 + (marketing_spend * 1.5) + (discount_percent * 10) + np.random.normal(0, 100, 365)

df = pd.DataFrame({
    'Date': dates,
    'Category': categories,
    'Marketing_Spend': marketing_spend.round(2),
    'Discount_Percent': discount_percent.round(2),
    'Daily_Sales': sales.round(2)
})

print("--- RETAIL DATASET (First 5 Days) ---")
print(df.head())
print("\n")

# ==========================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
# Let's find out which category makes the most money
category_sales = df.groupby('Category')['Daily_Sales'].sum().reset_index()

# ==========================================
# 3. PREDICTIVE MODELING (Linear Regression)
# ==========================================
# Can we predict daily sales if we know our marketing budget and discount rate?
X = df[['Marketing_Spend', 'Discount_Percent']] # Features
y = df['Daily_Sales'] # Target

# 80/20 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make Predictions & Evaluate
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print(f"Model R-Squared Score: {accuracy:.2f} (1.0 is perfect prediction)\n")

# ==========================================
# 4. PRESENT FINDINGS (Visualizations)
# ==========================================
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Retail Sales Analysis & Prediction Dashboard', fontsize=16, fontweight='bold')

# Plot 1: Total Sales by Category (EDA)
sns.barplot(x='Category', y='Daily_Sales', data=category_sales, ax=axes[0], palette='viridis')
axes[0].set_title('Total Revenue by Category')
axes[0].set_ylabel('Total Sales ($)')

# Plot 2: Sales Trend Over Time (Time Series)
# We use a 14-day rolling average to smooth out the daily spikes and see the actual trend
df.set_index('Date')['Daily_Sales'].rolling(window=14).mean().plot(ax=axes[1], color='coral', lw=2)
axes[1].set_title('14-Day Rolling Average Sales Trend')
axes[1].set_ylabel('Average Daily Sales ($)')

# Plot 3: Actual vs Predicted Sales (Model Evaluation)
sns.scatterplot(x=y_test, y=y_pred, ax=axes[2], color='dodgerblue', alpha=0.7)
# Add a perfect prediction line for reference
axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
axes[2].set_title('Predictive Model: Actual vs Expected Sales')
axes[2].set_xlabel('Actual Sales ($)')
axes[2].set_ylabel('Predicted Sales ($)')

plt.tight_layout()
plt.show()