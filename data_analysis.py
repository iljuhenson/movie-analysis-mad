import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "output/movies_relevant_data.csv"
movies_df = pd.read_csv(file_path)

numerical_columns = [
    "avg_of_rating",
    "adult",
    "budget",
    "genres",
    "original_language",
    "release_date",
    "revenue",
    "spoken_languages",
    "runtime",
    "production_countries",
    "vote_count",
]
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(3, 4, i)
    sns.histplot(movies_df[col])
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("output/histograms_numerical_variables.png")
print("histograms_numerical_variables done")

# Plotting the relationship between budget and worldwide gross revenue
plt.figure(figsize=(12, 6))
sns.scatterplot(x="budget", y="revenue", data=movies_df)
plt.title("Relationship between Budget and Revenue")
plt.xlabel("Budget")
plt.ylabel("Revenue")
plt.savefig("output/budget_vs_revenue.png")
print("budget_vs_revenue done")

# Plotting the relationship between domestic and international gross revenue
plt.figure(figsize=(12, 6))
sns.scatterplot(x="avg_of_rating", y="budget", data=movies_df)
plt.title("Relationship between Budget and Average movie rating")
plt.xlabel("Budget")
plt.ylabel("Avg of Rating")
plt.savefig("output/budget_vs_avg_rating.png")
print("budget_vs_avg_rating done")

# Plotting the trend of worldwide gross revenue over the years
plt.figure(figsize=(12, 6))
sns.lineplot(x="release_date", y="revenue", data=movies_df)
plt.title("Trend of Revenue Over the Years")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.savefig("output/revenue_over_years.png")
print("revenue_over_years done")

# # Plotting the trend of budget over the years
plt.figure(figsize=(12, 6))
sns.lineplot(x="release_date", y="budget", data=movies_df)
plt.title("Trend of Budget Over the Years")
plt.xlabel("Date")
plt.ylabel("Budget")
plt.savefig("output/budget_over_years.png")
print("budget_over_years done")
print("done")
