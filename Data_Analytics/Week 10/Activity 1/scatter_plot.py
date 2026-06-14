import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
# Replace line 6 with the full path to your CSV file:
df = pd.read_csv("C:\Yoobee_Trimester 2_Weekly Assigments\Data_Analytics\Week 10\salary-dataset (1).csv", index_col=0)

# Linear regression line
m, b = np.polyfit(df["YearsExperience"], df["Salary"], 1)
x_line = np.linspace(df["YearsExperience"].min(), df["YearsExperience"].max(), 100)
y_line = m * x_line + b

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(
    df["YearsExperience"],
    df["Salary"],
    color="#4C72B0",
    edgecolors="white",
    s=80,
    linewidths=0.8,
    zorder=3,
    label="Employees"
)

ax.plot(x_line, y_line, color="#DD4949", linewidth=2, linestyle="--", label=f"Trend line  (slope: ${m:,.0f}/yr)")

# Labels & formatting
ax.set_title("Years of Experience vs Salary", fontsize=16, fontweight="bold", pad=15)
ax.set_xlabel("Years of Experience", fontsize=13)
ax.set_ylabel("Salary (USD)", fontsize=13)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(fontsize=11)

# Correlation annotation
corr = df["YearsExperience"].corr(df["Salary"])
ax.annotate(
    f"Pearson r = {corr:.2f}",
    xy=(0.05, 0.92),
    xycoords="axes fraction",
    fontsize=11,
    color="#333333",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="#cccccc")
)

plt.tight_layout()
plt.savefig("scatter_plot.png", dpi=150)
plt.show()
print("Plot saved as scatter_plot.png")
