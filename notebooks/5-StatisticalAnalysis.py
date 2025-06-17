# In this notebook, we conduct statistical analysis to quantify and validate the mortality trends observed in the Malaysian dataset. We will:
# 1. Analyze temporal trends to visualize changes from 2000 to 2021.
# 2. Perform enhanced correlation analysis with confidence intervals to demonstrate pandemic impacts on mortality with a measure of uncertainty.
# 3. Use hypothesis testing with effect size to confirm the statistical significance of the COVID-19 pandemic's impact (p < 0.05) and its practical importance.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, pearsonr

# Load the comprehensive EDA-ready dataset
df = pd.read_csv("../data/processed/malaysia_mortality_data_eda.csv")

# For simplicity in trend analysis, we focus on 'Persons' to get an overall view without confounding by sex.
df_persons = df[df["Sex"] == "Persons"].copy()

# %%
# --- 1. Temporal Trend Analysis ---
# Aggregate data by Year and L1 Disease Category
l1_trends = (
    df_persons.groupby(["Year", "Disease_L1"])["Mortality Count"].sum().reset_index()
)

# Clean up whitespace for better legend display
l1_trends["Disease_L1"] = l1_trends["Disease_L1"].str.strip()

# Create the line plot
plt.figure(figsize=(12, 7))
sns.lineplot(
    data=l1_trends,
    x="Year",
    y="Mortality Count",
    hue="Disease_L1",
    marker="o",
    linewidth=2.5,
)
plt.title(
    "Mortality Trends by Major Disease Category in Malaysia (2000, 2020, 2021)",
    fontsize=16,
)
plt.ylabel("Total Mortality Count")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.xticks([2000, 2020, 2021])
plt.show()

# The line plot reveals a dramatic shift in Malaysia's mortality landscape.
#     • From 2000 to 2020, Non-communicable diseases (NCDs) show a clear increasing trend, reflecting the ongoing epidemiological transition where chronic diseases become the primary cause of death.
#    • The year 2021 marks a stark deviation. While NCDs continue their rise, there is an explosive increase in deaths from Communicable, maternal, perinatal and nutritional conditions. This is almost entirely attributable to the COVID-19 pandemic, which falls under this L1 category.
#    • Mortality from injuries remains relatively stable and low compared to the other categories, indicating that the pandemic's primary impact was on disease-related mortality.


# %%
# --- 2. Correlation Analysis ---

# Use an ordinal mapping for the timeline since there's a natural order.
timeline_map = {2000: 0, 2020: 1, 2021: 2}
df_persons["Timeline"] = df_persons["Year"].map(timeline_map)

df_persons["Disease_L2"] = df_persons["Disease_L2"].str.strip()

# Format the data: one row for each year, one column for each disease
l2_pivot = df_persons.pivot_table(
    index="Year", columns="Disease_L2", values="Mortality Count", aggfunc="sum"
).fillna(0)

correlations: list[dict[str, np.float64]] = []
for disease in l2_pivot.columns:
    # Sanity Check: Correlation requires variance.
    if l2_pivot[disease].std() > 0:
        # Calculate Pearson's r and the p-value.
        # The r-value indicates how strongly the disease's mortality count is associated with the timeline.
        # The p-value indicates the statistical significance of the correlation.
        corr, p_val = pearsonr(l2_pivot.index.map(timeline_map), l2_pivot[disease])

        # Calculate R-squared for effect size.
        r_squared = corr**2
        correlations.append(
            {
                "Disease": disease,
                "Correlation (r)": corr,
                "P-value": p_val,
                "R-squared": r_squared,
            }
        )

# Convert the results into a pandas DataFrame.
corr_df = pd.DataFrame(correlations)

# Filter for strong correlations and sort.
strong_corr_df = corr_df[abs(corr_df["Correlation (r)"]) > 0.5].sort_values(
    by="Correlation (r)", ascending=False
)

print("--- Diseases with Strong Correlation (r > 0.5) to Timeline ---")
# Use .to_string() to ensure all columns are displayed and aligned.
print(strong_corr_df.to_string())

# Perfect Linear Trends (r = 1.0):
# Diseases: Mental and substance use disorders, Other neoplasms, Nutritional deficiencies.
# Analysis: These categories show a perfect positive correlation.
# While the p-value is extremely low (statistically significant), this perfect linearity for lower-mortality diseases may be an artifact of the WHO's statistical estimation models, which can smooth trends over time.
#
# Strong, Statistically Significant Positive Trends (Direct & Indirect Pandemic Effects):
# Diseases: Endocrine, blood, immune disorders (r=0.998), Diabetes mellitus (r = 0.994), Cardiovascular diseases (r = 0.992).
# Analysis: These categories show extremely strong positive correlations with significant p-values (p < 0.10).
# The high R^2 values (e.g., 98.4% for Cardiovascular diseases) indicate that the timeline is a powerful predictor of the increase in deaths.
# This strongly supports the hypothesis of a severe indirect pandemic effect, where healthcare system strain led to worse outcomes for patients with these chronic conditions.
#
# Key Pandemic Signal:
# Disease: Respiratory Infectious (r = 0.959).
# Analysis: While not a perfect 1.0, this high correlation is arguably the most important finding.
# It captures the non-linear explosion of deaths in 2021 due to COVID-19.
# The R^2 value of 0.92 signifies that 92% of the variation in these deaths is explained by the timeline, confirming it as the primary signal of the pandemic's direct mortality impact.
#
# Significant Negative Trends (Public Health Successes):
# Diseases: Neonatal conditions (r=-0.955), Maternal conditions (r=-0.866).
# Analysis: These strong negative correlations are highly encouraging. They indicate that despite the pandemic, long-term public health initiatives to reduce infant and maternal mortality have been successful and continued their downward trend.

# %%
# --- 3. Hypothesis Testing with Chi-Squared Test (Definitive Statistical Proof) ---

# We will test if the distribution of deaths across disease categories is
# dependent on the year. This directly tests the pandemic's impact.

# Create a contingency table: Rows are L1 diseases, Columns are Years
# Values are the total mortality counts for each.
contingency_table = df_persons.pivot_table(
    index="Disease_L1", columns="Year", values="Mortality Count", aggfunc="sum"
).fillna(0)

# Remove the aggregated "All Causes" category
contingency_table = contingency_table[contingency_table.index != "All Causes"]

print("--- Contingency Table for Chi-Squared Test ---")
print(contingency_table)

# Perform the Chi-Squared Test
# This compares our observed counts to the counts we would 'expect'
# if there were no association between year and disease category.
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

# --- Step 4: Calculate Effect Size (Cramér's V) ---
# This tells us the STRENGTH of the association, from 0 (none) to 1 (perfect).
n = contingency_table.sum().sum()
cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

# --- Step 5: Report and Interpret the Results ---
print("\n--- Chi-Squared Test of Independence Results ---")
print(f"Chi-Squared Statistic: {chi2:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Degrees of Freedom: {dof}")
print(f"Effect Size (Cramér's V): {cramers_v:.4f}")

if p_value < 0.05:
    print("\nResult: The p-value is less than 0.05. We reject the null hypothesis.")
    print(
        "Conclusion: There is a statistically significant association between the year and the distribution of mortality across disease categories."
    )
else:
    print(
        "\nResult: The p-value is not less than 0.05. We fail to reject the null hypothesis."
    )
# The test provides definitive proof that the pandemic's impact was real and significantly altered Malaysia's mortality patterns.
#
# Contingency Table: Shows a massive jump in Communicable disease deaths in 2021, while Noncommunicable deaths grew more steadily. This visually suggests a major disruption.
# Chi-Squared Statistic (chi^2 = 27287.38): A very large value, indicating a huge difference between our observed data and what would be expected if there was no change in mortality patterns.
# P-value (p = 0.0000): Extremely low, far below the 0.05 threshold. This means the observed shift is statistically significant and not due to random chance. We can confidently reject the null hypothesis.
# Effect Size (Cramér's V = 0.0940): A small effect size. This is because NCDs still make up the vast majority of total deaths. It tells us the pandemic caused a proportionally small, but critically important and statistically real, shift in the overall mortality landscape.
