import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Global YouTube Statistics.csv')

#Q1
top_10 = df.sort_values(by='subscribers', ascending=False).head(10)
print(top_10[['rank', 'Youtuber', 'subscribers', 'video views', 'category']])

plt.figure(figsize=(12, 6))
sns.barplot(
    data=top_10,
    x='subscribers',
    y='Youtuber',
    palette='viridis'
)
plt.title('Top 10 YouTubers by Subscriber Count', fontsize=14, fontweight='bold')
plt.xlabel('Subscribers (in millions)', fontsize=12)
plt.ylabel('YouTuber', fontsize=12)
plt.ticklabel_format(style='plain', axis='x')
plt.tight_layout()
plt.show()

# Q2
df_clean = df.dropna(subset=['category'])
avg_subs = df_clean.groupby('category')['subscribers'].mean().sort_values(ascending=False)
print(avg_subs)

plt.figure(figsize=(12, 6))
sns.barplot(
    x=avg_subs.values,
    y=avg_subs.index,
    palette='coolwarm'
)
plt.title('Average Subscribers by Category', fontsize=14, fontweight='bold')
plt.xlabel('Average Subscribers', fontsize=12)
plt.ylabel('Category', fontsize=12)
plt.ticklabel_format(style='plain', axis='x')
plt.tight_layout()
plt.show()

# #Q3
df_clean = df.dropna(subset=['category', 'uploads'])
avg_uploads = df_clean.groupby('category')['uploads'].mean().sort_values(ascending=False)
print(avg_uploads)

# #Q4
df_clean = df.dropna(subset=['Country of origin'])
country_counts = df_clean['Country of origin'].value_counts().head()
print(country_counts)

# #Q5
df_clean = df.dropna(subset=['category', 'channel_type'])
distribution_pct = pd.crosstab(df_clean['category'], df_clean['channel_type'], normalize='index') * 100
distribution_pct.plot(kind='bar', stacked=True, figsize=(12,6), colormap='tab20')
plt.ylabel("Percentage of Channel Types")
plt.title("Distribution of Channel Types Across Categories")
plt.legend(title="Channel Type")
plt.show()
print(distribution_pct.round(2))

#Q6
df_clean = df[(df['subscribers'] > 0) & (df['video views'] > 0)]
corr = df_clean['subscribers'].corr(df_clean['video views'])
print(corr)

plt.figure(figsize=(8,5))
sns.scatterplot(
   data=df_clean, 
   x='subscribers', 
   y='video views',
   alpha=0.5
)
plt.title("Subscribers vs Total Video Views (Log-Log plot)")
plt.xlabel("Subscribers")
plt.ylabel("Video Views")
plt.xscale('log')
plt.yscale('log')
plt.show()

# #Q7
df['avg_monthly_earnings'] = (df['lowest_monthly_earnings'] + df['highest_monthly_earnings'])
df_clean = df.dropna(subset=['category', 'avg_monthly_earnings'])

mean_earnings = df_clean.groupby('category')['avg_monthly_earnings'].mean().sort_values(ascending=False)
print("\nAverage Monthly Earnings by Category (Mean):")
print(mean_earnings)

median_earnings = df_clean.groupby('category')['avg_monthly_earnings'].median().sort_values(ascending=False)
print("\nAverage Monthly Earnings by Category (Median):")
print(median_earnings)

plt.figure(figsize=(12,6))
sns.boxplot(
   data=df_clean,
   x='category',
   y='avg_monthly_earnings'
)
plt.xticks(rotation=90)
plt.title("Monthly Earnings Distribution Across Categories")
plt.ylabel("Average Monthly Earnings (USD)")
plt.xlabel("Category")
plt.tight_layout()
plt.show()

# #Q8
df_clean = df.dropna(subset=['subscribers_for_last_30_days'])

mean_gain = df_clean['subscribers_for_last_30_days'].mean()
median_gain = df_clean['subscribers_for_last_30_days'].median()
std_gain = df_clean['subscribers_for_last_30_days'].std()

print("Mean subscribers gained in last 30 days:", mean_gain)
print("Median subscribers:", median_gain)
print("Standard Deviation:", std_gain)

plt.figure(figsize=(10,5))
sns.histplot(df_clean['subscribers_for_last_30_days'], bins=50, kde=True)
plt.title("Distribution of Subscribers Gained in the Last 30 Days")
plt.xlabel("Subscribers Gained")
plt.ylabel("Frequency")
plt.show()


# #Q9
df['avg_yearly_earnings'] = (df['lowest_yearly_earnings'] + df['highest_yearly_earnings'])
df_clean = df.dropna(subset=['avg_yearly_earnings'])

Q1 = df_clean['avg_yearly_earnings'].quantile(0.25)
Q3 = df_clean['avg_yearly_earnings'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_clean[
    (df_clean['avg_yearly_earnings'] < lower_bound) |
    (df_clean['avg_yearly_earnings'] > upper_bound)
]

print("Number of outliers: ", len(outliers))
print(outliers[['Youtuber', 'avg_yearly_earnings']].head())

plt.figure(figsize=(6,5))
sns.boxplot(x=df_clean['avg_yearly_earnings'])
plt.title("Outliers in Yearly Earnings")
plt.xlabel("Average Yearly Earnings (USD)")
plt.show()

# #Q10
df_years = df[(df['created_year'] >= 2005)]
year_counts = df_years['created_year'].value_counts().sort_index()
print(year_counts)

plt.figure(figsize=(12,6))
sns.lineplot(x=year_counts.index, y=year_counts.values)
plt.title("Trend of YouTube Channel Creation Over Time")
plt.xlabel("Year")
plt.ylabel("Number of Channels Created")
plt.grid(True)
plt.show()


#Q11
df_clean = df.dropna(subset=['Country of origin'])

channel_counts = df_clean['Country of origin'].value_counts().reset_index()
channel_counts.columns = ['Country of origin', 'channel_count']

edu_per_country = (df_clean.groupby('Country of origin')['Gross tertiary education enrollment (%)'].mean().reset_index())

merged = pd.merge(channel_counts, edu_per_country, on='Country of origin', how='left')

merged_clean = merged.dropna(subset=['Gross tertiary education enrollment (%)'])

corr_value = merged_clean['channel_count'].corr(merged_clean['Gross tertiary education enrollment (%)'])
print("Correlation:", corr_value)

plt.figure(figsize=(8,5))
sns.scatterplot(
    data=merged_clean,
    x='Gross tertiary education enrollment (%)',
    y='channel_count'
)
plt.title("Education Enrollment vs Number of YouTube Channels per Country")
plt.xlabel("Gross Tertiary Enrollment (%)")
plt.ylabel("Number of Channels")
plt.show()

#Q12
top10_countries = (df['Country of origin'].value_counts().head(10).reset_index())
top10_countries.columns = ['Country of origin', 'channel_count']

unemployment = (df.groupby('Country of origin')['Unemployment rate'].mean().reset_index())

top10_merged = pd.merge(top10_countries,unemployment, on='Country of origin', how='left')
top10_clean = top10_merged.dropna(subset=['Unemployment rate'])

plt.figure(figsize=(10,5))
sns.barplot(
    data=top10_clean,
    x='Country of origin',
    y='Unemployment rate'
)
plt.title("Unemployment Rate Among Top 10 Countries by YouTube Channels")
plt.xticks(rotation=45)
plt.ylabel("Unemployment Rate (%)")
plt.xlabel("Country")
plt.show()

#Q13
df_country = df.dropna(subset=['Country of origin', 'Population', 'Urban_population'])
country_pop = (df_country.groupby('Country of origin')[['Population', 'Urban_population']].mean().reset_index())
country_pop['urban_percentage'] = (country_pop['Urban_population'] / country_pop['Population']) * 100
average_urban_percentage = country_pop['urban_percentage'].mean()
print("Average urban population percentage:", average_urban_percentage)

plt.figure(figsize=(10, 6))
plt.hist(country_pop['urban_percentage'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(average_urban_percentage, color='red', linestyle='--', linewidth=2, label=f'Mean: {average_urban_percentage:.2f}%')
plt.title('Distribution of Urban Population Percentage by Country', fontsize=14, fontweight='bold')
plt.xlabel('Urban Population Percentage (%)', fontsize=12)
plt.ylabel('Number of Countries', fontsize=12)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

#Q14
df_geo = df.dropna(subset=['Latitude', 'Longitude'])
plt.figure(figsize=(10,5))
sns.scatterplot(
    data=df_geo,
    x='Longitude',
    y='Latitude',
    alpha=0.5
)
plt.title("Geographical Distribution of YouTube Channels")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

sns.kdeplot(
    data=df_geo,
    x='Longitude',
    y='Latitude',
    fill=True,
    cmap='Reds',
    thresh=0.05
)
plt.title("Density of YouTube Channels by Location")
plt.show()

#Q15
df_clean = df.dropna(subset=['Country of origin', 'Population'])

country_agg = (
    df_clean
    .groupby('Country of origin')
    .agg(
        total_subscribers=('subscribers', 'sum'),
        population=('Population', 'mean')
    )
    .reset_index()
)

corr_value = country_agg['total_subscribers'].corr(country_agg['population'])
print("Correlation:", corr_value)

plt.figure(figsize=(8,5))
sns.scatterplot(
    data=country_agg,
    x='population',
    y='total_subscribers'
)
plt.xscale('log')
plt.yscale('log')
plt.title("Population vs Total YouTube Subscribers by Country")
plt.xlabel("Population (log scale)")
plt.ylabel("Total Subscribers (log scale)")
plt.show()

#Q16
top10_countries = (
    df['Country of origin']
    .value_counts()
    .head(10)
    .reset_index()
)

top10_countries.columns = ['Country of origin', 'channel_count']
population_by_country = (
    df.groupby('Country of origin')['Population']
    .mean()
    .reset_index()
)

top10_pop = pd.merge(
    top10_countries,
    population_by_country,
    on='Country of origin',
    how='left'
)
top10_pop_clean = top10_pop.dropna(subset=['Population'])
print(top10_pop_clean.sort_values(by='Population', ascending=False))

plt.figure(figsize=(10,5))
sns.barplot(
    data=top10_pop_clean,
    x='Country of origin',
    y='Population'
)
plt.xticks(rotation=45)
plt.title("Population of Top 10 Countries by YouTube Channel Count")
plt.ylabel("Population")
plt.xlabel("Country")
plt.show()

#Q17
df_clean = df.dropna(subset=['Country of origin'])

subs_gain_country = (df_clean.groupby('Country of origin')['subscribers_for_last_30_days'].sum().reset_index())
unemployment_country = (df_clean.groupby('Country of origin')['Unemployment rate'].mean().reset_index())
merged = pd.merge(subs_gain_country, unemployment_country, on='Country of origin', how='left')

merged_clean = merged.dropna(subset=['subscribers_for_last_30_days', 'Unemployment rate'])
corr_value = merged_clean['subscribers_for_last_30_days'].corr(merged_clean['Unemployment rate'])
print("Correlation:", corr_value)

plt.figure(figsize=(8,5))
sns.scatterplot(
    data=merged_clean,
    x='Unemployment rate',
    y='subscribers_for_last_30_days'
)
plt.title("Unemployment Rate vs Subscriber Growth (Last 30 Days)")
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Subscribers Gained (Last 30 Days)")
plt.show()

#Q18
df_dist = df.dropna(subset=['channel_type', 'video_views_for_the_last_30_days'])

plt.figure(figsize=(12,6))
sns.boxplot(
    data=df_dist,
    x='channel_type',
    y='video_views_for_the_last_30_days'
)
plt.yscale('log')
plt.xticks(rotation=45)
plt.title("Distribution of Video Views in the Last 30 Days by Channel Type")
plt.xlabel("Channel Type")
plt.ylabel("Video Views (Last 30 Days, log scale)")
plt.show()

median_views = (df_dist.groupby('channel_type')['video_views_for_the_last_30_days'].median().sort_values(ascending=False))
print(median_views)

#Q19
month_counts = (df['created_month'].value_counts().sort_index())
print(month_counts)

plt.figure(figsize=(10,5))
sns.barplot(
    x=month_counts.index,
    y=month_counts.values
)
plt.title("Distribution of Channel Creation by Month")
plt.xlabel("Month")
plt.ylabel("Number of Channels Created")
plt.show()

#Q20
df_age = df[df['created_year'] >= 2005].copy()

month_map = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
    'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
    'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

df_age['created_month_num'] = df_age['created_month'].map(month_map)

df_age['created_datetime'] = pd.to_datetime(
    dict(
        year=df_age['created_year'],
        month=df_age['created_month_num'],
        day=df_age['created_date']
    ),
    errors='coerce'
)

df_age = df_age.dropna(subset=['created_datetime'])

today = pd.Timestamp.today()

df_age['age_months'] = (
    (today.year - df_age['created_datetime'].dt.year) * 12 +
    (today.month - df_age['created_datetime'].dt.month)
)

df_age = df_age[df_age['age_months'] > 0]

df_age['subs_per_month'] = df_age['subscribers'] / df_age['age_months']

average_subs_per_month = df_age['subs_per_month'].mean()
median_subs_per_month = df_age['subs_per_month'].median()

print("Average subscribers gained per month since creation:", average_subs_per_month)
print("Median subscribers gained per month since creation:", median_subs_per_month)

plt.figure(figsize=(12, 6))
plt.hist(df_age['subs_per_month'], bins=50, color='teal', edgecolor='black', alpha=0.7)
plt.axvline(average_subs_per_month, color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {average_subs_per_month:,.0f}')
plt.axvline(median_subs_per_month, color='orange', linestyle='--', linewidth=2, 
            label=f'Median: {median_subs_per_month:,.0f}')
plt.title('Distribution of Subscribers Gained Per Month', fontsize=14, fontweight='bold')
plt.xlabel('Subscribers Per Month', fontsize=12)
plt.ylabel('Number of Channels', fontsize=12)
plt.xscale('log')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()