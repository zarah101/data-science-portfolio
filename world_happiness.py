# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import data, let's use 2015 and 2019
df_2015 = pd.read_csv('2015.csv')
df_2019 = pd.read_csv('2019.csv')

# Let's see the first 5 rows of the data
print(df_2015.head())
print(df_2019.head())

# Let's see the columns of the data
print(df_2015.columns)
print(df_2019.columns)

# Let's see the shape of the data
print(df_2015.shape)
print(df_2019.shape)

# Let's see the data types of the data
print(df_2015.dtypes)
print(df_2019.dtypes)

# Let's see the summary statistics of the data
print(df_2015.describe())
print(df_2019.describe())

# Let's see the missing values of the data
print(df_2015.isnull().sum())
print(df_2019.isnull().sum())

# Let's see the distribution of the data
sns.pairplot(df_2015)
plt.show()

sns.pairplot(df_2019)
plt.show()

# That's a lot to look at, let's focus in on the happiness score
sns.distplot(df_2015['Happiness Score'])
plt.show()

sns.distplot(df_2019['Score'])
plt.show()

# But what effects the happiness score?
# Let's look at the correlation between the variables
df_2015_numeric = df_2015.select_dtypes(include=[np.number])
print(df_2015_numeric.corr())

df_2019_numeric = df_2019.select_dtypes(include=[np.number])
print(df_2019_numeric.corr())

# Let's visualize the correlation
sns.heatmap(df_2015_numeric.corr(), annot=True)
plt.show()

sns.heatmap(df_2019_numeric.corr(), annot=True)
plt.show()

# Now let's do some comparision between 2015 and 2019
# Let's compare the happiness score
sns.distplot(df_2015['Happiness Score'], label='2015')
sns.distplot(df_2019['Score'], label='2019')
plt.legend()
plt.show()

# Let's map this out
import plotly.graph_objs as gobj
import plotly.express as px
from plotly.offline import plot

# Data for the choropleth map
data_scores_2015 = df_2015[['Country', 'Happiness Score']].rename(columns={'Country': 'country', 'Happiness Score': '2015_score'})

# Initializing the data for the choropleth map
data = dict(
    type='choropleth',
    locations=data_scores_2015['country'],  
    locationmode='country names',  
    autocolorscale=False,
    colorscale='Pinkyl', 
    text=data_scores_2015['country'],  
    z=data_scores_2015['2015_score'],  # Happiness scores for 2015
    colorbar={'title': 'Happiness Score', 'len': 0.75, 'lenmode': 'fraction'}
)

# Initializing the layout for the map
layout_2015 = dict(
    title='Happiness Score by Country (Year 2015)',
    geo=dict(showframe=False, showcoastlines=False, projection_type='equirectangular')
)


# Creating the figure using graph_objects
happiness_map_2015 = gobj.Figure(data=[data])
happiness_map_2015.update_layout(layout_2015)

# Display the map
plot(happiness_map_2015)

# Now let's do the same for 2019
# Preparing the data for 2019
data_scores_2019 = df_2019[['Country or region', 'Score']].rename(columns={'Country or region': 'Country or region', 'Score': '2019_score'})

# Initializing the data for the choropleth map
data_2019 = dict(
    type='choropleth',
    locations=data_scores_2019['Country or region'],  # Ensure 'Country or region' is correct
    locationmode='country names',  # Specifies the location mode (country names or country codes)
    autocolorscale=False,
    colorscale='Blues',  # Adjust the color scale as needed
    text=data_scores_2019['Country or region'],  # Show country name when hovering
    z=data_scores_2019['2019_score'],  # Happiness scores for 2019
    colorbar={'title': 'Happiness Score', 'len': 0.75, 'lenmode': 'fraction'}
)

# Initializing the layout for the map
layout_2019 = dict(
    title='Happiness Score by Country (Year 2019)',
    geo=dict(showframe=False, showcoastlines=False, projection_type='equirectangular')
)

# Creating the figure using graph_objects
happiness_map_2019 = gobj.Figure(data=[data_2019])
happiness_map_2019.update_layout(layout_2019)

# Display the map
plot(happiness_map_2019)

# Let's do one last comparison
# Let's graph the top 10 happiest countries in 2015 and 2019
top_10_2015 = df_2015[['Country', 'Happiness Score']].sort_values(by='Happiness Score', ascending=False).head(10)

top_10_2019 = df_2019[['Country or region', 'Score']].sort_values(by='Score', ascending=False).head(10)

# Plotting the top 10 happiest countries in 2015
plt.figure(figsize=(10, 6))
sns.barplot(x='Happiness Score', y='Country', data=top_10_2015, palette='coolwarm')
plt.title('Top 10 Happiest Countries in 2015')
plt.show()

# Plotting the top 10 happiest countries in 2019
plt.figure(figsize=(10, 6))
sns.barplot(x='Score', y='Country or region', data=top_10_2019, palette='coolwarm')
plt.title('Top 10 Happiest Countries in 2019')
plt.show()



