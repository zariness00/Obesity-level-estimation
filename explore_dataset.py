from ucimlrepo import fetch_ucirepo 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from PIL import Image
import numpy as np
# fetch dataset 
obesity_data = fetch_ucirepo(id=544) 
  
# data (as pandas dataframes) 
X = obesity_data.data.features 
y = obesity_data.data.targets 
  
# # metadata 
# print(obesity_data.metadata) 
  
# # variable information 
# print(obesity_data.variables) 

obesity = pd.concat([X, y], axis=1)
# print(df.head())

# Mapping old column names to new, readable ones
new_column_names = {
    'Gender': 'Gender',
    'Age': 'Age',
    'Height': 'Height',
    'Weight': 'Weight',
    'family_history_with_overweight': 'Family_history',
    'FAVC': 'High_cal_food_freq',
    'FCVC': 'Veggie_Consum',
    'NCP': 'Num_of_meals',
    'CAEC': 'Snacks',
    'SMOKE': 'Smoke',
    'CH2O': 'Water_intake',
    'SCC': 'Calories_check',
    'FAF': 'Phys_act_freq',
    'TUE': 'Tech_use',
    'CALC': 'Alcohol_Consump',
    'MTRANS': 'Transport_mode',
    'NObeyesdad': 'Obesity_lvl'
}

obesity = obesity.rename(columns=new_column_names)

print(obesity.head())
#print(df.info())

# Get a summary of the dataset, including data types and non-null counts
print(obesity.info())

# Get basic statistics for numerical features
print(obesity.describe())

# Check for missing values in the dataset
print(obesity.isnull().sum())

# Display the distribution of the target variable 'species'
print(obesity['Obesity_lvl'].value_counts())



#Transform categorical variables to numerical representation
"""
I am planning to build a pair plot to visualize the relationship between the features and the target variable(obesity level)
As I am dealing with the multiclass classification problem, I will perform several steps to group my 7 classes into 4 broader categories:
1. Label the Insufficient weight as 0
2. Label the Normal weight as 1
3. Group all groups of Overweight (Overweight I, Overweight II) into a single category and label it as 2
4. Group all groups of Obesity (Obesity I, Obesity II, Obesity III) into a single category and label it as 3
"""


def map_obesity_level(obesity, source_col = "Obesity_lvl", target_col = "Obesity_group"):
    group_map = {
        "Insufficient_Weight": 0,
        "Normal_Weight" : 1,
        "Overweight_Level_I" : 2,
        "Overweight_Level_II" : 2,
        "Obesity_Type_I" :3,
        "Obesity_Type_II" : 3,
        "Obesity_Type_III" : 3
    }

    obesity[target_col] = obesity[source_col].map(group_map)
    return obesity

obesity = map_obesity_level(obesity) 

def pie_chart(obesity): 
    group_labels = {0: 'Insufficient', 1: 'Normal', 2: 'Overweight', 3: 'Obesity'}

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for i in range(4):
        subset = obesity[obesity['Obesity_group'] == i]['Gender'].value_counts()
        axes[i].pie(subset, labels=subset.index, autopct='%1.1f%%', startangle=90)
        axes[i].set_title(f"Gender Dist. – {group_labels[i]}")

    plt.tight_layout()
    plt.show()


# Create age bins
bins = [14, 19, 24, 29, 34, 39, 44, 49, 54, 61]
labels = ['14-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-61']
obesity['Age'] = pd.cut(obesity['Age'], bins=bins, labels=labels)

def set_seaborn_style(font_family, background_color, grid_color, text_color):
    sns.set_style({
        "axes.facecolor": background_color,
        "figure.facecolor": background_color,

        "axes.labelcolor": text_color,

        "axes.edgecolor": grid_color,
        "axes.grid": True,
        "axes.axisbelow": True,

        "grid.color": grid_color,

        "font.family": font_family,
        "text.color": text_color,
        "xtick.color": text_color,
        "ytick.color": text_color,

        "xtick.bottom": False,
        "xtick.top": False,
        "ytick.left": False,
        "ytick.right": False,

        "axes.spines.left": False,
        "axes.spines.bottom": True,
        "axes.spines.right": False,
        "axes.spines.top": False,
    }
)
FEMALE_COLOR = "#F64740"
MALE_COLOR = "#05B2DC"
font_family="PT Mono"
background_color="#253D5B"
grid_color="#355882"
text_color="#EEEEEE"

set_seaborn_style(font_family, background_color, grid_color, text_color)

def create_age_distribution(obesity, obesity_gr):
    # Filter by gender and obesity group
    female_df = obesity[(obesity.Gender == "Female") & (obesity.Obesity_group == obesity_gr)]
    male_df = obesity[(obesity.Gender == "Male") & (obesity.Obesity_group == obesity_gr)]

    # Count number of people per age group for each gender
    female_counts = female_df['Age'].value_counts().sort_index()
    male_counts = male_df['Age'].value_counts().sort_index()

    # Create bar chart
    age_labels = female_counts.index.tolist()  # Assuming both genders have same age bins
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.barh(age_labels, male_counts.values * -1, color=MALE_COLOR, label='Male')  # Left side
    ax.barh(age_labels, female_counts.values, color=FEMALE_COLOR, label='Female') # Right side

    return ax



fig = plt.figure(figsize=(10, 6))
ax = create_age_distribution(obesity, obesity_gr=0)  # For example, group 3 = Obesity
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: int(abs(x))))
plt.title("Age Distribution for Insufficient Weight Group")
plt.tight_layout()


fig = plt.figure(figsize=(10, 6))
ax = create_age_distribution(obesity, obesity_gr=1)  # For example, group 3 = Obesity
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: int(abs(x))))
plt.title("Age Distribution for Normal Weight Group")
plt.tight_layout()





fig = plt.figure(figsize=(10, 6))
ax = create_age_distribution(obesity, obesity_gr=2)  # For example, group 3 = Obesity
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: int(abs(x))))
plt.title("Age Distribution for Overweight Group")
plt.tight_layout()



fig = plt.figure(figsize=(10, 6))
ax = create_age_distribution(obesity, obesity_gr=3)  # For example, group 3 = Obesity
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: int(abs(x))))
plt.title("Age Distribution for Obesity Group")
plt.tight_layout()
