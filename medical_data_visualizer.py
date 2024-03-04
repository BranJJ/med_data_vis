import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def import_data():
    # Import data
    df = pd.read_csv("medical_examination.csv")

    # Add 'overweight' column
    bmi = df["weight"]/ ((df["height"]/100) ** 2)
    df["overweight"] = bmi.apply(lambda x : 1 if x >25 else 0)

    # Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
    df["cholesterol"] = df["cholesterol"].apply(lambda x : 0 if x == 1 else 1)
    df["gluc"] = df["gluc"].apply(lambda x : 0 if x == 1 else 1)
    return df

# Draw Categorical Plot
def draw_cat_plot():

    data = import_data()
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    melted_data = pd.melt(data, id_vars=["cardio"], value_vars=["cholesterol", "gluc", "smoke", "alco", "active","overweight"])


    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    melted_data["total"] = 1
    melted_data = melted_data.groupby(["cardio","variable","value"],as_index=False).count()


    # Draw the catplot with 'sns.catplot()'
    catplot = sns.catplot(data=melted_data, x = "variable", y = "total", col = "cardio", kind = "bar", hue = "value")


    # Get the figure for the output
    fig = catplot.fig


    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig

# Draw Heat Map
def draw_heat_map():

    data = import_data()
    # Clean the data
    data_heat = data[
        (data["ap_lo"] <= data["ap_hi"]) &
        (data["height"] >= data["height"].quantile(0.025)) &
        (data["height"] <= data["height"].quantile(0.975)) &
        (data["weight"] >= data["weight"].quantile(0.025)) &
        (data["weight"] <= data["weight"].quantile(0.975))]

    # Calculate the correlation matrix
    corr = data_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(corr)


    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12,12))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, linewidths=.5, annot = True, square = True, mask= mask, fmt= ".1f", center = 0.08, cbar_kws = {"shrink":0.5})


    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
