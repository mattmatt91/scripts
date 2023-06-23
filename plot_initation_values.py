import pandas as pd
import seaborn as sns
from os import listdir, path
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns


my_path = "C:\\Users\\matth\\Desktop\\PaperII\\data\\impact_sensitivity.csv"

df = pd.read_csv(my_path, decimal=',', sep=';')
df['Kinetic Energy [mJ]'] = [round(i, 2)for i in df['Kinetic Energy [mJ]']]
print(df)


# Create subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

# Plot on each subplot
sns_plot1 = sns.barplot(
    data=df, x="Kinetic Energy [mJ]", y="Initiation", hue='sample', ax=axes[0])
sns_plot1.set_title("Total Initiation")

sns_plot2 = sns.barplot(
    data=df, x="Kinetic Energy [mJ]", y="Partial Initiation Count", hue='sample', ax=axes[1])
sns_plot2.set_title("Partial Initiations")

sns_plot3 = sns.barplot(
    data=df, x="Kinetic Energy [mJ]", y="Full Initiation Count", hue='sample', ax=axes[2])
sns_plot3.set_title("Full Initiaions")

# Adjust subplot layout
plt.tight_layout()

# Display the figure
plt.show()
