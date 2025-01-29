import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as sp
from scipy.stats import zscore

# Combined strip and violin plot of 5Xtime plot of head motion as a function of scan order (=time)
motionvars = pd.read_csv("/Users/Cabria/Downloads/HeadMotionVarscopy.csv", low_memory=False)
#Do z-score before plot
motionvars_long = pd.melt(motionvars, id_vars=['eid'], var_name='Imaging Type', value_name='Head Motion')
mean_head_motion = motionvars_long['Head Motion'].mean()
std_head_motion = motionvars_long['Head Motion'].std()
motionvars_long['Z-Score_Head_Motion'] = (motionvars_long['Head Motion'] - mean_head_motion) / std_head_motion
print(motionvars_long)
# Strip plot 2nd
strip_plot = sns.stripplot(data=motionvars_long, x='Imaging Type', y='Z-Score_Head_Motion', jitter=True, size=1.5, color='green')
plt.xlabel('Time(Scan Type)')
plt.ylabel('Z-Score_Head_Motion')
plt.title('Head Motion in MRI Scan by Imaging Type')
plt.xticks(rotation=90)
plt.ylim(0, 10)

# Violin plot 1st
sns.violinplot(
    data=motionvars_long,
    x='Imaging Type',
    y='Z-Score_Head_Motion',
    color='#DDDDDD', width=1.2, density_norm="count")
plt.show()


swarm_plot = sns.swarmplot(data=motionvars_long, x='Imaging Type', y='Z-Score_Head_Motion', size=1.5, color='green')
# Zscore: handles, labels = strip_plot.get_legend_handles_labels() plt.legend(handles, labels, title='Z-Score', loc='upper right')






# Heat map (head motion x head motion correlation)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
motionvars = pd.read_csv("/Users/Cabria/Downloads/HeadMotionVarscopy.csv", low_memory=False)

# Exclude columns with "-3." and the 'eid' column for the correlation matrix
subsetcolumns = [x for x in list(motionvars.columns) if "-3." not in x and x != 'eid']
motionvarssub = motionvars[subsetcolumns]

# Calculate the correlation matrix
corrM = motionvarssub.corr()
# Reverse the correlation matrix for the y-axis to start with 'Measure of T1' at the top
corrM = corrM.iloc[::-1]

# Define the color map
cmap = sns.color_palette("crest", as_cmap=True)
sns.heatmap(corrM, cmap=cmap, cbar_kws={'ticks': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]})

# Labels for the axes, in the correct order for x-axis
labels = [ "Measure of T1",
    "Mean abs motion rfMRI",
    "Median abs motion rfMRI",
    "90th %ile abs motion rfMRI",
    "Mean rel motion rfMRI",
    "Median rel motion rfMRI",
    "90th %ile rel motion rfMRI",
    "Mean motion rfMRI avg",
    "Mean abs motion dMRI",
    "Median abs motion dMRI",
    "90th %ile abs motion dMRI",
    "Mean rel motion dMRI",
    "Median rel motion dMRI",
    "90th %ile rel motion dMRI",
    "Mean abs motion tfMRI",
    "Median abs motion tfMRI",
    "90th %ile abs motion tfMRI",
    "Mean rel motion tfMRI",
    "Median rel motion tfMRI",
    "90th %ile rel motion tfMRI",
    "Mean motion tfMRI avg"
]

# Calculate the position for each label to be at the center of the cell
tick_positions = np.arange(len(labels))

# Apply the correct label order for x-axis and y-axis
plt.xticks(ticks=tick_positions, labels=labels, rotation=45)
plt.yticks(ticks=tick_positions, labels=labels)  # Reverse order for y-ticks

# Add figure title
plt.title("Correlation between Head Motion Variables")

# Ensure the axis limits are large enough to display all labels and boxes
plt.xlim(0, len(labels))
plt.ylim(0, len(labels))

# Improve the layout
plt.tight_layout()

# Show the plot
plt.show()




# Heat map (head motion x head motion correlation) with grid and better ticks
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.transforms as transforms

# Load the dataset
motionvars = pd.read_csv("/Users/Cabria/Downloads/HeadMotionVarscopy.csv", low_memory=False)

# Exclude columns with "-3." and the 'eid' column for the correlation matrix
subsetcolumns = [x for x in list(motionvars.columns) if "-3." not in x and x != 'eid']
motionvarssub = motionvars[subsetcolumns]

# Calculate the correlation matrix
corrM = motionvarssub.corr()

# Define the color map
cmap = sns.color_palette("crest", as_cmap=True)

# Initialize the matplotlib figure
plt.figure(figsize=(20, 20))

# Create the heatmap
ax = sns.heatmap(corrM, cmap=cmap, annot=False, fmt=".2f", linewidths=.5, cbar_kws={'ticks': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]})

# Labels for the axes
labels = [
    "Measure of T1",
    "Mean abs motion rfMRI",
    "Median abs motion rfMRI",
    "90th %ile abs motion rfMRI",
    "Mean rel motion rfMRI",
    "Median rel motion rfMRI",
    "90th %ile rel motion rfMRI",
    "Mean motion rfMRI avg",
    "Mean abs motion dMRI",
    "Median abs motion dMRI",
    "90th %ile abs motion dMRI",
    "Mean rel motion dMRI",
    "Median rel motion dMRI",
    "90th %ile rel motion dMRI",
    "Mean abs motion tfMRI",
    "Median abs motion tfMRI",
    "90th %ile abs motion tfMRI",
    "Mean rel motion tfMRI",
    "Median rel motion tfMRI",
    "90th %ile rel motion tfMRI",
    "Mean motion tfMRI avg"
]

# Adjust the tick positions as found to be visually best
xtick_positions = np.arange(len(labels)) + 0.4
ytick_positions = np.arange(len(labels)) + 0.4

# Set the tick positions for x-axis and y-axis
plt.xticks(ticks=xtick_positions, labels=labels, rotation=45, ha='right', fontsize=9)
plt.yticks(ticks=ytick_positions, labels=labels, rotation=0)

# Apply a small translation transformation to the x-tick labels to shift them slightly
dx = 5/ 72.  # Shift 5 points to the right
dy = 0  # No shift vertically
offset = transforms.ScaledTranslation(dx, dy, plt.gcf().dpi_scale_trans)

# Apply the transformation to each x-tick label
for label in ax.get_xticklabels():
    label.set_transform(label.get_transform() + offset)

# Add figure title
plt.title("Correlation between Head Motion Variables")

# Ensure the axis limits are large enough to display all labels and boxes
plt.xlim(0, len(labels))
plt.ylim(len(labels), 0)

# Improve the layout
plt.tight_layout()

# Show the plot
plt.show()




# if those axes func don't work use these
# labels for axes should be shortened
#labels = [
"Eid",
"24419 T1 structural image",
"24438 Mean absolute head motion from rfMRI",
"24439 Median absolute head motion from rfMRI",
"24440 90th percentile of absolute head motion from rfMRI",
"24441 Mean relative head motion from rfMRI",
"24442 Median relative head motion from rfMRI",
"24443 90th percentile of relative head motion from rfMRI",
"25741 Mean rfMRI head motion, averaged across space and time points",
"24450 Mean absolute head motion from dMRI",
"24451 Median absolute head motion from dMRI",
"24452 90th percentile of absolute head motion from dMRI",
"24453 Mean relative head motion from dMRI",
"24454 Median relative head motion from dMRI",
"24455 90th percentile of relative head motion from dMRI",
"24444 Mean absolute head motion from tfMRI",
"24445 Median absolute head motion from tfMRI",
"24446 90th percentile of absolute head motion from tfMRI",
"24447 Mean relative head motion from tfMRI",
"24448 Median relative head motion from tfMRI",
"24449 90th percentile of relative head motion from tfMRI",
"25742 Mean tfMRI head motion, averaged across space and time points"]

#plt.yticks(range(len(labels)), labels, fontsize=12)
#plt.xticks(range(len(labels)), labels, rotation=90, fontsize=12)
#plt.show()





#Correlaton matrix: IDPXHeadMotion
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

motionvars = pd.read_csv("/Users/Cabria/Downloads/HeadMotionVarscopy.csv", low_memory=False)
idp = pd.read_csv("/Users/Cabria/Downloads/Head Motion Project/Desikan_IDP2(3.0).csv", low_memory=False)


# Preprocess and merge data
subsetcolumns_motionvars = [x for x in list(motionvars.columns) if "-3." not in x]
motionvarssub = motionvars[subsetcolumns_motionvars]
subsetcolumns_idp = [x for x in list(idp.columns) if "-3." not in x]
idpsub = idp[subsetcolumns_idp]
newdf = pd.merge(motionvarssub, idpsub, on='eid', how='inner')

# Calculate correlation
I2M = newdf.corr()
I2Mcorner = I2M.iloc[1:20, 21:]

# Plotting
sns.heatmap(I2Mcorner, cmap='crest', vmin=-0.20, vmax=0.15)

# Update y-axis labels
y_labels = {
    "24419-2.0": "Measure of head motion in T1",
    "24438-2.0": "Mean abs. motion rfMRI",
    "24439-2.0": "Median abs. motion rfMRI",
    "24440-2.0": "90th %ile abs. motion rfMRI",
    "24441-2.0": "Mean rel. motion rfMRI",
    "24442-2.0": "Median rel. motion rfMRI",
    "24443-2.0": "90th %ile rel. motion rfMRI",
    "25741-2.0": "Mean rfMRI motion avg.",
    "24450-2.0": "Mean abs. motion dMRI",
    "24451-2.0": "Median abs. motion dMRI",
    "24452-2.0": "90th %ile abs. motion dMRI",
    "24453-2.0": "Mean rel. motion dMRI",
    "24454-2.0": "Median rel. motion dMRI",
    "24455-2.0": "90th %ile rel. motion dMRI",
    "24444-2.0": "Mean abs. motion tfMRI",
    "24445-2.0": "Median abs. motion tfMRI",
    "24446-2.0": "90th %ile abs. motion tfMRI",
    "24447-2.0": "Mean rel. motion tfMRI",
    "24448-2.0": "Median rel. motion tfMRI",
    "24449-2.0": "90th %ile rel. motion tfMRI",
    "25742-2.0": "Mean tfMRI motion avg."
}

# Replace y-ticks with descriptive labels
plt.yticks(ticks=range(len(I2Mcorner.index)), labels=[y_labels.get(label, label) for label in I2Mcorner.index], rotation=0)

plt.xticks([]) + 0.4

# Add axis titles
plt.xlabel("Structural Imaging Variables", labelpad=5)
plt.ylabel("Head Motion Variables")

# Add figure title
plt.title("Correlation between Head Motion and Structural Images")

# Add legend for abbreviations used in y-labels
abbreviations = {
    "abs.": "absolute",
    "rel.": "relative",
    "motion": "head motion",
    "rfMRI": "resting fMRI",
    "dMRI": "diffusion MRI",
    "tfMRI": "task fMRI",
    "%ile": "percentile",
    "avg.": "averaged"
}
legend_text = "\n".join([f"{key}: {value}" for key, value in abbreviations.items()])
plt.gcf().text(1.02, 0.5, legend_text, fontsize=9, va='center')

plt.tight_layout()
plt.show()



#Correlaton matrix: IDPXHeadMotion (with all 21 hm variables)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the datasets
motionvars = pd.read_csv("/Users/Cabria/Downloads/HeadMotionVarscopy.csv", low_memory=False)
idp = pd.read_csv("/Users/Cabria/Downloads/Head Motion Project/Desikan_IDP2(3.0).csv", low_memory=False)

# Preprocess and merge data
subsetcolumns_motionvars = [x for x in list(motionvars.columns) if "-3." not in x]
motionvarssub = motionvars[subsetcolumns_motionvars]
subsetcolumns_idp = [x for x in list(idp.columns) if "-3." not in x]
idpsub = idp[subsetcolumns_idp]
newdf = pd.merge(motionvarssub, idpsub, on='eid', how='inner')

# Calculate correlation
I2M = newdf.corr()
I2Mcorner = I2M.iloc[:21, 21:]  # Include the first 21 rows for correlation analysis

# Plotting
plt.figure(figsize=(10, 12))
sns.heatmap(I2Mcorner, cmap='crest', vmin=-0.20, vmax=0.15)

# Define y-labels
y_labels = {
    # Your full y_labels dictionary mapping should be here
    "24419-2.0": "Measure of head motion in T1",
    "24438-2.0": "Mean abs. motion rfMRI",
    "24439-2.0": "Median abs. motion rfMRI",
    "24440-2.0": "90th %ile abs. motion rfMRI",
    # Add all necessary mappings
}

# Set y-ticks with labels
plt.yticks(ticks=np.arange(len(I2Mcorner.index)) + 0.4,
           labels=[y_labels.get(idx, idx) for idx in I2Mcorner.index], rotation=0)

# Remove x-tick labels
plt.xticks([])  # This removes x-tick labels entirely

# Add axis titles with padding
plt.xlabel("Structural Imaging Variables", labelpad=35)
plt.ylabel("Head Motion Variables")

# Add figure title
plt.title("Correlation between Head Motion and Structural Images")

# Adjust layout to fit titles and labels without cutting off
plt.subplots_adjust(bottom=0.2, top=0.95, left=0.15, right=0.95)

# Show the plot
plt.show()





#Old (doesnt work)Correlation matrix between IDPxHM
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
motionvars = pd.read_csv("/Users/Cabria/Downloads/HeadMotionVarscopy.csv", low_memory=False)
subsetcolumns = [x for x in list(motionvars.columns) if "-3." not in x]
motionvarssub = motionvars[subsetcolumns]
corrM = motionvarssub.corr()

idp = pd.read_csv("/Users/Cabria/Downloads/Head Motion Project/Desikan_IDP2(3.0).csv", low_memory=False)
idp.columns
subsetcolumns = [x for x in list(idp.columns) if "-3." not in x]
idpsub = idp[subsetcolumns]
idpsub.shape
motionvarssub.shape
newdf = pd.merge(motionvarssub, idpsub, on='eid', how='inner')

I2M = newdf.corr()
print(list(I2M.columns))

I2Mcorner = I2M.iloc[1:20, 21:]
plt.figure(figsize=(8, 10))
plt.clim(-0.20, 0.15)
sns.heatmap(I2Mcorner, cmap='crest')

labelsy = ["T-1 structural image variables"]
plt.show()
# cbar_kws={'ticks': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}, vmin=0, vmax=1)


labelsx = ["27239 Entorhinal (L)", "27173 Insula (L)", '27245 Lateral Orbitofrontal (R)', "27157 Paracentral (L)",
"27252 Pars Orbitalis (R)", "27163 Posterior Cingulate (L)", "27259 Rostral Anterior Cingulate (R)",
"27170 Superior Temporal (L)", "27267 Thickness Caudal Anterior Cingulate (R)",
"27178 Thickness Fusiform (L)", "27297 Thickness Insula (R)", "27184 Thickness Lingual (L)",
"27281 Thickness Paracentral (R)", "27191 Thickness Pars Triangularis (L)",
"27287 Thickness Posterior Cingulate (R)", "27198 Thickness Rostral Middle Frontal (L)",
"27294 Thickness Superior Temporal (R)", "27206 Volume Caudal Middle Frontal (L)",
"27302 Volume Fusiform (R)", "27212 Volume Isthmus Cingulate (L)", "27308 Volume Lingual (R)",
"27218 Volume Parahippocampal (L)", "27315 Volume Pars Triangularis (R)", "27226 Volume Precentral (L)",
"27322 Volume Rostral Middle Frontal (R)", "27233 Volume Supramarginal (L)"]

labelsyx = ["T-1 structural image variables"]
#labelsy = [
#"Eid",
"24419 structural variable...",
"24438 Mean absolute head motion from rfMRI",
"24439 Median absolute head motion from rfMRI",
"24440 90th percentile of absolute head motion from rfMRI",
"24441 Mean relative head motion from rfMRI",
"24442 Median relative head motion from rfMRI",
"24443 90th percentile of relative head motion from rfMRI",
"24444 Mean absolute head motion from tfMRI",
"24445 Median absolute head motion from tfMRI",
"24446 90th percentile of absolute head motion from tfMRI",
"24447 Mean relative head motion from tfMRI",
"24448 Median relative head motion from tfMRI",
"24449 90th percentile of relative head motion from tfMRI",
"24450 Mean absolute head motion from dMRI",
"24451 Median absolute head motion from dMRI",
"24452 90th percentile of absolute head motion from dMRI",
"24453 Mean relative head motion from dMRI",
"24454 Median relative head motion from dMRI",
"24455 90th percentile of relative head motion from dMRI",
"25741 Mean rfMRI head motion, averaged across space and time points rfMRI",
"25742 structural variable... "]

x_tick_positions = [i + 0.5 for i in range(len(labelsx))]
y_tick_positions = [i + 0.5 for i in range(len(labelsy))]

plt.xticks(range(len(labelsx)), labelsx, fontsize=7, labelspacing=0.5)
plt.yticks(range(len(labelsy)), labelsy, fontsize=3)

plt.tight_layout()
plt.show()






#Correlaton matrix: N-12XHeadMotion
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the combined dataset
df = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset2.csv", low_memory=False)


# Columns for N-12 and head motion
n_12_columns = ['1920-2.0', '1930-2.0', '1940-2.0', '1950-2.0', '1960-2.0',
                '1970-2.0', '1980-2.0', '1990-2.0', '2000-2.0', '2010-2.0',
                '2020-2.0', '2030-2.0']

head_motion_columns = [
    '24419-2.0', '24438-2.0', '24439-2.0', '24440-2.0', '24441-2.0',
    '24442-2.0', '24443-2.0', '25741-2.0', '24450-2.0', '24451-2.0',
    '24452-2.0', '24453-2.0', '24454-2.0', '24455-2.0', '24444-2.0',
    '24445-2.0', '24446-2.0', '24447-2.0', '24448-2.0', '24449-2.0',
    '25742-2.0'
]

# Calculate correlation matrix for the relevant columns only
correlation_matrix = df[n_12_columns + head_motion_columns].corr()

# Extract the correlation matrix between N-12 and head motion variables
n_12_head_motion_corr = correlation_matrix.loc[n_12_columns, head_motion_columns]

# Custom tick labels for x and y axes
xticklabels = [
    "Measure of T1",
    "Mean abs motion rfMRI",
    "Median abs motion rfMRI",
    "90th %ile abs motion rfMRI",
    "Mean rel motion rfMRI",
    "Median rel motion rfMRI",
    "90th %ile rel motion rfMRI",
    "Mean motion rfMRI",
    "Mean abs motion dMRI",
    "Median abs motion dMRI",
    "90th %ile abs motion dMRI",
    "Mean rel motion dMRI",
    "Median rel motion dMRI",
    "90th %ile rel motion dMRI",
    "Mean abs motion tfMRI",
    "Median abs motion tfMRI",
    "90th %ile abs motion tfMRI",
    "Mean rel motion tfMRI",
    "Median rel motion tfMRI",
    "90th %ile rel motion tfMRI",
    "Mean motion tfMRI"
]

yticklabels = [
    "Mood swings",
    "Miserableness",
    "Irritability",
    "Sensitivity/hurt feelings",
    "Fed-up feelings",
    "Nervous feelings",
    "Worrier/anxious feelings",
    "Tense/’highly strung’",
    "Worry too long after embarrassment",
    "Suffer from ‘nerves’",
    "Loneliness/isolation",
    "Guilty feelings"
]

# Plotting the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(n_12_head_motion_corr, cmap='crest', annot=False, cbar_kws={'label': 'Correlation Coefficient'}, fmt=".2f", xticklabels=xticklabels, yticklabels=yticklabels)
plt.title("Correlation between Neuroticism (N-12) and Head Motion Variables")
plt.xlabel("Head Motion Variables")
plt.ylabel("N-12 Variables")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()




#Correlaton matrix: RDSXHeadMotion
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the combined dataset
df = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset2.csv", low_memory=False)


# Columns for RDS-4 and head motion
#rds_columns = ['2080-2.0']
rds_columns = ['2050-2.0', '2060-2.0', '2070-2.0', '2080-2.0']

head_motion_columns = [
    '24419-2.0', '24438-2.0', '24439-2.0', '24440-2.0', '24441-2.0',
    '24442-2.0', '24443-2.0', '25741-2.0', '24450-2.0', '24451-2.0',
    '24452-2.0', '24453-2.0', '24454-2.0', '24455-2.0', '24444-2.0',
    '24445-2.0', '24446-2.0', '24447-2.0', '24448-2.0', '24449-2.0',
    '25742-2.0'
]

# Calculate correlation matrix for the relevant columns only
correlation_matrix = df[rds_columns + head_motion_columns].corr()

# Extract the correlation matrix between N-12 and head motion variables
rds_head_motion_corr = correlation_matrix.loc[rds_columns, head_motion_columns]

# Custom tick labels for x and y axes
xticklabels = [
    "Measure of T1",
    "Mean abs motion rfMRI",
    "Median abs motion rfMRI",
    "90th %ile abs motion rfMRI",
    "Mean rel motion rfMRI",
    "Median rel motion rfMRI",
    "90th %ile rel motion rfMRI",
    "Mean motion rfMRI avg",
    "Mean abs motion dMRI",
    "Median abs motion dMRI",
    "90th %ile abs motion dMRI",
    "Mean rel motion dMRI",
    "Median rel motion dMRI",
    "90th %ile rel motion dMRI",
    "Mean abs motion tfMRI",
    "Median abs motion tfMRI",
    "90th %ile abs motion tfMRI",
    "Mean rel motion tfMRI",
    "Median rel motion tfMRI",
    "90th %ile rel motion tfMRI",
    "Mean motion tfMRI avg"
]

#yticklabels = [#
    #"Tiredness/lethargy (2080)"
#]

yticklabels = [
    "Depressed mood (2050)",
    "Unenthusiasm (2060)",
    "Tenseness/restlessness (2070)",
    "Tiredness/lethargy (2080)",
]

# Plotting the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(rds_head_motion_corr, cmap='crest', annot=False, cbar_kws={'label': 'Correlation Coefficient'}, fmt=".2f", xticklabels=xticklabels, yticklabels=yticklabels, vmin=-0.01,vmax=0.08)
plt.title("Correlation between Depression (RDS-4) and Head Motion Variables")
plt.xlabel("Head Motion Variables")
plt.ylabel("RDS-4 Variables")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()





#Correlaton matrix: IQ X HeadMotion
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the combined dataset
df = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset2.csv", low_memory=False)


# Columns for IQ and head motion
iq_columns = ['20016-2.0']

head_motion_columns = [
    '24419-2.0', '24438-2.0', '24439-2.0', '24440-2.0', '24441-2.0',
    '24442-2.0', '24443-2.0', '25741-2.0', '24450-2.0', '24451-2.0',
    '24452-2.0', '24453-2.0', '24454-2.0', '24455-2.0', '24444-2.0',
    '24445-2.0', '24446-2.0', '24447-2.0', '24448-2.0', '24449-2.0',
    '25742-2.0'
]

# Calculate correlation matrix for the relevant columns only
correlation_matrix = df[iq_columns + head_motion_columns].corr()

# Extract the correlation matrix between N-12 and head motion variables
iq_head_motion_corr = correlation_matrix.loc[iq_columns, head_motion_columns]

# Custom tick labels for x and y axes
xticklabels = [
    "Measure of T1",
    "Mean abs motion rfMRI",
    "Median abs motion rfMRI",
    "90th %ile abs motion rfMRI",
    "Mean rel motion rfMRI",
    "Median rel motion rfMRI",
    "90th %ile rel motion rfMRI",
    "Mean motion rfMRI",
    "Mean abs motion dMRI",
    "Median abs motion dMRI",
    "90th %ile abs motion dMRI",
    "Mean rel motion dMRI",
    "Median rel motion dMRI",
    "90th %ile rel motion dMRI",
    "Mean abs motion tfMRI",
    "Median abs motion tfMRI",
    "90th %ile abs motion tfMRI",
    "Mean rel motion tfMRI",
    "Median rel motion tfMRI",
    "90th %ile rel motion tfMRI",
    "Mean motion tfMRI"
]

yticklabels = [
    "Fluid intelligence score (20016)"
]

# Plotting the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(iq_head_motion_corr, cmap='crest', annot=False, cbar_kws={'label': 'Correlation Coefficient'}, fmt=".2f", xticklabels=xticklabels, yticklabels=yticklabels)
plt.title("Correlation between IQ and Head Motion Variables")
plt.xlabel("Head Motion Variables")
plt.ylabel("IQ Variable")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()





#Correlaton matrix: Reaction time X HeadMotion
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the combined dataset
df = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset3.csv", low_memory=False)

# Columns for reaction time and head motion
rxnt_columns = ['20023-2.0']

# Correct order of head motion columns
head_motion_columns = [
    '24419-2.0', '24438-2.0', '24439-2.0', '24440-2.0', '24441-2.0',
    '24442-2.0', '24443-2.0', '25741-2.0', '24450-2.0', '24451-2.0',
    '24452-2.0', '24453-2.0', '24454-2.0', '24455-2.0', '24444-2.0',
    '24445-2.0', '24446-2.0', '24447-2.0', '24448-2.0', '24449-2.0',
    '25742-2.0'
]

# Correct order of xticklabels to match the head_motion_columns
xticklabels = [
    "Measure of T1",
    "Mean abs motion rfMRI",
    "Median abs motion rfMRI",
    "90th %ile abs motion rfMRI",
    "Mean rel motion rfMRI",
    "Median rel motion rfMRI",
    "90th %ile rel motion rfMRI",
    "Mean motion rfMRI",
    "Mean abs motion dMRI",
    "Median abs motion dMRI",
    "90th %ile abs motion dMRI",
    "Mean rel motion dMRI",
    "Median rel motion dMRI",
    "90th %ile rel motion dMRI",
    "Mean abs motion tfMRI",
    "Median abs motion tfMRI",
    "90th %ile abs motion tfMRI",
    "Mean rel motion tfMRI",
    "Median rel motion tfMRI",
    "90th %ile rel motion tfMRI",
    "Mean motion tfMRI"
]

yticklabels = ["Reaction time (20023)"]

# Calculate correlation matrix for the relevant columns only
correlation_matrix = df[rxnt_columns + head_motion_columns].corr()

# Extract the correlation matrix between N-12 and head motion variables
rxnt_head_motion_corr = correlation_matrix.loc[rxnt_columns, head_motion_columns]

# Plotting the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(rxnt_head_motion_corr, cmap='crest', annot=False, cbar_kws={'label': 'Correlation Coefficient'}, fmt=".2f", xticklabels=xticklabels, yticklabels=yticklabels)
plt.title("Correlation between Reaction time and Head Motion Variables")
plt.xlabel("Head Motion Variables")
plt.ylabel("Reaction time Variable")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()




#Correlaton matrix: ControlsXHeadMotion
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the combined dataset
df = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset2.csv", low_memory=False)


# Columns for controls (age, sex, site, icv) and head motion
ctrl_columns = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0']

head_motion_columns = [
    '24419-2.0', '24438-2.0', '24439-2.0', '24440-2.0', '24441-2.0',
    '24442-2.0', '24443-2.0', '25741-2.0', '24450-2.0', '24451-2.0',
    '24452-2.0', '24453-2.0', '24454-2.0', '24455-2.0', '24444-2.0',
    '24445-2.0', '24446-2.0', '24447-2.0', '24448-2.0', '24449-2.0',
    '25742-2.0'
]

# Calculate correlation matrix for the relevant columns only
correlation_matrix = df[ctrl_columns + head_motion_columns].corr()

# Extract the correlation matrix between N-12 and head motion variables
ctrl_head_motion_corr = correlation_matrix.loc[ctrl_columns, head_motion_columns]

# Custom tick labels for x and y axes
xticklabels = [
    "Measure of T1",
    "Mean abs motion rfMRI",
    "Median abs motion rfMRI",
    "90th %ile abs motion rfMRI",
    "Mean rel motion rfMRI",
    "Median rel motion rfMRI",
    "90th %ile rel motion rfMRI",
    "Mean motion rfMRI",
    "Mean abs motion dMRI",
    "Median abs motion dMRI",
    "90th %ile abs motion dMRI",
    "Mean rel motion dMRI",
    "Median rel motion dMRI",
    "90th %ile rel motion dMRI",
    "Mean abs motion tfMRI",
    "Median abs motion tfMRI",
    "90th %ile abs motion tfMRI",
    "Mean rel motion tfMRI",
    "Median rel motion tfMRI",
    "90th %ile rel motion tfMRI",
    "Mean motion tfMRI"
]

yticklabels = [
    "Age (21003)",
    "Sex (31)",
    "Site (54)",
    "Intracranial volume (26521)",
]

# Plotting the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(ctrl_head_motion_corr, cmap='crest', annot=False, cbar_kws={'label': 'Correlation Coefficient'}, fmt=".2f", xticklabels=xticklabels, yticklabels=yticklabels)
plt.title("Correlation between Controls and Head Motion Variables")
plt.xlabel("Head Motion Variables")
plt.ylabel("Control Variables")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()






# Correlation Matrix: All phenotypes, controls, and head motion variables
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the combined dataset
df = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset3.csv", low_memory=False)

# Columns for controls (age, sex, site, icv), all phenotypes, and head motion
ctrl_columns = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0']

# Columns for depression (RDS-4)
rds_columns = ['2050-2.0', '2060-2.0', '2070-2.0', '2080-2.0']

# Columns for N-12
n_12_columns = ['1920-2.0', '1930-2.0', '1940-2.0', '1950-2.0', '1960-2.0',
                '1970-2.0', '1980-2.0', '1990-2.0', '2000-2.0', '2010-2.0',
                '2020-2.0', '2030-2.0']

# Column for reaction time
rxnt_columns = ['20023-2.0']
# Column for IQ
iq_columns = ['20016-2.0']

# Combine all relevant columns
all_columns = ctrl_columns + rds_columns + n_12_columns + rxnt_columns + iq_columns + head_motion_columns

# Calculate correlation matrix for the relevant columns only
correlation_matrix = df[all_columns].corr()

# Extract the correlation matrix between N-12 and head motion variables
ctrl_head_motion_corr = correlation_matrix.loc[ctrl_columns + rds_columns + n_12_columns + rxnt_columns + iq_columns, head_motion_columns]

# Custom tick labels for x and y axes
xticklabels = [
    "Measure of T1",
    "Mean abs motion rfMRI",
    "Median abs motion rfMRI",
    "90th %ile abs motion rfMRI",
    "Mean rel motion rfMRI",
    "Median rel motion rfMRI",
    "90th %ile rel motion rfMRI",
    "Mean motion rfMRI avg",
    "Mean abs motion dMRI",
    "Median abs motion dMRI",
    "90th %ile abs motion dMRI",
    "Mean rel motion dMRI",
    "Median rel motion dMRI",
    "90th %ile rel motion dMRI",
    "Mean abs motion tfMRI",
    "Median abs motion tfMRI",
    "90th %ile abs motion tfMRI",
    "Mean rel motion tfMRI",
    "Median rel motion tfMRI",
    "90th %ile rel motion tfMRI",
    "Mean motion tfMRI avg"
]

yticklabels = [
    "Age (21003)",
    "Sex (31)",
    "Site (54)",
    "Intracranial volume (26521)",
    "Depression (2050)",
    "Depression (2060)",
    "Depression (2070)",
    "Depression (2080)",
    "N-12 (1920)",
    "N-12 (1930)",
    "N-12 (1940)",
    "N-12 (1950)",
    "N-12 (1960)",
    "N-12 (1970)",
    "N-12 (1980)",
    "N-12 (1990)",
    "N-12 (2000)",
    "N-12 (2010)",
    "N-12 (2020)",
    "N-12 (2030)",
    "Reaction Time (20023)",
    "IQ (20016)"
]

# Plotting the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(ctrl_head_motion_corr, cmap='crest', annot=False, cbar_kws={'label': 'Correlation Coefficient'}, fmt=".2f", xticklabels=xticklabels, yticklabels=yticklabels, vmin= -0.2, vmax= 0.2)
plt.title("Correlation between all Phenotypes, Controls, and Head Motion Variables")
plt.xlabel("Head Motion Variables")
plt.ylabel("Variables")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()







# New funpack variables extracted
Behavior = pd.read_csv("/Users/Cabria/Downloads/Behavior.csv", low_memory=False)
# Find the first column with non-empty data (assuming row 0 is the header)
first_non_empty_col = Behavior.columns.get_loc(Behavior.iloc[0].first_valid_index())
# Assuming columns A and B are the first two columns, you want to check columns C to V for emptiness
columns_to_check = Behavior.columns[first_non_empty_col:]
columns_to_check = columns_to_check[:23]  # Check columns C to W
# Create a boolean mask to identify rows where A and B are filled but C to W are empty
mask = Behavior.apply(lambda row: not pd.isna(row.iloc[0]) and not pd.isna(row.iloc[1]) and all(pd.isna(row.iloc[2:])),
axis=1)
# Keep rows where the condition is not met
Behavior = Behavior[~mask]
# Check for -1 and -3 and replace them with NaN
Behavior = Behavior.replace([-1, -3], [pd.NA, pd.NA])
# Save the cleaned data to a new CSV file
Behavior.to_csv('cleaned_data.csv', index=False)
Behavior.to_csv('/Users/Cabria/Downloads/cleaned_Behavior.csv', index=False)
UpdatedBehavior = pd.read_csv("/Users/Cabria/Downloads/cleaned_Behavior.csv", low_memory=False)
# Remove NaN values from csv
UpdatedBehavior.dropna()




#Convert tsv file to csv file
import pandas as pd
# Convert tsv file to csv file
tsv_file_path = r"\Users\Cabria\Downloads\behavior.tsv"
csv_table = pd.read_csv(tsv_file_path, sep='\t')
csv_table.to_csv('behavior.csv', index=False)
#Saved new file
csv_table.to_csv(r'C:/Users/Cabria/Downloads/Behavior.csv', index=False)
behavior= pd.read_csv("/Users/Cabria/Downloads/Behavior.csv", low_memory=False)
#New and cleaned file
columns_to_replace = ['eid', '2010-2.0']
Neuro[columns_to_replace] = Neuro[columns_to_replace].replace([-3, -1], [pd.NA, pd.NA])
Neuro.to_csv("/Users/Cabria/Downloads/cleaned_Neuroticism.csv", index=False)




#More cleaning of neuroticism file
import pandas as pd
neuroticism = pd.read_csv("/Users/Cabria/Downloads/Head Motion Project/cleaned_Neuroticism.csv", low_memory=False)
# Create a boolean mask to identify rows where A is filled but B is empty
mask = neuroticism.apply(lambda row: not pd.isna(row.iloc[0]) and pd.isna(row.iloc[1]), axis=1)
# Keep rows where the condition is not met
neuroticism = neuroticism[~mask]
# Check for -1 and -3 and replace them with NaN
neuroticism = neuroticism.replace([-1, -3], pd.NA)
# Save the cleaned data to a new CSV file
neuroticism.to_csv("/Users/Cabria/Downloads/cleaned_Neuroticism.csv", index=False)
# Read the cleaned CSV file
Updatedneuroticism = pd.read_csv("/Users/Cabria/Downloads/cleaned_Neuroticism.csv", low_memory=False)
# Remove NaN values from the DataFrame
Updatedneuroticism.dropna(inplace=True)




#Combine datasets to prepare for loop
import pandas as pd
dataset1=pd.read_csv("Dataset1.csv")
dataset2=pd.read_csv("Dataset2.csv")
combinedDataset=pd.merge(dataset1,dataset2, on='eid', how='outer')
combinedDataset.shape
combinedDataset.head(10)
#You can only merge two data sets at time
dataset1 = pd.read_csv("/Users/Cabria/Downloads/Reactiontime.csv", low_memory=False)
dataset2 = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset2.csv", low_memory=False)
dataset3 = pd.read_csv("/Users/Cabria/Downloads/Head Motion Project/cleaned_Behavior.csv", low_memory=False)
combinedHeadmotionproject=pd.merge(dataset1,dataset2, on='eid', how='outer')


#Save the file
combinedDataset.to_csv("CombinedDataset3.csv", index=False)
#Combine new file to data set 3 now
dataset1 = pd.read_csv("C:/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset.csv", low_memory=False)
dataset2 = pd.read_csv("/Users/Cabria/Downloads/Head Motion Project/cleaned_Behavior.csv", low_memory=False)
combinedHeadmotionproject2 = pd.merge(dataset1,dataset2, on='eid', how='outer')
#Save the 3/4 merged file and repeat
combinedHeadmotionproject2.to_csv("CombinedDataset2.csv", index=False)


#Combine datasets to do linear regressions
import pandas as pd
dataset1 = pd.read_csv("/Users/Cabria/Downloads/Reactiontime.csv", low_memory=False)
dataset2 = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset2.csv", low_memory=False)
combinedHeadmotionproject=pd.merge(dataset1,dataset2, on='eid', how='outer')
#Save the file
combinedHeadmotionproject.to_csv("C:/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset3.csv", index=False)



#Simple linear regression for one DKT variable
import pandas as pd
import numpy as np
from scipy.stats import linregress
import statsmodels.api as sm
df = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset2.csv", low_memory=False)
# Drop rows with NaN values in any of the specified columns
df = df.dropna(subset=['27143-2.0', '2050-2.0', '2060-2.0', '2070-2.0', '2080-2.0'])

# Calculate the sum column 'RDS'
df['RDS'] = df[['2050-2.0', '2060-2.0', '2070-2.0', '2080-2.0']].sum(axis=1)

# Perform linear regression
x = df['27143-2.0']
y = df['RDS']
slope, intercept, r_value, p_value, std_err = linregress(x, y)

# Create a new DataFrame for results
results = pd.DataFrame({
    'Variable': ['Intercept', '27143-2.0'],
    'Coefficient': [intercept, slope],
    'p-value': [np.nan, p_value],
    'R-squared': [np.nan, r_value**2]
})

# Display the regression results
print("Regression Results:")
print(results)

# Perform detailed linear regression using statsmodels
x = sm.add_constant(x)
model = sm.OLS(y, x).fit()

# Display detailed regression summary
print("\nDetailed Regression Summary:")
print(model.summary())





#(New) Linear regression for loop with head motion variables (adjusting x and y variables)
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset2.csv", low_memory=False)

# Drop rows with NaN values
df = df.dropna()

# Calculate 'RDS' variable and demean it
df['RDS'] = df[['2050-2.0', '2060-2.0', '2070-2.0', '2080-2.0']].sum(axis=1)
df['RDS'] = df['RDS'] - df['RDS'].mean()

# Demean head motion variables
headmotion_vars = ['24419-2.0', '24438-2.0', '24439-2.0', '24440-2.0', '24441-2.0', '24442-2.0', '24443-2.0', '24444-2.0', '24445-2.0', '24446-2.0', '24447-2.0', '24448-2.0', '24449-2.0', '24450-2.0', '24451-2.0', '24452-2.0', '24453-2.0', '24454-2.0', '24455-2.0', '25741-2.0', '25742-2.0']
df[headmotion_vars] = df[headmotion_vars] - df[headmotion_vars].mean()

# Demean other covariates (age, sex, site, icv)
covariates = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0']
df[covariates] = df[covariates] - df[covariates].mean()

# Initialize an empty list to store results
results_list = []

# Get the index positions for DKT variables
start_index = df.columns.get_loc('27143-2.0')
end_index = df.columns.get_loc('27327-2.0')

# Perform linear regression for each DKT variable
for dkt_var in df.iloc[:, start_index:end_index + 1]:
    df['DKT'] = df[dkt_var] - df[dkt_var].mean()  # Create a demeaned DKT variable for the current column

    # Prepare the independent variables: DKT, covariates, and head motion variables
    all_vars = ['DKT'] + covariates + headmotion_vars
    X = df[all_vars]
    X = sm.add_constant(X)  # Add a constant term for the intercept

    # The dependent variable is 'RDS'
    Y = df['RDS']

    # Perform linear regression
    model = sm.OLS(Y, X)
    results = model.fit()

    # Append results to the list
    results_list.append({'DKT Variable': dkt_var, 'Coefficient': results.params['DKT'], 'p-value': results.pvalues['DKT']})

# Create a DataFrame from the list
Table2 = pd.DataFrame(results_list)

# Display the results
print(Table2)

# Save the DataFrame to a CSV file
Table2.to_csv('results_with_dkt_p_values_with_head_motion.csv', index=False)







#Old for loop linear regression adjusted for RDS without head motion (checking params)
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset2.csv", low_memory=False)

# Drop rows with NaN values
df = df.dropna()

# Calculate 'RDS' variable and demean it
df['RDS'] = df[['2050-2.0', '2060-2.0', '2070-2.0', '2080-2.0']].sum(axis=1)
df['RDS'] = df['RDS'] - df['RDS'].mean()

# Define covariates (age, sex, site, icv)
covariates = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0']

# Initialize an empty list to store results
results_list = []

# Get the index positions for DKT variables
start_index = df.columns.get_loc('27143-2.0')
end_index = df.columns.get_loc('27327-2.0')


# Perform linear regression for each DKT variable
for dkt_var in df.iloc[:, start_index:end_index + 1]:
    DKT = df[dkt_var] - df[dkt_var].mean()

    DKT_df = pd.DataFrame(DKT, columns=[dkt_var])
    covariates_and_RDS = pd.concat(df[covariates] - df[covariates].mean(), [df['RDS']], axis=1)  # Combine covariates and RDS

    # Perform linear regression without a constant
    model = sm.OLS(DKT_df, covariates_and_RDS, hasconst=True)
    results = model.fit()

    # Append results to the list
    results_list.append({'DKT Variable': dkt_var, 'B': results.params[4], 'p-value': results.pvalues[1]})

# Create a DataFrame from the list
Table2 = pd.DataFrame(results_list)

# Display the results
print(Table2)

# Save the DataFrame to a CSV file
Table2.to_csv('results_with_dkt_p_values_no_head_motion.csv', index=False)




#Correcting for x and y values without head motion
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset3.csv", low_memory=False)
# Drop rows with NaN values
df = df.dropna()

# Calculate 'RDS' variable and demean it
df['RDS'] = df[['2050-2.0', '2060-2.0', '2070-2.0', '2080-2.0']].sum(axis=1)
df['RDS'] = df['RDS'] - df['RDS'].mean()

# Define covariates (age, sex, site, icv) and demean them
covariates = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0']
df[covariates] = df[covariates] - df[covariates].mean()

# Initialize an empty list to store results
results_list = []

# Get the index positions for DKT variables
start_index = df.columns.get_loc('27143-2.0')
end_index = df.columns.get_loc('27327-2.0')

# Perform linear regression for each DKT variable
for dkt_var in df.iloc[:, start_index:end_index + 1]:
    df['DKT'] = df[dkt_var] - df[dkt_var].mean()  # Creates a demeaned DKT variable for the current column

    # Prepare the independent variables: DKT and covariates
    X = df[['DKT'] + covariates]
    X = sm.add_constant(X)  # Add a constant term for the intercept

    # The dependent variable is 'RDS'
    Y = df['RDS']

    # Perform linear regression
    model = sm.OLS(Y, X)
    results = model.fit()

    # Append results to the list
    results_list.append({'DKT Variable': dkt_var, 'Coefficient': results.params['DKT'], 'p-value': results.pvalues['DKT']})

# Create a DataFrame from the list
Table2 = pd.DataFrame(results_list)

# Display the results
print(Table2)

# Save the DataFrame to a CSV file
Table2.to_csv('results_with_dkt_p_values_no_head_motion3.csv', index=False)


#Checking one DKT variable to see if getting the same results
import pandas as pd
import statsmodels.api as sm
df = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset2.csv", low_memory=False)
# Drop rows with NaN values
df = df.dropna()
# Calculate 'RDS' variable and demean it
df['RDS'] = df[['2050-2.0', '2060-2.0', '2070-2.0', '2080-2.0']].sum(axis=1)
df['RDS'] = df['RDS'] - df['RDS'].mean()
# Define covariates (age, sex, site, icv)
covariates = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0']
# Initialize an empty list to store results
results_list = []
# Get the index position for DKT variable '27142-2.0'
dkt_var = '27143-2.0'
DKT = df[dkt_var] - df[dkt_var].mean()
DKT_df = pd.DataFrame(DKT, columns=[dkt_var])
results_list.append({'DKT Variable': dkt_var, 'B': results.params['RDS'], 'p-value': results.pvalues['RDS']}) # Combine covariates and RDS
# Perform linear regression without a constant
model = sm.OLS(DKT_df, sm.add_constant(covariates_and_RDS), hasconst=True)
results = model.fit()
# Display detailed regression summary
print("\nDetailed Regression Summary for DKT Variable", dkt_var)
print(results.summary())
# Save the results to the list
results_list.append({'DKT Variable': dkt_var, 'B': results.params[4], 'p-value': results.pvalues[4]})
# Create a DataFrame from the list
Table2 = pd.DataFrame(results_list)
# Display the results
print("\nTable2:")
print(Table2)
# Save the DataFrame to a CSV file
Table2.to_csv('Table2.csv', index=False)


#Accounting for heteroskediscity
import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset2.csv", low_memory=False)

# Drop rows with NaN values
df = df.dropna()

# Calculate 'RDS' variable and then demean it
df['RDS'] = df[['2050-2.0', '2060-2.0', '2070-2.0', '2080-2.0']].sum(axis=1)
df['RDS_mean'] = df['RDS'].mean()

# Sampling based on RDS values (assuming continuous values need categorization or adjust the condition for exact matches)
df_RDS_1 = df[(df['RDS'] == 1)].sample(frac=0.25, random_state=1)  # Change random_state for reproducibility
df_RDS_2 = df[(df['RDS'] == 2)].sample(frac=0.50, random_state=1)
df_rest = df[~df['RDS'].isin([1, 2])]

# Concatenate the sampled and rest of the data
df = pd.concat([df_RDS_1, df_RDS_2, df_rest])

# Demean head motion variables
headmotion_vars = ['24419-2.0', '24438-2.0', '24439-2.0', '24440-2.0', '24441-2.0', '24442-2.0', '24443-2.0', '24444-2.0', '24445-2.0', '24446-2.0', '24447-2.0', '24448-2.0', '24449-2.0', '24450-2.0', '24451-2.0', '24452-2.0', '24453-2.0', '24454-2.0', '24455-2.0', '25741-2.0', '25742-2.0']
df[headmotion_vars] = df[headmotion_vars].sub(df[headmotion_vars].mean())

# Demean other covariates (age, sex, site, icv)
covariates = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0']
df[covariates] = df[covariates].sub(df[covariates].mean())

# Initialize an empty list to store results
results_list = []

# Get the index positions for DKT variables
start_index = df.columns.get_loc('27143-2.0')
end_index = df.columns.get_loc('27327-2.0')

# Perform linear regression for each DKT variable
for dkt_var in df.iloc[:, start_index:end_index + 1].columns:
    # Create a demeaned DKT variable for the current column
    df['DKT'] = df[dkt_var] - df[dkt_var].mean()

    # Prepare the independent variables: DKT, covariates, and head motion variables
    all_vars = ['DKT'] + covariates + headmotion_vars
    X = df[all_vars]
    X = sm.add_constant(X)  # Add a constant term for the intercept

    # The dependent variable is 'RDS'
    Y = df['RDS']

    # Perform linear regression
    model = sm.OLS(Y, X)
    results = model.fit()

    # Append results to the list
    results_list.append({'DKT Variable': dkt_var, 'Coefficient': results.params['DKT'], 'p-value': results.pvalues['DKT']})

# Create a DataFrame from the list
Table2 = pd.DataFrame(results_list)

# Display the results
print(Table2)

# Save the DataFrame to a CSV file
Table2.to_csv('results_with_dkt_p_values_with_head_motion4.csv', index=False)







#Bonferroni method from above script (use this one)
import pandas as pd
import numpy as np

data_no_motion = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/results_with_dkt_p_values_no_head_motion3.csv", low_memory=False)
data_with_motion = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/results_with_dkt_p_values_with_head_motion4.csv", low_memory=False)

alpha = 0.05  #significance level

# define Bonferroni correction function and params
def apply_bonferroni_correction(data, p_value_col='p-value', alpha=0.05):
    n_tests = len(data[p_value_col])
    corrected_alpha = alpha / n_tests  # Corrected significance level
    # Creates new column for significant values after correction
    data['Significant After Correction'] = data[p_value_col] < corrected_alpha
    return data

# Apply the function to both datasets
data_with_motion = apply_bonferroni_correction(data_with_motion, 'p-value', alpha)
data_no_motion = apply_bonferroni_correction(data_no_motion, 'p-value', alpha)

# Save out the datasets to new CSV files
new_file_path_with_motion = '/Users/Cabria/PycharmProjects/pythonProject/RDS_results_with_correction_with_motion2.csv'
new_file_path_no_motion = '/Users/Cabria/PycharmProjects/pythonProject/RDS_results_with_correction_no_motion2.csv'

data_with_motion.to_csv(new_file_path_with_motion, index=False)
data_no_motion.to_csv(new_file_path_no_motion, index=False)






#(different method)Bonferroni Correction for p-values on RDS linear regression output
import pandas as pd

data_with_motion = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/results_with_dkt_p_values_no_head_motion3.csv", low_memory=False)
data_no_motion = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/results_with_dkt_p_values_with_head_motion4.csv", low_memory=False)

# Calculate the Bonferroni corrected significance level
num_tests_with_motion = len(data_with_motion['p-value'])
num_tests_no_motion = len(data_no_motion['p-value'])

alpha = 0.05  # Original significance level
corrected_alpha_with_motion = alpha / num_tests_with_motion
corrected_alpha_no_motion = alpha / num_tests_no_motion

# Apply Bonferroni Correction and determine significance
significant_with_motion = data_with_motion['p-value'] < corrected_alpha_with_motion
significant_no_motion = data_no_motion['p-value'] < corrected_alpha_no_motion

# Add a "Significant After Correction" column to indicate significant p-values after correction
data_with_motion['Significant After Correction'] = significant_with_motion
data_no_motion['Significant After Correction'] = significant_no_motion

# Save the datasets to new CSV files
new_file_path_with_motion = '/Users/Cabria/PycharmProjects/pythonProject/RDS_results_with_correction_with_motion.csv'
new_file_path_no_motion = '/Users/Cabria/PycharmProjects/pythonProject/RDS_results_with_correction_no_motion.csv'

data_with_motion.to_csv(new_file_path_with_motion, index=False)
data_no_motion.to_csv(new_file_path_no_motion, index=False)


# Extras....(Converts boolean to more descriptive text (than just true/false))
    #data['Significant After Correction'] = np.where(data['Significant After Correction'], 'Significant', 'Not Significant')





#New subjects with depression
import pandas as pd
df_main = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset3.csv", low_memory=False)

# Display the first few rows of the dataset
print(df_main.head())

# Load the list of subjects for Split A
split_a_subjects = pd.read_csv('C:/Users/Cabria/Downloads/SplitAsubjects.txt', header=None, names=['eid'])

# Display the first few rows to confirm
print(split_a_subjects.head())

# Filter the main dataset for subjects in Split A
df_split_a = df_main[df_main['eid'].isin(split_a_subjects['eid'])]

# Display the first few rows of the filtered dataset
print(df_split_a.head())




#Running linear regression on groups A and B using singular head motion variables
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load the main dataset
df_main = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset3.csv", low_memory=False)
split_a_subjects = pd.read_csv('C:/Users/Cabria/Downloads/SplitAsubjects.txt', header=None, names=['eid'])
split_b_subjects = pd.read_csv('C:/Users/Cabria/Downloads/SplitBsubjects.txt', header=None, names=['eid'])

# Assuming 'RDS' calculation and demeaning is correct
df_main['RDS'] = df_main[['2050-2.0', '2060-2.0', '2070-2.0', '2080-2.0']].sum(axis=1)
df_main['RDS'] = df_main['RDS'] - df_main['RDS'].mean()

# Corrected DataFrame reference for start_index and end_index
start_index = df_main.columns.get_loc('27143-2.0')
end_index = df_main.columns.get_loc('27327-2.0')

# Adjusted function with NaN/Inf checks
def perform_analysis_and_save(df_main, subject_list, file_name):
    df_filtered = df_main[df_main['eid'].isin(subject_list['eid'])]
    df_filtered = df_filtered.dropna(subset=['25741-2.0', '21003-2.0', '31-0.0', '54-2.0', '26521-2.0', 'RDS'])

    results_list = []
    for dkt_var in df_filtered.columns[start_index:end_index + 1]:
        df_filtered['DKT'] = df_filtered[dkt_var] - df_filtered[dkt_var].mean()

        # Check for NaNs or infinite values after calculating 'DKT'
        if df_filtered['DKT'].isnull().any() or np.isinf(df_filtered['DKT']).any():
            print(f"NaNs or infinite values detected in DKT calculations for variable {dkt_var}.")
            continue

        covariates = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0']
        all_vars = ['DKT'] + covariates + ['25741-2.0']
        X = df_filtered[all_vars]
        X = sm.add_constant(X)

        # Additional check for NaNs or Inf values in X
        if X.isnull().any().any() or np.isinf(X).any().any():
            print(f"NaNs or infinite values detected in X for variable {dkt_var}.")
            continue

        Y = df_filtered['RDS']
        model = sm.OLS(Y, X).fit()
        results_list.append({'Variable': dkt_var, 'Coefficient': model.params['DKT'], 'p-value': model.pvalues['DKT']})

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(file_name, index=False)
    return results_df

# Adjusted paths for saving the results
new_file_path_subject_A = '/Users/Cabria/PycharmProjects/pythonProject/RDS_results_split_A.csv'
new_file_path_subject_B = '/Users/Cabria/PycharmProjects/pythonProject/RDS_results_split_B.csv'

# Perform the analysis for Split A and Split B, then save the results
results_a = perform_analysis_and_save(df_main, split_a_subjects, new_file_path_subject_A)
print(f"Results for Split A saved to {new_file_path_subject_A}.")

results_b = perform_analysis_and_save(df_main, split_b_subjects, new_file_path_subject_B)
print(f"Results for Split B saved to {new_file_path_subject_B}.")






#Running linear regression on groups A and B without head motion variables
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load the main dataset
df_main = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset3.csv", low_memory=False)
split_a_subjects = pd.read_csv('C:/Users/Cabria/Downloads/SplitAsubjects.txt', header=None, names=['eid'])
split_b_subjects = pd.read_csv('C:/Users/Cabria/Downloads/SplitBsubjects.txt', header=None, names=['eid'])

# Assuming 'RDS' calculation and demeaning is correct
df_main['RDS'] = df_main[['2050-2.0', '2060-2.0', '2070-2.0', '2080-2.0']].sum(axis=1)
df_main['RDS'] = df_main['RDS'] - df_main['RDS'].mean()

# Corrected DataFrame reference for start_index and end_index
start_index = df_main.columns.get_loc('27143-2.0')
end_index = df_main.columns.get_loc('27327-2.0')

# Adjusted function with NaN/Inf checks
def perform_analysis_and_save(df_main, subject_list, file_name):
    df_filtered = df_main[df_main['eid'].isin(subject_list['eid'])]
    df_filtered = df_filtered.dropna(subset=['21003-2.0', '31-0.0', '54-2.0', '26521-2.0', 'RDS'])

    results_list = []
    for dkt_var in df_filtered.columns[start_index:end_index + 1]:
        df_filtered['DKT'] = df_filtered[dkt_var] - df_filtered[dkt_var].mean()

        # Check for NaNs or infinite values after calculating 'DKT'
        if df_filtered['DKT'].isnull().any() or np.isinf(df_filtered['DKT']).any():
            print(f"NaNs or infinite values detected in DKT calculations for variable {dkt_var}.")
            continue

        covariates = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0']
        all_vars = ['DKT'] + covariates
        X = df_filtered[all_vars]
        X = sm.add_constant(X)

        # Additional check for NaNs or Inf values in X
        if X.isnull().any().any() or np.isinf(X).any().any():
            print(f"NaNs or infinite values detected in X for variable {dkt_var}.")
            continue

        Y = df_filtered['RDS']
        model = sm.OLS(Y, X).fit()
        results_list.append({'Variable': dkt_var, 'Coefficient': model.params['DKT'], 'p-value': model.pvalues['DKT']})

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(file_name, index=False)
    return results_df

# Adjusted paths for saving the results
new_file_path_subject_A_no_motion = '/Users/Cabria/PycharmProjects/pythonProject/RDS_results_split_A_no_motion.csv'
new_file_path_subject_B_no_motion = '/Users/Cabria/PycharmProjects/pythonProject/RDS_results_split_B_no_motion.csv'

# Perform the analysis for Split A and Split B, then save the results
results_a = perform_analysis_and_save(df_main, split_a_subjects, new_file_path_subject_A_no_motion)
print(f"Results for Split A saved to {new_file_path_subject_A_no_motion}.")

results_b = perform_analysis_and_save(df_main, split_b_subjects, new_file_path_subject_B_no_motion)
print(f"Results for Split B saved to {new_file_path_subject_B_no_motion}.")




#Running linear regression on whole group using singular head motion variable
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load the main dataset
df_main = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset3.csv", low_memory=False)

# Drop rows with NaN values
df_main = df_main.dropna()

#'RDS' calculation and demeaning
df_main['RDS'] = df_main[['2050-2.0', '2060-2.0', '2070-2.0', '2080-2.0']].sum(axis=1)
df_main['RDS'] = df_main['RDS'] - df_main['RDS'].mean()

# Corrected DataFrame reference for start_index and end_index
start_index = df_main.columns.get_loc('27143-2.0')
end_index = df_main.columns.get_loc('27327-2.0')

# Adjusted function with NaN/Inf checks
def perform_analysis_and_save(df_main, file_name):

    results_list = []
    for dkt_var in df_main.columns[start_index:end_index + 1]:
        df_main['DKT'] = df_main[dkt_var] - df_main[dkt_var].mean()

        # Check for NaNs or infinite values after calculating 'DKT'
        if df_main['DKT'].isnull().any() or np.isinf(df_main['DKT']).any():
            print(f"NaNs or infinite values detected in DKT calculations for variable {dkt_var}.")
            continue

        covariates = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0']
        all_vars = ['DKT'] + covariates + ['25741-2.0']
        X = df_main[all_vars]
        X = sm.add_constant(X)

        # Additional check for NaNs or Inf values in X
        if X.isnull().any().any() or np.isinf(X).any().any():
            print(f"NaNs or infinite values detected in X for variable {dkt_var}.")
            continue

        Y = df_main['RDS']
        model = sm.OLS(Y, X).fit()
        results_list.append({'Variable': dkt_var, 'Coefficient': model.params['DKT'], 'p-value': model.pvalues['DKT']})

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(file_name, index=False)
    return results_df

# Adjusted paths for saving the results
new_file_path_whole_group = '/Users/Cabria/PycharmProjects/pythonProject/RDS_results_whole_group_singular_HM.csv'


# Perform the analysis for whole group, then save the results
results = perform_analysis_and_save(df_main, new_file_path_whole_group)
print(f"Whole group linear regression + 25741-2.0 {new_file_path_whole_group}.")



#Running linear regression on whole group using singular head motion variable and adding ZSCORE
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import zscore

# Load the main dataset
df_main = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset3.csv", low_memory=False)

# Drop rows with NaN values
df_main = df_main.dropna()

# Apply z-scoring to all variables before any computations
for col in df_main.columns:
    df_main[col] = zscore(df_main[col])

# 'RDS' calculation and demeaning
df_main['RDS'] = df_main[['2050-2.0', '2060-2.0', '2070-2.0', '2080-2.0']].sum(axis=1)
df_main['RDS'] = df_main['RDS'] - df_main['RDS'].mean()

# Corrected DataFrame reference for start_index and end_index
start_index = df_main.columns.get_loc('27143-2.0')
end_index = df_main.columns.get_loc('27327-2.0')

# Adjusted function with NaN/Inf checks (just hashtag head motion variable to remove)
def perform_analysis_and_save(df_main, file_name):
    results_list = []
    for dkt_var in df_main.columns[start_index:end_index + 1]:
        df_main['DKT'] = df_main[dkt_var] - df_main[dkt_var].mean()

        # Check for NaNs or infinite values after calculating 'DKT'
        if df_main['DKT'].isnull().any() or np.isinf(df_main['DKT']).any():
            print(f"NaNs or infinite values detected in DKT calculations for variable {dkt_var}.")
            continue

        covariates = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0']
        all_vars = ['DKT'] + covariates
        X = df_main[all_vars]
        X = sm.add_constant(X)

        # Additional check for NaNs or Inf values in X
        if X.isnull().any().any() or np.isinf(X).any().any():
            print(f"NaNs or infinite values detected in X for variable {dkt_var}.")
            continue

        Y = df_main['RDS']
        model = sm.OLS(Y, X).fit()
        results_list.append({'Variable': dkt_var, 'Coefficient': model.params['DKT'], 'p-value': model.pvalues['DKT']})

    results_df = pd.DataFrame(results_list)
    results_df.to_csv(file_name, index=False)
    return results_df

# Adjusted paths for saving the results
#new_file_path_whole_group2 = '/Users/Cabria/PycharmProjects/pythonProject/RDS_results_whole_group_singular_HM2.csv'

# Adjusted paths for saving the results
new_file_path_whole_group3 = '/Users/Cabria/PycharmProjects/pythonProject/RDS_results_whole_group_singular_3.csv'

# Perform the analysis for whole group, then save the results
#results = perform_analysis_and_save(df_main, new_file_path_whole_group2)
#print(f"Whole group linear regression + 25741-2.0 {new_file_path_whole_group2}.")

# Perform the analysis for whole group w/o head motion, then save the results
results = perform_analysis_and_save(df_main, new_file_path_whole_group3)
print(f"Whole group linear regression + no hm {new_file_path_whole_group3}.")


#Retrying linear regression on whole group + RDS + zscore to check for correctness
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import zscore

# Load the main dataset
df_main = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset3.csv", low_memory=False)
df_main = df_main.dropna()  # Drop rows with NaN values

# Calculate the number of unique participant IDs in the cleaned dataset
sample_size = df_main['eid'].nunique()
print(f"Sample size for linear regression: {sample_size}")

# Apply z-scoring
for col in df_main.columns:
    df_main[col] = zscore(df_main[col])

df_main['RDS'] = df_main[['2050-2.0', '2060-2.0', '2070-2.0', '2080-2.0']].sum(axis=1)
df_main['RDS'] -= df_main['RDS'].mean()

def perform_regression(df, include_head_motion):
    start_index = df.columns.get_loc('27143-2.0')
    end_index = df.columns.get_loc('27327-2.0')
    results_list = []

    for dkt_var in df.columns[start_index:end_index + 1]:
        df['DKT'] = df[dkt_var] - df[dkt_var].mean()
        if df['DKT'].isnull().any() or np.isinf(df['DKT']).any():
            continue

        covariates = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0']
        if include_head_motion:
            covariates.append('25741-2.0')  # Adding head motion control

        all_vars = ['DKT'] + covariates
        X = df[all_vars]
        X = sm.add_constant(X)
        if X.isnull().any().any() or np.isinf(X).any().any():
            continue

        Y = df['RDS']
        model = sm.OLS(Y, X).fit()
        results_list.append({'Variable': dkt_var, 'Coefficient': model.params['DKT'], 'p-value': model.pvalues['DKT']})

    return pd.DataFrame(results_list)

# Run without and with head motion control
results_no_hm = perform_regression(df_main, False)
results_with_hm = perform_regression(df_main, True)

# Save the results
results_no_hm.to_csv("/Users/Cabria/PycharmProjects/pythonProject/RDS_results_whole_group_no_hm7.csv", index=False)
results_with_hm.to_csv("/Users/Cabria/PycharmProjects/pythonProject/RDS_results_whole_group_with_hm7.csv", index=False)
print("Analysis completed for both scenarios.")





#False Discovery Rate correction on whole group + subject lists A & B (with head motion)
import pandas as pd
import numpy as np
import statsmodels.stats.multitest as smm

# Load the datasets
data_whole_group = pd.read_csv('/Users/Cabria/PycharmProjects/pythonProject/RDS_results_whole_group_singular_HM.csv')
data_subject_A = pd.read_csv('/Users/Cabria/PycharmProjects/pythonProject/RDS_results_split_A.csv')
data_subject_B = pd.read_csv('/Users/Cabria/PycharmProjects/pythonProject/RDS_results_split_B.csv')

alpha = 0.05  # Significance level

# Define function to apply FDR correction using Benjamini-Hochberg procedure
def apply_fdr_correction(data, p_value_col='p-value', alpha=0.05):
    p_values = data[p_value_col].values
    rejected, corrected_p_values = smm.multipletests(p_values, alpha=alpha, method='fdr_bh')[:2]
    data['Significant After FDR'] = rejected
    data['Corrected p-value'] = corrected_p_values
    return data

# Apply the FDR correction function to the datasets
data_whole_group = apply_fdr_correction(data_whole_group, 'p-value', alpha)
data_subject_A = apply_fdr_correction(data_subject_A, 'p-value', alpha)
data_subject_B = apply_fdr_correction(data_subject_B, 'p-value', alpha)

# Define new file paths for saving the results after FDR correction
new_file_path_whole_group_fdr = '/Users/Cabria/PycharmProjects/pythonProject/RDS_results_whole_group_FDR.csv'
new_file_path_subject_A_fdr = '/Users/Cabria/PycharmProjects/pythonProject/RDS_results_split_A_FDR.csv'
new_file_path_subject_B_fdr = '/Users/Cabria/PycharmProjects/pythonProject/RDS_results_split_B_FDR.csv'

# Save out the datasets to new CSV files
data_whole_group.to_csv(new_file_path_whole_group_fdr, index=False)
data_subject_A.to_csv(new_file_path_subject_A_fdr, index=False)
data_subject_B.to_csv(new_file_path_subject_B_fdr, index=False)

print("FDR correction applied and results saved for the whole group, subject list A, and subject list B.")





#False Discovery Rate correction on whole group + subject lists A & B (WITHOUT head motion)
import pandas as pd
import numpy as np
import statsmodels.stats.multitest as smm

# Load the datasets
data_whole_group_no_motion = pd.read_csv('/Users/Cabria/PycharmProjects/pythonProject/RDS_results_whole_group_no_motion.csv')
data_subject_A_no_motion = pd.read_csv('/Users/Cabria/PycharmProjects/pythonProject/RDS_results_split_A_no_motion.csv')
data_subject_B_no_motion = pd.read_csv('/Users/Cabria/PycharmProjects/pythonProject/RDS_results_split_B_no_motion.csv')

alpha = 0.05  # Significance level

# Define function to apply FDR correction using Benjamini-Hochberg procedure
def apply_fdr_correction(data, p_value_col='p-value', alpha=0.05):
    p_values = data[p_value_col].values
    rejected, corrected_p_values = smm.multipletests(p_values, alpha=alpha, method='fdr_bh')[:2]
    data['Significant After FDR'] = rejected
    data['Corrected p-value'] = corrected_p_values
    return data

# Apply the FDR correction function to the datasets
data_whole_group_no_motion = apply_fdr_correction(data_whole_group_no_motion, 'p-value', alpha)
data_subject_A_no_motion = apply_fdr_correction(data_subject_A_no_motion, 'p-value', alpha)
data_subject_B_no_motion = apply_fdr_correction(data_subject_B_no_motion, 'p-value', alpha)

# Define new file paths for saving the results after FDR correction
new_file_path_whole_group_fdr_no_motion = '/Users/Cabria/PycharmProjects/pythonProject/RDS_results_whole_group_FDR_no_motion.csv'
new_file_path_subject_A_fdr_no_motion = '/Users/Cabria/PycharmProjects/pythonProject/RDS_results_split_A_FDR_no_motion.csv'
new_file_path_subject_B_fdr_no_motion = '/Users/Cabria/PycharmProjects/pythonProject/RDS_results_split_B_FDR_no_motion.csv'

# Save out the datasets to new CSV files
data_whole_group_no_motion.to_csv(new_file_path_whole_group_fdr_no_motion, index=False)
data_subject_A_no_motion.to_csv(new_file_path_subject_A_fdr_no_motion, index=False)
data_subject_B_no_motion.to_csv(new_file_path_subject_B_fdr_no_motion, index=False)

print("FDR correction applied and results saved for the whole group with no motion, subject list A, and subject list B.")




#Applying FDR correction to zscored data whole group with HM and no HM
import pandas as pd
import numpy as np
import statsmodels.stats.multitest as smm

# Load the datasets (datasets were reversed which is why output was incorrect)
data_whole_group_with_motion = pd.read_csv('/Users/Cabria/PycharmProjects/pythonProject/RDS_results_whole_group_singular_3.csv')
data_whole_group_no_motion = pd.read_csv('/Users/Cabria/PycharmProjects/pythonProject/RDS_results_whole_group_singular_HM2.csv')

alpha = 0.05  # Significance level

# Define function to apply FDR correction using Benjamini-Hochberg procedure
def apply_fdr_correction(data, p_value_col='p-value', alpha=0.05):
    p_values = data[p_value_col].values
    rejected, corrected_p_values = smm.multipletests(p_values, alpha=alpha, method='fdr_bh')[:2]
    data['Significant After FDR'] = rejected
    data['Corrected p-value'] = corrected_p_values
    return data

# Apply the FDR correction function to the datasets
data_whole_group_no_motion = apply_fdr_correction(data_whole_group_no_motion)
data_whole_group_with_motion = apply_fdr_correction(data_whole_group_with_motion)

# Define new file paths for saving the results after FDR correction
new_file_path_whole_group_fdr_no_motion = '/Users/Cabria/PycharmProjects/pythonProject/RDS_results_whole_group_FDR_no_motion.csv'
new_file_path_whole_group_fdr_with_motion = '/Users/Cabria/PycharmProjects/pythonProject/RDS_results_whole_group_FDR_with_motion.csv'

# Save out the datasets to new CSV files
data_whole_group_no_motion.to_csv(new_file_path_whole_group_fdr_no_motion, index=False)
data_whole_group_with_motion.to_csv(new_file_path_whole_group_fdr_with_motion, index=False)

print("FDR correction applied and results saved for the whole group with and without motion.")




#Reapplying FDR correction to zscored RRDS-4 data whole group with HM and no HM to check for correctness
import pandas as pd
import numpy as np
import statsmodels.stats.multitest as smm

# Load the datasets
data_whole_group_with_motion = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/RDS_results_whole_group_with_hm6.csv")
data_whole_group_no_motion = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/RDS_results_whole_group_no_hm6.csv")

alpha = 0.05  # Significance level

# Define function to apply FDR correction using Benjamini-Hochberg procedure
def apply_fdr_correction(data, p_value_col='p-value', alpha=0.05):
    p_values = data[p_value_col].values
    rejected, corrected_p_values = smm.multipletests(p_values, alpha=alpha, method='fdr_bh')[:2]
    data['Significant After FDR'] = rejected
    data['Corrected p-value'] = corrected_p_values
    return data

# Apply the FDR correction function to the datasets
data_whole_group_no_motion = apply_fdr_correction(data_whole_group_no_motion)
data_whole_group_with_motion = apply_fdr_correction(data_whole_group_with_motion)

# Define new file paths for saving the results after FDR correction
new_file_path_whole_group_fdr_no_motion6 = '/Users/Cabria/PycharmProjects/pythonProject/RDS_results_whole_group_FDR_no_motion6.csv'
new_file_path_whole_group_fdr_with_motion6 = '/Users/Cabria/PycharmProjects/pythonProject/RDS_results_whole_group_FDR_with_motion6.csv'

# Save out the datasets to new CSV files
data_whole_group_no_motion.to_csv(new_file_path_whole_group_fdr_no_motion6, index=False)
data_whole_group_with_motion.to_csv(new_file_path_whole_group_fdr_with_motion6, index=False)

print("FDR correction applied and results saved for the whole group with and without motion.")







#Substracting RDS coefficents to get the mean
import pandas as pd

# Load the data
df_no_hm = pd.read_csv('/Users/Cabria/PycharmProjects/pythonProject/RDS_results_whole_group_FDR_no_motion6.csv')
df_with_hm = pd.read_csv('/Users/Cabria/PycharmProjects/pythonProject/RDS_results_whole_group_FDR_with_motion6.csv')

# Merge dataframes on the 'Variable' column for comparison
merged_df = pd.merge(df_no_hm, df_with_hm, on='Variable', suffixes=('_no_hm', '_with_hm'))

# Filter for DKT variables where significance changes from True to False
significant_change_df = merged_df[
    (merged_df['Significant After FDR_no_hm'] == True) &
    (merged_df['Significant After FDR_with_hm'] == False)
]

# Calculate the difference in coefficients for these variables
significant_change_df['Coefficient_Difference'] = significant_change_df['Coefficient_with_hm'] - significant_change_df['Coefficient_no_hm']

# Calculate the mean of these differences
mean_difference = significant_change_df['Coefficient_Difference'].mean()

# Output the results
print("Variables with changed significance:")
print(significant_change_df[['Variable', 'Coefficient_no_hm', 'Coefficient_with_hm', 'Coefficient_Difference']])
print(f"Mean coefficient difference: {mean_difference}")



#Linear regression for N-12, mean rfMRI, covariates all zscored
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import zscore

# Load the main dataset
df_main = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset3.csv", low_memory=False)
df_main = df_main.dropna()  # Drop rows with NaN values

# Calculate the number of unique participant IDs in the cleaned dataset
sample_size = df_main['eid'].nunique()
print(f"Sample size for linear regression: {sample_size}")

# Apply z-scoring
for col in df_main.columns:
    df_main[col] = zscore(df_main[col])

# N-12 calculation
n_12_columns = ['1920-2.0', '1930-2.0', '1940-2.0', '1950-2.0', '1960-2.0',
                '1970-2.0', '1980-2.0', '1990-2.0', '2000-2.0', '2010-2.0',
                '2020-2.0', '2030-2.0']
df_main['N-12'] = df_main[n_12_columns].sum(axis=1)
df_main['N-12'] -= df_main['N-12'].mean()

def perform_regression(df, include_head_motion):
    start_index = df.columns.get_loc('27143-2.0')
    end_index = df.columns.get_loc('27327-2.0')
    results_list = []

    for dkt_var in df.columns[start_index:end_index + 1]:
        df['DKT'] = df[dkt_var] - df[dkt_var].mean()
        if df['DKT'].isnull().any() or np.isinf(df['DKT']).any():
            continue

        covariates = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0']
        if include_head_motion:
            covariates.append('25741-2.0')  # Adding head motion control

        all_vars = ['DKT'] + covariates
        X = df[all_vars]
        X = sm.add_constant(X)
        if X.isnull().any().any() or np.isinf(X).any().any():
            continue

        Y = df['N-12']
        model = sm.OLS(Y, X).fit()
        results_list.append({'Variable': dkt_var, 'Coefficient': model.params['DKT'], 'p-value': model.pvalues['DKT']})

    return pd.DataFrame(results_list)

# Run without and with head motion control
results_no_hm = perform_regression(df_main, False)
results_with_hm = perform_regression(df_main, True)

# Save the results
results_no_hm.to_csv("/Users/Cabria/PycharmProjects/pythonProject/N-12_results_whole_group_no_hm_new.csv", index=False)
results_with_hm.to_csv("/Users/Cabria/PycharmProjects/pythonProject/N-12_results_whole_group_with_hm_new.csv", index=False)
print("Analysis completed for both scenarios.")






# FDR Correction for N-12: Whole Group with/without rfMRI HM
import pandas as pd
import numpy as np
import statsmodels.stats.multitest as smm

# Load the datasets
data_whole_group_with_motion = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/N-12_results_whole_group_with_hm.csv", low_memory=False)
data_whole_group_no_motion = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/N-12_results_whole_group_no_hm.csv", low_memory=False)

alpha = 0.05  # Significance level

# Define function to apply FDR correction using Benjamini-Hochberg procedure
def apply_fdr_correction(data, p_value_col='p-value', alpha=0.05):
    p_values = data[p_value_col].values
    rejected, corrected_p_values = smm.multipletests(p_values, alpha=alpha, method='fdr_bh')[:2]
    data['Significant After FDR'] = rejected
    data['Corrected p-value'] = corrected_p_values
    return data.copy()  # Return a defragmented copy of the DataFrame

# Apply the FDR correction function to the datasets
data_whole_group_no_motion = apply_fdr_correction(data_whole_group_no_motion)
data_whole_group_with_motion = apply_fdr_correction(data_whole_group_with_motion)

# Define new file paths for saving the results after FDR correction
new_file_path_whole_group_fdr_no_motion6 = '/Users/Cabria/PycharmProjects/pythonProject/N-12_results_whole_group_FDR_no_motion2.csv'
new_file_path_whole_group_fdr_with_motion6 = '/Users/Cabria/PycharmProjects/pythonProject/N-12_results_whole_group_FDR_with_motion2.csv'

# Save out the datasets to new CSV files
data_whole_group_no_motion.to_csv(new_file_path_whole_group_fdr_no_motion6, index=False)
data_whole_group_with_motion.to_csv(new_file_path_whole_group_fdr_with_motion6, index=False)

print("FDR correction applied and results saved for the whole group with and without motion.")




# Linear Regression for IQ, Mean rfMRI, Covariates (Z-Scored)
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import zscore

# Load the main dataset
df_main = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset3.csv", low_memory=False)
df_main = df_main.dropna()  # Drop rows with NaN values

# Calculate the number of unique participant IDs in the cleaned dataset
sample_size = df_main['eid'].nunique()
print(f"Sample size for linear regression: {sample_size}")

# Apply z-scoring
for col in df_main.columns:
    df_main[col] = zscore(df_main[col])

# IQ calculation
df_main['IQ'] = df_main['20016-2.0']

def perform_regression(df, include_head_motion):
    start_index = df.columns.get_loc('27143-2.0')
    end_index = df.columns.get_loc('27327-2.0')
    results_list = []

    for dkt_var in df.columns[start_index:end_index + 1]:
        df['DKT'] = df[dkt_var] - df[dkt_var].mean()
        if df['DKT'].isnull().any() or np.isinf(df['DKT']).any():
            continue

        covariates = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0']
        if include_head_motion:
            covariates.append('25741-2.0')  # Adding head motion control

        all_vars = ['DKT'] + covariates
        X = df[all_vars]
        X = sm.add_constant(X)
        if X.isnull().any().any() or np.isinf(X).any().any():
            continue

        Y = df['IQ']
        model = sm.OLS(Y, X).fit()
        results_list.append({'Variable': dkt_var, 'Coefficient': model.params['DKT'], 'p-value': model.pvalues['DKT']})

    return pd.DataFrame(results_list)

# Run without and with head motion control
results_no_hm = perform_regression(df_main, False)
results_with_hm = perform_regression(df_main, True)

# Save the results
results_no_hm.to_csv("/Users/Cabria/PycharmProjects/pythonProject/IQ_results_whole_group_no_hm.csv", index=False)
results_with_hm.to_csv("/Users/Cabria/PycharmProjects/pythonProject/IQ_results_whole_group_with_hm.csv", index=False)
print("IQ analysis completed for both scenarios.")





# Linear Regression for Reaction Time, Mean rfMRI, Covariates (Z-Scored)
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import zscore

# Load the main dataset
df_main = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset3.csv", low_memory=False)
df_main = df_main.dropna()  # Drop rows with NaN values

# Calculate the number of unique participant IDs in the cleaned dataset
sample_size = df_main['eid'].nunique()
print(f"Sample size for linear regression: {sample_size}")

# Apply z-scoring
for col in df_main.columns:
    df_main[col] = zscore(df_main[col])

# Reaction Time calculation
df_main['Reaction_Time'] = df_main['20023-2.0']

def perform_regression(df, include_head_motion):
    start_index = df.columns.get_loc('27143-2.0')
    end_index = df.columns.get_loc('27327-2.0')
    results_list = []

    for dkt_var in df.columns[start_index:end_index + 1]:
        df['DKT'] = df[dkt_var] - df[dkt_var].mean()
        if df['DKT'].isnull().any() or np.isinf(df['DKT']).any():
            continue

        covariates = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0']
        if include_head_motion:
            covariates.append('25741-2.0')  # Adding head motion control

        all_vars = ['DKT'] + covariates
        X = df[all_vars]
        X = sm.add_constant(X)
        if X.isnull().any().any() or np.isinf(X).any().any():
            continue

        Y = df['Reaction_Time']
        model = sm.OLS(Y, X).fit()
        results_list.append({'Variable': dkt_var, 'Coefficient': model.params['DKT'], 'p-value': model.pvalues['DKT']})

    return pd.DataFrame(results_list)

# Run without and with head motion control
results_no_hm = perform_regression(df_main, False)
results_with_hm = perform_regression(df_main, True)

# Save the results
results_no_hm.to_csv("/Users/Cabria/PycharmProjects/pythonProject/Reaction_Time_results_whole_group_no_hm.csv", index=False)
results_with_hm.to_csv("/Users/Cabria/PycharmProjects/pythonProject/Reaction_Time_results_whole_group_with_hm.csv", index=False)
print("Reaction Time analysis completed for both scenarios.")




#FDR Correction for IQ and Reaction Time on mean rfMRI zscored data
import pandas as pd
import numpy as np
import statsmodels.stats.multitest as smm

# Load the datasets for IQ
data_iq_with_motion = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/IQ_results_whole_group_with_hm.csv", low_memory=False)
data_iq_no_motion = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/IQ_results_whole_group_no_hm.csv", low_memory=False)

# Load the datasets for Reaction Time
data_rt_with_motion = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/Reaction_Time_results_whole_group_with_hm.csv", low_memory=False)
data_rt_no_motion = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/Reaction_Time_results_whole_group_no_hm.csv", low_memory=False)

alpha = 0.05  # Significance level

# Define function to apply FDR correction using Benjamini-Hochberg procedure
def apply_fdr_correction(data, p_value_col='p-value', alpha=0.05):
    p_values = data[p_value_col].values
    rejected, corrected_p_values = smm.multipletests(p_values, alpha=alpha, method='fdr_bh')[:2]
    data['Significant After FDR'] = rejected
    data['Corrected p-value'] = corrected_p_values
    return data.copy()  # Return a defragmented copy of the DataFrame

# Apply the FDR correction function to the datasets for IQ
data_iq_no_motion = apply_fdr_correction(data_iq_no_motion)
data_iq_with_motion = apply_fdr_correction(data_iq_with_motion)

# Apply the FDR correction function to the datasets for Reaction Time
data_rt_no_motion = apply_fdr_correction(data_rt_no_motion)
data_rt_with_motion = apply_fdr_correction(data_rt_with_motion)

# Define new file paths for saving the results after FDR correction for IQ
new_file_path_iq_fdr_no_motion = '/Users/Cabria/PycharmProjects/pythonProject/IQ_results_FDR_no_motion.csv'
new_file_path_iq_fdr_with_motion = '/Users/Cabria/PycharmProjects/pythonProject/IQ_results_FDR_with_motion.csv'

# Define new file paths for saving the results after FDR correction for Reaction Time
new_file_path_rt_fdr_no_motion = '/Users/Cabria/PycharmProjects/pythonProject/Reaction_Time_results_FDR_no_motion.csv'
new_file_path_rt_fdr_with_motion = '/Users/Cabria/PycharmProjects/pythonProject/Reaction_Time_results_FDR_with_motion.csv'

# Save out the datasets to new CSV files for IQ
data_iq_no_motion.to_csv(new_file_path_iq_fdr_no_motion, index=False)
data_iq_with_motion.to_csv(new_file_path_iq_fdr_with_motion, index=False)

# Save out the datasets to new CSV files for Reaction Time
data_rt_no_motion.to_csv(new_file_path_rt_fdr_no_motion, index=False)
data_rt_with_motion.to_csv(new_file_path_rt_fdr_with_motion, index=False)

print("FDR correction applied and results saved for IQ and Reaction Time, with and without motion.")



#Calculate Average Coefficient Differences for N-12, IQ, and Reaction time
import pandas as pd

def calculate_mean_difference(file_path_no_hm, file_path_with_hm):
    # Load the data
    df_no_hm = pd.read_csv('/Users/Cabria/PycharmProjects/pythonProject/N-12_results_whole_group_FDR_no_motion.csv')
    df_with_hm = pd.read_csv('/Users/Cabria/PycharmProjects/pythonProject/N-12_results_whole_group_FDR_with_motion.csv')

    # Merge dataframes on the 'Variable' column for comparison
    merged_df = pd.merge(df_no_hm, df_with_hm, on='DKT Variable', suffixes=('_no_hm', '_with_hm'))

    # Calculate the difference in coefficients for each variable
    merged_df['Coefficient_Difference'] = merged_df['Coefficient_with_hm'] - merged_df['Coefficient_no_hm']

    # Calculate the mean of these differences
    mean_difference = merged_df['Coefficient_Difference'].mean()

    # Output the results
    print(f"Mean coefficient difference for {file_path_no_hm.split('/')[-1]}: {mean_difference}")

# Files for N-12
calculate_mean_difference(
    '/Users/Cabria/PycharmProjects/pythonProject/N-12_results_whole_group_FDR_no_motion.csv',
    '/Users/Cabria/PycharmProjects/pythonProject/N-12_results_whole_group_FDR_with_motion.csv'
)

# Files for IQ
calculate_mean_difference(
    '/Users/Cabria/PycharmProjects/pythonProject/IQ_results_FDR_no_motion.csv',
    '/Users/Cabria/PycharmProjects/pythonProject/IQ_results_FDR_with_motion.csv'
)

# Files for Reaction Time
calculate_mean_difference(
    '/Users/Cabria/PycharmProjects/pythonProject/Reaction_Time_results_FDR_no_motion.csv',
    '/Users/Cabria/PycharmProjects/pythonProject/Reaction_Time_results_FDR_with_motion.csv'
)


#This one works better
import pandas as pd

def calculate_mean_difference(file_path_no_hm, file_path_with_hm):
    try:
        # Load the data
        df_no_hm = pd.read_csv(file_path_no_hm)
        df_with_hm = pd.read_csv(file_path_with_hm)

        # Print column names to verify
        print("Columns in no HM file:", df_no_hm.columns)
        print("Columns in with HM file:", df_with_hm.columns)

        # Merge dataframes on the 'Variable' column for comparison
        merged_df = pd.merge(df_no_hm, df_with_hm, on='DKT Variable', suffixes=('_no_hm', '_with_hm'))

        # Calculate the difference in coefficients for each variable
        merged_df['Coefficient_Difference'] = merged_df['Coefficient_with_hm'] - merged_df['Coefficient_no_hm']

        # Calculate the mean of these differences
        mean_difference = merged_df['Coefficient_Difference'].mean()

        # Output the results
        print("Coefficient differences for all variables:")
        print(merged_df[['DKT Variable', 'Coefficient_no_hm', 'Coefficient_with_hm', 'Coefficient_Difference']])
        print(f"Mean coefficient difference across all variables: {mean_difference}")

    except Exception as e:
        print(f"An error occurred: {e}")

# usage of the function with file paths
calculate_mean_difference(
    '/Users/Cabria/PycharmProjects/pythonProject/N-12_results_whole_group_FDR_no_motion.csv',
    '/Users/Cabria/PycharmProjects/pythonProject/N-12_results_whole_group_FDR_with_motion.csv'
)





# Get demographics for UKB head motion sample: average age with mean STD, percent male
import pandas as pd

df_main = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset3.csv", low_memory=False)

# Extract relevant columns
age_column = '21003-2.0'
sex_column = '31-0.0'
eid_column = 'eid'

# Drop NaN values in relevant columns
df_main = df_main.dropna()  # Drop rows with NaN values

# Calculate average age and standard deviation
average_age = df_main[age_column].mean()
std_age = df_main[age_column].std()

# Calculate the percentage of males
percent_male = (df_main[sex_column] == 1).mean() * 100

# Calculate the number of unique participant IDs in the cleaned dataset
sample_size = df_main[eid_column].nunique()

# Print the results
print(f"Average Age: {average_age:.2f} years")
print(f"Age Standard Deviation: {std_age:.2f} years")
print(f"Percentage Male: {percent_male:.2f}%")
print(f"Sample Size: {sample_size}")




#Removing outliers code (wapiaw2024)
import os
import pandas as pd
import numpy as np
import warnings
import sys

warnings.filterwarnings('ignore')

indir = sys.argv[1]
outdir=sys.argv[2]
parcel_type_path=sys.argv[3]


def is_binary_or_categorical(column,df):
    return df[column].dtype == 'object' or df[column].nunique() == 2

def exclude_outliers(df, std_threshold=5, outdir=''):
    subject_ids = df.index

    # Mean and STD
    mean = df.mean()
    std = df.std()

    # Outliner detection
    outliers = (df > mean + std_threshold * std) | (df < mean - std_threshold * std)
    outlier_columns = outliers.any(axis=1)

    exclude_subject_ids = outlier_columns.index[outlier_columns == 1].tolist()
    remaining_subject_ids = outlier_columns.index[outlier_columns == 0].tolist()

    pd.DataFrame(remaining_subject_ids).to_csv(f'{outdir}/final_sublist.csv', header=None, index=None)

    print(f"Excluded {len(exclude_subject_ids)} subjects from {len(remaining_subject_ids) + len(exclude_subject_ids)} due to outliers
.")

    return remaining_subject_ids

with open(parcel_type_path, 'r') as file:
    parcel_type = file.read().splitlines()

vals = pd.DataFrame()
# merge for all structural
file_list = ['aseg_yeo.csv','confounds.csv','dkt_yeo.csv']
print(file_list)
for i in file_list:
    valset = pd.read_csv(os.path.join(indir, i), index_col=0, header=0)
    vals = pd.concat([vals, valset], axis=1, ignore_index=False)
# merge for functional
func_file_path = os.path.join(indir, 'func_idp')
func_file_list = [file for file in os.listdir(func_file_path) if file.endswith('.csv')]
for i in func_file_list:
    valset = pd.read_csv(os.path.join(func_file_path, i), index_col=0, header=0)
    vals = pd.concat([vals, valset], axis=1, ignore_index=False)

for parcel in parcel_type:
    file_path = os.path.join(func_file_path, parcel)
    netmat_path = os.path.join(file_path, 'Netmats_flat50_yeo')
    partialNM_path = os.path.join(file_path, 'partial_NMsflat50_yeo')
    subjfnames = [fname for fname in os.listdir(netmat_path) if "sub-" in fname]
    if subjfnames:
        sublist = [f.split('sub-')[1].split('.')[0] for f in subjfnames]
    else:
        # assumes only text files are in directory and all have form '<subjID>.txt'
        subjfnames = os.listdir(netmat_path)
        sublist = [f.split('.')[0] for f in os.listdir(netmat_path)]
    netmats = [np.loadtxt(os.path.join(netmat_path, subjfname)) for subjfname in subjfnames]
    df_netmats = pd.DataFrame({
        'subjfname': sublist,
        'netmat': netmats
    })
    df_netmats.set_index('subjfname', inplace=True)
    partial_netmats = [np.loadtxt(os.path.join(partialNM_path, subjfname)) for subjfname in subjfnames]
    df_partial_netmats = pd.DataFrame({
        'subjfname': sublist,
        'partial_netmat': partial_netmats
    })
    df_partial_netmats.set_index('subjfname', inplace=True)
    df = pd.concat([df_netmats, df_partial_netmats], axis=1, ignore_index=False)

vals = pd.concat([vals, df], axis=1, ignore_index=False)

vals=vals.dropna()
print(vals.shape)

columns_to_keep = [col for col in vals.columns if not is_binary_or_categorical(col,vals)]
vals_numeric = vals[columns_to_keep]
print(vals_numeric.shape)

subjs= exclude_outliers(vals_numeric,std_threshold=9,outdir=outdir)




# ggseg plotting visualization

import matplotlib
matplotlib.use('TkAgg')  # Can also try 'Qt5Agg', 'GTK3Agg', or 'Agg' if 'TkAgg' does not work
import ggseg

data = {
    'bankssts_left': 1.1,
    'caudalanteriorcingulate_left': 1.0,
    'caudalmiddlefrontal_left': 2.6,
    'cuneus_left': 2.6,
    'entorhinal_left': 0.6,
    # Add more ROIs as needed
}
# Plotting
plot = ggseg.plot_dk(
    data=data,
    cmap='Spectral',           # Color map for visualization
    figsize=(15, 15),          # Size of the figure
    background='k',            # Background color of the plot
    edgecolor='w',             # Color of the edges between regions
    bordercolor='gray',        # Color of the border around the plot
    ylabel='Cortical thickness (mm)',  # Label for the y-axis
    title='Cortical ROI Analysis'      # Title of the figure
)



#Plotting RDS delta brain coefficents
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Can also try 'Qt5Agg', 'GTK3Agg', or 'Agg' if 'TkAgg' does not work
import ggseg

# Load the DKT descriptions and delta coefficients CSV files
dkt_descriptions = pd.read_csv("C:/Users/Cabria/Downloads/DKT_descriptions_dictionary.csv")
delta_coefficients = pd.read_csv('C:/Users/Cabria/PycharmProjects/pythonProject//RDS_Delta_Coefficients.csv')
import pandas as pd

# ggseg DKT ROI list
ggseg_rois = [
    'NA_left', 'NA_right', 'bankssts_left', 'bankssts_right',
    'caudalanteriorcingulate_left', 'caudalanteriorcingulate_right',
    'caudalmiddlefrontal_left', 'caudalmiddlefrontal_right',
    'corpuscallosum_left', 'corpuscallosum_right', 'cuneus_left', 'cuneus_right',
    'entorhinal_left', 'entorhinal_right', 'fusiform_left', 'fusiform_right',
    'inferiorparietal_left', 'inferiorparietal_right', 'inferiortemporal_left',
    'inferiortemporal_right', 'insula_left', 'insula_right', 'isthmuscingulate_left',
    'isthmuscingulate_right', 'lateral_left', 'lateral_right', 'lateraloccipital_left',
    'lateraloccipital_right', 'lateralorbitofrontal_left', 'lateralorbitofrontal_right',
    'lingual_left', 'lingual_right', 'medial_left', 'medial_right', 'medialorbitofrontal_left',
    'medialorbitofrontal_right', 'middletemporal_left', 'middletemporal_right',
    'paracentral_left', 'paracentral_right', 'parahippocampal_left', 'parahippocampal_right',
    'parsopercularis_left', 'parsopercularis_right', 'parsorbitalis_left', 'parsorbitalis_right',
    'parstriangularis_left', 'parstriangularis_right', 'pericalcarine_left', 'pericalcarine_right',
    'postcentral_left', 'postcentral_right', 'posteriorcingulate_left', 'posteriorcingulate_right',
    'precentral_left', 'precentral_right', 'precuneus_left', 'precuneus_right',
    'rostralanteriorcingulate_left', 'rostralanteriorcingulate_right', 'rostralmiddlefrontal_left',
    'rostralmiddlefrontal_right', 'superiorfrontal_left', 'superiorfrontal_right',
    'superiorparietal_left', 'superiorparietal_right', 'superiortemporal_left',
    'superiortemporal_right', 'supramarginal_left', 'supramarginal_right',
    'transversetemporal_left', 'transversetemporal_right'
]







# Updated Depression Linear Regression Code with 4 different levels

# 1. Importing packages
import pandas as pd
import numpy as np
from scipy.stats import zscore
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from adjustText import adjust_text
import os
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Helper Functions
def is_binary_or_categorical(column, df):
    return df[column].dtype == 'object' or df[column].nunique() == 2

def remove_outliers(df, columns_for_outlier_detection, std_threshold=6):
    df_outlier_detection = df[columns_for_outlier_detection]
    mean = df_outlier_detection.mean()
    std = df_outlier_detection.std()
    outliers = (df_outlier_detection > mean + std_threshold * std) | (df_outlier_detection < mean - std_threshold * std)
    df_clean = df[~outliers.any(axis=1)]

    num_excluded = df.shape[0] - df_clean.shape[0]
    num_remaining = df_clean.shape[0]

    print(f"Excluded {num_excluded} subjects due to outliers.")
    print(f"Remaining subjects: {num_remaining}")

    return df_clean

# 2. Preparing and Loading RDS-4 Data for Precise Linear Regression
df_main = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset3.csv", low_memory=False)
df_main = df_main.dropna()

df_main['RDS'] = df_main[['2050-2.0', '2060-2.0', '2070-2.0', '2080-2.0']].sum(axis=1)

# Specify columns for outlier detection
columns_for_outlier_detection = [col for col in df_main.columns if not is_binary_or_categorical(col, df_main) and col not in ['RDS']]
df_main = remove_outliers(df_main, columns_for_outlier_detection, std_threshold=6)

# Apply z-score normalization after outlier removal
df_main = df_main.apply(zscore)

# Ensure all required columns are present
required_columns = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0', '25741-2.0', '27143-2.0', '27327-2.0']
missing_columns = [col for col in required_columns if col not in df_main.columns]
if missing_columns:
    raise KeyError(f"The following required columns are missing: {missing_columns}")

# 3. Performing Linear Regression Analysis and Saving Results
def perform_analysis_and_save(df_main, head_motion_columns):
    results_dict = {'T1_head_motion': [], 'rfMRI_head_motion': [], 'all_head_motion': [], 'no_head_motion': []}
    start_index = df_main.columns.get_loc('27143-2.0')
    end_index = df_main.columns.get_loc('27327-2.0')

    for dkt_var in df_main.columns[start_index:end_index + 1]:
        df_main['DKT'] = df_main[dkt_var]

        if df_main['DKT'].isnull().any() or np.isinf(df_main['DKT']).any():
            print(f"NaNs or infinite values detected in DKT calculations for variable {dkt_var}.")
            continue

        for include_head_motion in ['T1', 'rfMRI', 'all', 'none']:
            covariates = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0']
            if include_head_motion == 'T1':
                covariates.append('24419-2.0')
            elif include_head_motion == 'rfMRI':
                covariates.append('25741-2.0')
            elif include_head_motion == 'all':
                covariates.extend(head_motion_columns)

            all_vars = ['DKT'] + covariates
            X = df_main[all_vars]

            if X.isnull().any().any() or np.isinf(X).any().any():
                print(f"NaNs or infinite values detected in X for variable {dkt_var}.")
                continue

            Y = df_main['RDS']
            model = sm.OLS(Y, X).fit()
            result = {
                'Variable': dkt_var,
                'Coefficient': model.params['DKT'],
                'p-value': model.pvalues['DKT']
            }

            key = f"{include_head_motion}_head_motion" if include_head_motion != 'none' else 'no_head_motion'
            results_dict[key].append(result)

    for key in results_dict:
        if results_dict[key]:  # Only save if there are results to save
            results_df = pd.DataFrame(results_dict[key])
            file_name = f"/Users/Cabria/PycharmProjects/pythonProject/RDS_results_{key}.csv"
            results_df.to_csv(file_name, index=False)
            apply_fdr(results_df, key)

    return results_dict

# 4. Applying False Discovery Rate (FDR) Correction to P-values
def apply_fdr(results_df, key):
    p_values = results_df['p-value'].values
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    results_df['p-value_corrected'] = pvals_corrected
    results_df['Significant After FDR'] = reject
    file_name = f"/Users/Cabria/PycharmProjects/pythonProject/RDS_results_{key}_FDR.csv"
    results_df.to_csv(file_name, index=False)
    return results_df

head_motion_columns = [
    '24419-2.0', '24438-2.0', '24439-2.0', '24440-2.0', '24441-2.0',
    '24442-2.0', '24443-2.0', '25741-2.0', '24450-2.0', '24451-2.0',
    '24452-2.0', '24453-2.0', '24454-2.0', '24455-2.0', '24444-2.0',
    '24445-2.0', '24446-2.0', '24447-2.0', '24448-2.0', '24449-2.0',
    '25742-2.0'
]

results_dict = perform_analysis_and_save(df_main, head_motion_columns)
print("Results and FDR-corrected results saved for both models")

# Calculate Delta (Difference) of Coefficients and Save to CSV
def calculate_and_save_delta():
    base_path = "/Users/Cabria/PycharmProjects/pythonProject/"
    file_name_template = "RDS_results_{}_head_motion_FDR.csv"

    delta_list = []

    for include_head_motion in ['T1', 'rfMRI', 'all']:
        file_with_hm = base_path + file_name_template.format(include_head_motion)
        file_no_hm = base_path + "RDS_results_no_head_motion_FDR.csv"

        if not os.path.exists(file_with_hm) or not os.path.exists(file_no_hm):
            print(f"File not found: {file_with_hm} or {file_no_hm}")
            continue

        df_with_hm = pd.read_csv(file_with_hm)
        df_no_hm = pd.read_csv(file_no_hm)

        for variable in df_with_hm['Variable']:
            with_hm_coef = df_with_hm.loc[df_with_hm['Variable'] == variable, 'Coefficient'].values[0]
            without_hm_coef = df_no_hm.loc[df_no_hm['Variable'] == variable, 'Coefficient'].values[0] if variable in df_no_hm['Variable'].values else np.nan
            delta = abs(with_hm_coef) - abs(without_hm_coef)
            delta_list.append({
                'Variable': variable,
                'Coefficient_with_head_motion': with_hm_coef,
                'Coefficient_without_head_motion': without_hm_coef,
                'Delta_Coefficient': delta,
                'Dataset': include_head_motion + ' motion'
            })

    delta_df = pd.DataFrame(delta_list)
    delta_df.to_csv(base_path + "RDS_Delta_Coefficients.csv", index=False)

calculate_and_save_delta()
print("Delta coefficients saved.")

# 5. Combining Results and Creating Plots
# Load the descriptions dictionary
dkt_descriptions_df = pd.read_csv("C:/Users/Cabria/Downloads/DKT_descriptions_dictionary.csv")
dkt_descriptions_df.columns = [col.strip() for col in dkt_descriptions_df.columns]  # Strip any leading/trailing whitespace
dkt_descriptions_df['DKT Variable'] = dkt_descriptions_df['DKT Variable'].astype(str)
dkt_descriptions_dict = pd.Series(dkt_descriptions_df['DKT Description'].values, index=dkt_descriptions_df['DKT Variable']).to_dict()

# Function to classify DKT variables based on description
def get_dkt_type(dkt_variable):
    dkt_description = dkt_descriptions_dict.get(dkt_variable, '').lower()
    if "thickness" in dkt_description:
        return 'thickness'
    elif "area" in dkt_description:
        return 'area'
    elif "volume" in dkt_description:
        return 'volume'
    return 'other'

# Define colors for each group
model_colors = {
    'Exclusive to Model with Head Motion': 'red',
    'Exclusive to Model without Head Motion': 'blue',
    'Shared Between Both Models': 'green'
}

legend_elements = [
    Line2D([0], [0], color='red', lw=4, label='Exclusive to Model with Head Motion'),
    Line2D([0], [0], color='blue', lw=4, label='Exclusive to Model without Head Motion'),
    Line2D([0], [0], color='green', lw=4, label='Shared Between Both Models')
]

# Plotting Significant Coefficients
def plot_significant_dkt_coefficients(df, dkt_type, title, output_path=None):
    df_filtered = df[df['DKT Type'] == dkt_type]
    significant_with_hm = df_filtered['Significant_with_head_motion'] == True
    significant_without_hm = df_filtered['Significant_without_head_motion'] == True
    significant_indices = df_filtered[significant_with_hm | significant_without_hm].index

    fig, ax = plt.subplots(figsize=(20, 15))

    for idx in significant_indices:
        row = df_filtered.loc[idx]
        with_hm_coef = row['Coefficient_with_head_motion']
        without_hm_coef = row['Coefficient_without_head_motion']
        description = dkt_descriptions_dict.get(row['Variable'], 'No Description')

        if significant_with_hm[idx] and not significant_without_hm[idx]:
            color = model_colors['Exclusive to Model with Head Motion']
        elif not significant_with_hm[idx] and significant_without_hm[idx]:
            color = model_colors['Exclusive to Model without Head Motion']
        else:
            color = model_colors['Shared Between Both Models']

        jitter = np.random.uniform(-0.02, 0.02, 2)

        if pd.notnull(with_hm_coef) and pd.notnull(without_hm_coef):
            ax.plot([0 + jitter[0], 1 + jitter[1]], [with_hm_coef, without_hm_coef], marker='o', markersize=4, linestyle='-', linewidth=1, color=color, alpha=0.7)
            ax.text(1.05, without_hm_coef, description, fontsize=6, verticalalignment='center', color=color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['With Head Motion', 'Without Head Motion'])
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Coefficient")
    ax.axhline(y=0, color='black', linewidth=1, linestyle='-')
    ax.legend(handles=legend_elements, loc='best')

    if output_path:
        plt.savefig(output_path)
    plt.show()

# Visualization
def visualize_delta_coefficients():
    base_path = "/Users/Cabria/PycharmProjects/pythonProject/"
    delta_file_path = base_path + "RDS_Delta_Coefficients.csv"

    if not os.path.exists(delta_file_path):
        print(f"File not found: {delta_file_path}")
        return

    delta_df = pd.read_csv(delta_file_path)

    delta_df['Type'] = delta_df['Variable'].apply(get_dkt_type)

    melted_data = delta_df.melt(id_vars=['Dataset', 'Variable', 'Type'], value_vars=['Delta_Coefficient'], var_name='Metric', value_name='Coefficient')

    plt.figure(figsize=(12, 8))
    sns.swarmplot(x='Type', y='Coefficient', hue='Dataset', data=melted_data, dodge=True)
    plt.title('Depression (RDS-4) ∆ Coefficients Across Models')
    plt.xlabel('Brain Measurement Type')
    plt.ylabel('∆ Coefficient (model with motion control - model without motion control)')
    plt.legend(title='Effect of')
    plt.show()

visualize_delta_coefficients()

def plot_all_dkt_coefficients(df, dkt_type, title, output_path=None):
    df_filtered = df[df['DKT Type'] == dkt_type]

    fig, ax = plt.subplots(figsize=(20, 15))

    for idx, row in df_filtered.iterrows():
        if 'Coefficient_with_head_motion' not in row or 'Coefficient_without_head_motion' not in row:
            raise KeyError(f"'Coefficient_with_head_motion' or 'Coefficient_without_head_motion' column is missing in the DataFrame for row: {row}")

        with_hm_coef = row['Coefficient_with_head_motion']
        without_hm_coef = row['Coefficient_without_head_motion']
        description = dkt_descriptions_dict.get(row['Variable'], 'No Description')

        if pd.notnull(with_hm_coef) and pd.notnull(without_hm_coef):
            if row['Significant_with_head_motion'] and not row['Significant_without_head_motion']:
                color = model_colors['Exclusive to Model with Head Motion']
            elif not row['Significant_with_head_motion'] and row['Significant_without_head_motion']:
                color = model_colors['Exclusive to Model without Head Motion']
            else:
                color = model_colors['Shared Between Both Models']
            jitter = np.random.uniform(-0.02, 0.02, 2)
            ax.plot([0 + jitter[0], 1 + jitter[1]], [with_hm_coef, without_hm_coef], marker='o', markersize=4, linestyle='-', linewidth=1, color=color, alpha=0.7)
            ax.text(1.05, without_hm_coef, description, fontsize=6, verticalalignment='center', color=color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['With Head Motion', 'Without Head Motion'])
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Coefficient")
    ax.axhline(y=0, color='black', linewidth=1, linestyle='-')
    ax.legend(handles=legend_elements, loc='best')

    if output_path:
        plt.savefig(output_path)
    plt.show()

# Create combined dataframe
combined_results = []

for include_head_motion in ['T1', 'rfMRI', 'all']:
    file_with_hm = f"/Users/Cabria/PycharmProjects/pythonProject/RDS_results_{include_head_motion}_head_motion_FDR.csv"
    file_no_hm = "/Users/Cabria/PycharmProjects/pythonProject/RDS_results_no_head_motion_FDR.csv"

    df_with_hm = pd.read_csv(file_with_hm)
    df_no_hm = pd.read_csv(file_no_hm)

    df_with_hm['Significant_with_head_motion'] = df_with_hm['Significant After FDR']
    df_no_hm['Significant_without_head_motion'] = df_no_hm['Significant After FDR']

    df_combined = pd.merge(df_with_hm, df_no_hm, on='Variable', suffixes=('_with_head_motion', '_without_head_motion'))
    df_combined['DKT Type'] = df_combined['Variable'].apply(get_dkt_type)

    combined_results.append(df_combined)

df_combined = pd.concat(combined_results)

# Plotting all coefficients
plot_all_dkt_coefficients(df_combined, 'volume', 'All Depression (RDS-4) Coefficients for Cortical Volume', "/Users/Cabria/PycharmProjects/pythonProject/RDS_all_volume_coefficients_plot.png")
plot_all_dkt_coefficients(df_combined, 'thickness', 'All Depression (RDS-4) Coefficients for Cortical Thickness', "/Users/Cabria/PycharmProjects/pythonProject/RDS_all_thickness_coefficients_plot.png")
plot_all_dkt_coefficients(df_combined, 'area', 'All Depression (RDS-4) Coefficients for Cortical Area', "/Users/Cabria/PycharmProjects/pythonProject/RDS_all_area_coefficients_plot.png")

# Plotting only significant coefficients
# plot_significant_dkt_coefficients(df_combined, 'volume', 'Significant Depression (RDS-4) Coefficients for Cortical Volume', "/Users/Cabria/PycharmProjects/pythonProject/RDS_significant_volume_coefficients_plot.png")
# plot_significant_dkt_coefficients(df_combined, 'thickness', 'Significant Depression (RDS-4) Coefficients for Cortical Thickness', "/Users/Cabria/PycharmProjects/pythonProject/RDS_significant_thickness_coefficients_plot.png")
# plot_significant_dkt_coefficients(df_combined, 'area', 'Significant Depression (RDS-4) Coefficients for Cortical Area', "/Users/Cabria/PycharmProjects/pythonProject/RDS_significant_area_coefficients_plot.png")



# Updated N-12 Linear Regression Code with 4 different levels

# 1. Importing packages
import pandas as pd
import numpy as np
from scipy.stats import zscore
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from adjustText import adjust_text
import os
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Helper Functions
def is_binary_or_categorical(column, df):
    return df[column].dtype == 'object' or df[column].nunique() == 2

def remove_outliers(df, columns_for_outlier_detection, std_threshold=6):
    df_outlier_detection = df[columns_for_outlier_detection]
    mean = df_outlier_detection.mean()
    std = df_outlier_detection.std()
    outliers = (df_outlier_detection > mean + std_threshold * std) | (df_outlier_detection < mean - std_threshold * std)
    df_clean = df[~outliers.any(axis=1)]

    num_excluded = df.shape[0] - df_clean.shape[0]
    num_remaining = df_clean.shape[0]

    print(f"Excluded {num_excluded} subjects due to outliers.")
    print(f"Remaining subjects: {num_remaining}")

    return df_clean

# 2. Preparing and Loading N-12 Data for Precise Linear Regression
df_main = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset3.csv", low_memory=False)
df_main = df_main.dropna()

# N-12 calculation and demeaning
n_12_columns = ['1920-2.0', '1930-2.0', '1940-2.0', '1950-2.0', '1960-2.0',
                '1970-2.0', '1980-2.0', '1990-2.0', '2000-2.0', '2010-2.0',
                '2020-2.0', '2030-2.0']
df_main['N_12'] = df_main[n_12_columns].sum(axis=1)

# Specify columns for outlier detection
columns_for_outlier_detection = [col for col in df_main.columns if not is_binary_or_categorical(col, df_main) and col not in ['N_12']]
df_main = remove_outliers(df_main, columns_for_outlier_detection, std_threshold=6)

# Apply z-score normalization after outlier removal
df_main = df_main.apply(zscore)

# Ensure all required columns are present
required_columns = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0', '25741-2.0', '27143-2.0', '27327-2.0']
missing_columns = [col for col in required_columns if col not in df_main.columns]
if missing_columns:
    raise KeyError(f"The following required columns are missing: {missing_columns}")

# 3. Performing Linear Regression Analysis and Saving Results
def perform_analysis_and_save(df_main, head_motion_columns):
    results_dict = {'T1_head_motion': [], 'rfMRI_head_motion': [], 'all_head_motion': [], 'no_head_motion': []}
    start_index = df_main.columns.get_loc('27143-2.0')
    end_index = df_main.columns.get_loc('27327-2.0')

    for dkt_var in df_main.columns[start_index:end_index + 1]:
        df_main['DKT'] = df_main[dkt_var]

        if df_main['DKT'].isnull().any() or np.isinf(df_main['DKT']).any():
            print(f"NaNs or infinite values detected in DKT calculations for variable {dkt_var}.")
            continue

        for include_head_motion in ['T1', 'rfMRI', 'all', 'none']:
            covariates = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0']
            if include_head_motion == 'T1':
                covariates.append('24419-2.0')
            elif include_head_motion == 'rfMRI':
                covariates.append('25741-2.0')
            elif include_head_motion == 'all':
                covariates.extend(head_motion_columns)

            all_vars = ['DKT'] + covariates
            X = df_main[all_vars]

            if X.isnull().any().any() or np.isinf(X).any().any():
                print(f"NaNs or infinite values detected in X for variable {dkt_var}.")
                continue

            Y = df_main['N_12']
            model = sm.OLS(Y, X).fit()
            result = {
                'Variable': dkt_var,
                'Coefficient': model.params['DKT'],
                'p-value': model.pvalues['DKT']
            }

            key = f"{include_head_motion}_head_motion" if include_head_motion != 'none' else 'no_head_motion'
            results_dict[key].append(result)

    for key in results_dict:
        if results_dict[key]:  # Only save if there are results to save
            results_df = pd.DataFrame(results_dict[key])
            file_name = f"/Users/Cabria/PycharmProjects/pythonProject/N_12_results_{key}.csv"
            results_df.to_csv(file_name, index=False)
            apply_fdr(results_df, key)

    return results_dict

# 4. Applying False Discovery Rate (FDR) Correction to P-values
def apply_fdr(results_df, key):
    p_values = results_df['p-value'].values
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    results_df['p-value_corrected'] = pvals_corrected
    results_df['Significant After FDR'] = reject
    file_name = f"/Users/Cabria/PycharmProjects/pythonProject/N_12_results_{key}_FDR.csv"
    results_df.to_csv(file_name, index=False)
    return results_df

head_motion_columns = [
    '24419-2.0', '24438-2.0', '24439-2.0', '24440-2.0', '24441-2.0',
    '24442-2.0', '24443-2.0', '25741-2.0', '24450-2.0', '24451-2.0',
    '24452-2.0', '24453-2.0', '24454-2.0', '24455-2.0', '24444-2.0',
    '24445-2.0', '24446-2.0', '24447-2.0', '24448-2.0', '24449-2.0',
    '25742-2.0'
]

results_dict = perform_analysis_and_save(df_main, head_motion_columns)
print("Results and FDR-corrected results saved for both models")

# Calculate Delta (Difference) of Coefficients and Save to CSV
def calculate_and_save_delta():
    base_path = "/Users/Cabria/PycharmProjects/pythonProject/"
    file_name_template = "N_12_results_{}_head_motion_FDR.csv"

    delta_list = []

    for include_head_motion in ['T1', 'rfMRI', 'all']:
        file_with_hm = base_path + file_name_template.format(include_head_motion)
        file_no_hm = base_path + "N_12_results_no_head_motion_FDR.csv"

        if not os.path.exists(file_with_hm) or not os.path.exists(file_no_hm):
            print(f"File not found: {file_with_hm} or {file_no_hm}")
            continue

        df_with_hm = pd.read_csv(file_with_hm)
        df_no_hm = pd.read_csv(file_no_hm)

        for variable in df_with_hm['Variable']:
            with_hm_coef = df_with_hm.loc[df_with_hm['Variable'] == variable, 'Coefficient'].values[0]
            without_hm_coef = df_no_hm.loc[df_no_hm['Variable'] == variable, 'Coefficient'].values[0] if variable in df_no_hm['Variable'].values else np.nan
            delta = abs(with_hm_coef) - abs(without_hm_coef)
            delta_list.append({
                'Variable': variable,
                'Coefficient_with_head_motion': with_hm_coef,
                'Coefficient_without_head_motion': without_hm_coef,
                'Delta_Coefficient': delta,
                'Dataset': include_head_motion + ' motion'
            })

    delta_df = pd.DataFrame(delta_list)
    delta_df.to_csv(base_path + "N_12_Delta_Coefficients.csv", index=False)

calculate_and_save_delta()
print("Delta coefficients saved.")

# 5. Combining Results and Creating Plots
# Load the descriptions dictionary
dkt_descriptions_df = pd.read_csv("C:/Users/Cabria/Downloads/DKT_descriptions_dictionary.csv")
dkt_descriptions_df.columns = [col.strip() for col in dkt_descriptions_df.columns]  # Strip any leading/trailing whitespace
dkt_descriptions_df['DKT Variable'] = dkt_descriptions_df['DKT Variable'].astype(str)
dkt_descriptions_dict = pd.Series(dkt_descriptions_df['DKT Description'].values, index=dkt_descriptions_df['DKT Variable']).to_dict()

# Function to classify DKT variables based on description
def get_dkt_type(dkt_variable):
    dkt_description = dkt_descriptions_dict.get(dkt_variable, '').lower()
    if "thickness" in dkt_description:
        return 'thickness'
    elif "area" in dkt_description:
        return 'area'
    elif "volume" in dkt_description:
        return 'volume'
    return 'other'

# Define colors for each group
model_colors = {
    'Exclusive to Model with Head Motion': 'red',
    'Exclusive to Model without Head Motion': 'blue',
    'Shared Between Both Models': 'green'
}

legend_elements = [
    Line2D([0], [0], color='red', lw=4, label='Exclusive to Model with Head Motion'),
    Line2D([0], [0], color='blue', lw=4, label='Exclusive to Model without Head Motion'),
    Line2D([0], [0], color='green', lw=4, label='Shared Between Both Models')
]

# Plotting Significant Coefficients
def plot_significant_dkt_coefficients(df, dkt_type, title, output_path=None):
    df_filtered = df[df['DKT Type'] == dkt_type]
    significant_with_hm = df_filtered['Significant_with_head_motion'] == True
    significant_without_hm = df_filtered['Significant_without_head_motion'] == True
    significant_indices = df_filtered[significant_with_hm | significant_without_hm].index

    fig, ax = plt.subplots(figsize=(20, 15))

    for idx in significant_indices:
        row = df_filtered.loc[idx]
        with_hm_coef = row['Coefficient_with_head_motion']
        without_hm_coef = row['Coefficient_without_head_motion']
        description = dkt_descriptions_dict.get(row['Variable'], 'No Description')

        if significant_with_hm[idx] and not significant_without_hm[idx]:
            color = model_colors['Exclusive to Model with Head Motion']
        elif not significant_with_hm[idx] and significant_without_hm[idx]:
            color = model_colors['Exclusive to Model without Head Motion']
        else:
            color = model_colors['Shared Between Both Models']

        jitter = np.random.uniform(-0.02, 0.02, 2)

        if pd.notnull(with_hm_coef) and pd.notnull(without_hm_coef):
            ax.plot([0 + jitter[0], 1 + jitter[1]], [with_hm_coef, without_hm_coef], marker='o', markersize=4, linestyle='-', linewidth=1, color=color, alpha=0.7)
            ax.text(1.05, without_hm_coef, description, fontsize=6, verticalalignment='center', color=color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['With Head Motion', 'Without Head Motion'])
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Coefficient")
    ax.axhline(y=0, color='black', linewidth=1, linestyle='-')
    ax.legend(handles=legend_elements, loc='best')

    if output_path:
        plt.savefig(output_path)
    plt.show()

# Visualization
def visualize_delta_coefficients():
    base_path = "/Users/Cabria/PycharmProjects/pythonProject/"
    delta_file_path = base_path + "N_12_Delta_Coefficients.csv"

    if not os.path.exists(delta_file_path):
        print(f"File not found: {delta_file_path}")
        return

    delta_df = pd.read_csv(delta_file_path)

    delta_df['Type'] = delta_df['Variable'].apply(get_dkt_type)

    melted_data = delta_df.melt(id_vars=['Dataset', 'Variable', 'Type'], value_vars=['Delta_Coefficient'], var_name='Metric', value_name='Coefficient')

    plt.figure(figsize=(12, 8))
    sns.swarmplot(x='Type', y='Coefficient', hue='Dataset', data=melted_data, dodge=True)
    plt.title('Neuroticism (N-12) ∆ Coefficients Across Models')
    plt.xlabel('Brain Measurement Type')
    plt.ylabel('∆ Coefficient (model with motion control - model without motion control)')
    plt.legend(title='Effect of')
    plt.show()

visualize_delta_coefficients()

def plot_all_dkt_coefficients(df, dkt_type, title, output_path=None):
    df_filtered = df[df['DKT Type'] == dkt_type]

    print(f"Filtered DataFrame for {dkt_type}:")
    print(df_filtered.head())
    print(f"Columns in df_filtered: {df_filtered.columns.tolist()}")

    fig, ax = plt.subplots(figsize=(20, 15))

    for idx, row in df_filtered.iterrows():
        if 'Coefficient_with_head_motion' not in row or 'Coefficient_without_head_motion' not in row:
            raise KeyError(f"'Coefficient_with_head_motion' or 'Coefficient_without_head_motion' column is missing in the DataFrame for row: {row}")

        with_hm_coef = row['Coefficient_with_head_motion']
        without_hm_coef = row['Coefficient_without_head_motion']
        description = dkt_descriptions_dict.get(row['Variable'], 'No Description')

        print(f"Plotting: Variable={row['Variable']}, with_hm_coef={with_hm_coef}, without_hm_coef={without_hm_coef}, description={description}")

        if pd.notnull(with_hm_coef) and pd.notnull(without_hm_coef):
            if row['Significant_with_head_motion'] and not row['Significant_without_head_motion']:
                color = model_colors['Exclusive to Model with Head Motion']
            elif not row['Significant_with_head_motion'] and row['Significant_without_head_motion']:
                color = model_colors['Exclusive to Model without Head Motion']
            else:
                color = model_colors['Shared Between Both Models']
            jitter = np.random.uniform(-0.02, 0.02, 2)
            ax.plot([0 + jitter[0], 1 + jitter[1]], [with_hm_coef, without_hm_coef], marker='o', markersize=4, linestyle='-', linewidth=1, color=color, alpha=0.7)
            ax.text(1.05, without_hm_coef, description, fontsize=6, verticalalignment='center', color=color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['With Head Motion', 'Without Head Motion'])
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Coefficient")
    ax.axhline(y=0, color='black', linewidth=1, linestyle='-')
    ax.legend(handles=legend_elements, loc='best')

    if output_path:
        plt.savefig(output_path)
    plt.show()

# Create combined dataframe
combined_results = []

for include_head_motion in ['T1', 'rfMRI', 'all']:
    file_with_hm = f"/Users/Cabria/PycharmProjects/pythonProject/N_12_results_{include_head_motion}_head_motion_FDR.csv"
    file_no_hm = "/Users/Cabria/PycharmProjects/pythonProject/N_12_results_no_head_motion_FDR.csv"

    df_with_hm = pd.read_csv(file_with_hm)
    df_no_hm = pd.read_csv(file_no_hm)

    df_with_hm['Significant_with_head_motion'] = df_with_hm['Significant After FDR']
    df_no_hm['Significant_without_head_motion'] = df_no_hm['Significant After FDR']

    df_combined = pd.merge(df_with_hm, df_no_hm, on='Variable', suffixes=('_with_head_motion', '_without_head_motion'))
    df_combined['DKT Type'] = df_combined['Variable'].apply(get_dkt_type)

    combined_results.append(df_combined)

df_combined = pd.concat(combined_results)

# Plotting all coefficients
plot_all_dkt_coefficients(df_combined, 'volume', 'All Neuroticism (N-12) Coefficients for Cortical Volume', "/Users/Cabria/PycharmProjects/pythonProject/N_12_all_volume_coefficients_plot.png")
plot_all_dkt_coefficients(df_combined, 'thickness', 'All Neuroticism (N-12) Coefficients for Cortical Thickness', "/Users/Cabria/PycharmProjects/pythonProject/N_12_all_thickness_coefficients_plot.png")
plot_all_dkt_coefficients(df_combined, 'area', 'All Neuroticism (N-12) Coefficients for Cortical Area', "/Users/Cabria/PycharmProjects/pythonProject/N_12_all_area_coefficients_plot.png")

# Plotting only significant coefficients
#plot_significant_dkt_coefficients(df_combined, 'volume', 'Significant Neuroticism (N-12) Coefficients for Cortical Volume', "/Users/Cabria/PycharmProjects/pythonProject/N_12_significant_volume_coefficients_plot.png")
#plot_significant_dkt_coefficients(df_combined, 'thickness', 'Significant Neuroticism (N-12) Coefficients for Cortical Thickness', "/Users/Cabria/PycharmProjects/pythonProject/N_12_significant_thickness_coefficients_plot.png")
#plot_significant_dkt_coefficients(df_combined, 'area', 'Significant Neuroticism (N-12) Coefficients for Cortical Area', "/Users/Cabria/PycharmProjects/pythonProject/N_12_significant_area_coefficients_plot.png")





# Updated Reaction Time Linear Regression Code with 4 different levels

# 1. Importing packages
import pandas as pd
import numpy as np
from scipy.stats import zscore
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from adjustText import adjust_text
import os
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Helper Functions
def is_binary_or_categorical(column, df):
    return df[column].dtype == 'object' or df[column].nunique() == 2

def remove_outliers(df, columns_for_outlier_detection, std_threshold=6):
    df_outlier_detection = df[columns_for_outlier_detection]
    mean = df_outlier_detection.mean()
    std = df_outlier_detection.std()
    outliers = (df_outlier_detection > mean + std_threshold * std) | (df_outlier_detection < mean - std_threshold * std)
    df_clean = df[~outliers.any(axis=1)]

    num_excluded = df.shape[0] - df_clean.shape[0]
    num_remaining = df_clean.shape[0]

    print(f"Excluded {num_excluded} subjects due to outliers.")
    print(f"Remaining subjects: {num_remaining}")

    return df_clean

# 2. Preparing and Loading Reaction Time Data for Precise Linear Regression
df_main = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset3.csv", low_memory=False)
df_main = df_main.dropna()

# Reaction Time calculation
df_main['Reaction_Time'] = df_main['20023-2.0']

# Specify columns for outlier detection
columns_for_outlier_detection = [col for col in df_main.columns if not is_binary_or_categorical(col, df_main) and col not in ['Reaction_Time']]
df_main = remove_outliers(df_main, columns_for_outlier_detection, std_threshold=6)

# Apply z-score normalization after outlier removal
df_main = df_main.apply(zscore)

# Ensure all required columns are present
required_columns = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0', '25741-2.0', '27143-2.0', '27327-2.0']
missing_columns = [col for col in required_columns if col not in df_main.columns]
if missing_columns:
    raise KeyError(f"The following required columns are missing: {missing_columns}")

# 3. Performing Linear Regression Analysis and Saving Results
def perform_analysis_and_save(df_main, head_motion_columns):
    results_dict = {'T1_head_motion': [], 'rfMRI_head_motion': [], 'all_head_motion': [], 'no_head_motion': []}
    start_index = df_main.columns.get_loc('27143-2.0')
    end_index = df_main.columns.get_loc('27327-2.0')

    for dkt_var in df_main.columns[start_index:end_index + 1]:
        df_main['DKT'] = df_main[dkt_var]

        if df_main['DKT'].isnull().any() or np.isinf(df_main['DKT']).any():
            print(f"NaNs or infinite values detected in DKT calculations for variable {dkt_var}.")
            continue

        for include_head_motion in ['T1', 'rfMRI', 'all', 'none']:
            covariates = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0']
            if include_head_motion == 'T1':
                covariates.append('24419-2.0')
            elif include_head_motion == 'rfMRI':
                covariates.append('25741-2.0')
            elif include_head_motion == 'all':
                covariates.extend(head_motion_columns)

            all_vars = ['DKT'] + covariates
            X = df_main[all_vars]

            if X.isnull().any().any() or np.isinf(X).any().any():
                print(f"NaNs or infinite values detected in X for variable {dkt_var}.")
                continue

            Y = df_main['Reaction_Time']
            model = sm.OLS(Y, X).fit()
            result = {
                'Variable': dkt_var,
                'Coefficient': model.params['DKT'],
                'p-value': model.pvalues['DKT']
            }

            key = f"{include_head_motion}_head_motion" if include_head_motion != 'none' else 'no_head_motion'
            results_dict[key].append(result)

    for key in results_dict:
        if results_dict[key]:  # Only save if there are results to save
            results_df = pd.DataFrame(results_dict[key])
            file_name = f"/Users/Cabria/PycharmProjects/pythonProject/Reaction_Time_results_{key}.csv"
            results_df.to_csv(file_name, index=False)
            apply_fdr(results_df, key)

    return results_dict

# 4. Applying False Discovery Rate (FDR) Correction to P-values
def apply_fdr(results_df, key):
    p_values = results_df['p-value'].values
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    results_df['p-value_corrected'] = pvals_corrected
    results_df['Significant After FDR'] = reject
    file_name = f"/Users/Cabria/PycharmProjects/pythonProject/Reaction_Time_results_{key}_FDR.csv"
    results_df.to_csv(file_name, index=False)
    return results_df

head_motion_columns = [
    '24419-2.0', '24438-2.0', '24439-2.0', '24440-2.0', '24441-2.0',
    '24442-2.0', '24443-2.0', '25741-2.0', '24450-2.0', '24451-2.0',
    '24452-2.0', '24453-2.0', '24454-2.0', '24455-2.0', '24444-2.0',
    '24445-2.0', '24446-2.0', '24447-2.0', '24448-2.0', '24449-2.0',
    '25742-2.0'
]

results_dict = perform_analysis_and_save(df_main, head_motion_columns)
print("Results and FDR-corrected results saved for both models")

# Calculate Delta (Difference) of Coefficients and Save to CSV
def calculate_and_save_delta():
    base_path = "/Users/Cabria/PycharmProjects/pythonProject/"
    file_name_template = "Reaction_Time_results_{}_head_motion_FDR.csv"

    delta_list = []

    for include_head_motion in ['T1', 'rfMRI', 'all']:
        file_with_hm = base_path + file_name_template.format(include_head_motion)
        file_no_hm = base_path + "Reaction_Time_results_no_head_motion_FDR.csv"

        if not os.path.exists(file_with_hm) or not os.path.exists(file_no_hm):
            print(f"File not found: {file_with_hm} or {file_no_hm}")
            continue

        df_with_hm = pd.read_csv(file_with_hm)
        df_no_hm = pd.read_csv(file_no_hm)

        for variable in df_with_hm['Variable']:
            with_hm_coef = df_with_hm.loc[df_with_hm['Variable'] == variable, 'Coefficient'].values[0]
            without_hm_coef = df_no_hm.loc[df_no_hm['Variable'] == variable, 'Coefficient'].values[0] if variable in df_no_hm['Variable'].values else np.nan
            delta = abs(with_hm_coef) - abs(without_hm_coef)
            delta_list.append({
                'Variable': variable,
                'Coefficient_with_head_motion': with_hm_coef,
                'Coefficient_without_head_motion': without_hm_coef,
                'Delta_Coefficient': delta,
                'Dataset': include_head_motion + ' motion'
            })

    delta_df = pd.DataFrame(delta_list)
    delta_df.to_csv(base_path + "Reaction_Time_Delta_Coefficients.csv", index=False)

calculate_and_save_delta()
print("Delta coefficients saved.")

# 5. Combining Results and Creating Plots
# Load the descriptions dictionary
dkt_descriptions_df = pd.read_csv("C:/Users/Cabria/Downloads/DKT_descriptions_dictionary.csv")
dkt_descriptions_df.columns = [col.strip() for col in dkt_descriptions_df.columns]  # Strip any leading/trailing whitespace
dkt_descriptions_df['DKT Variable'] = dkt_descriptions_df['DKT Variable'].astype(str)
dkt_descriptions_dict = pd.Series(dkt_descriptions_df['DKT Description'].values, index=dkt_descriptions_df['DKT Variable']).to_dict()

# Function to classify DKT variables based on description
def get_dkt_type(dkt_variable):
    dkt_description = dkt_descriptions_dict.get(dkt_variable, '').lower()
    if "thickness" in dkt_description:
        return 'thickness'
    elif "area" in dkt_description:
        return 'area'
    elif "volume" in dkt_description:
        return 'volume'
    return 'other'

# Define colors for each group
model_colors = {
    'Exclusive to Model with Head Motion': 'red',
    'Exclusive to Model without Head Motion': 'blue',
    'Shared Between Both Models': 'green'
}

legend_elements = [
    Line2D([0], [0], color='red', lw=4, label='Exclusive to Model with Head Motion'),
    Line2D([0], [0], color='blue', lw=4, label='Exclusive to Model without Head Motion'),
    Line2D([0], [0], color='green', lw=4, label='Shared Between Both Models')
]

# Plotting Significant Coefficients
def plot_significant_dkt_coefficients(df, dkt_type, title, output_path=None):
    df_filtered = df[df['DKT Type'] == dkt_type]
    significant_with_hm = df_filtered['Significant_with_head_motion'] == True
    significant_without_hm = df_filtered['Significant_without_head_motion'] == True
    significant_indices = df_filtered[significant_with_hm | significant_without_hm].index

    fig, ax = plt.subplots(figsize=(20, 15))

    for idx in significant_indices:
        row = df_filtered.loc[idx]
        with_hm_coef = row['Coefficient_with_head_motion']
        without_hm_coef = row['Coefficient_without_head_motion']
        description = dkt_descriptions_dict.get(row['Variable'], 'No Description')

        if significant_with_hm[idx] and not significant_without_hm[idx]:
            color = model_colors['Exclusive to Model with Head Motion']
        elif not significant_with_hm[idx] and significant_without_hm[idx]:
            color = model_colors['Exclusive to Model without Head Motion']
        else:
            color = model_colors['Shared Between Both Models']

        jitter = np.random.uniform(-0.02, 0.02, 2)

        if pd.notnull(with_hm_coef) and pd.notnull(without_hm_coef):
            ax.plot([0 + jitter[0], 1 + jitter[1]], [with_hm_coef, without_hm_coef], marker='o', markersize=4, linestyle='-', linewidth=1, color=color, alpha=0.7)
            ax.text(1.05, without_hm_coef, description, fontsize=6, verticalalignment='center', color=color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['With Head Motion', 'Without Head Motion'])
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Coefficient")
    ax.axhline(y=0, color='black', linewidth=1, linestyle='-')
    ax.legend(handles=legend_elements, loc='best')

    if output_path:
        plt.savefig(output_path)
    plt.show()

# Visualization
def visualize_delta_coefficients():
    base_path = "/Users/Cabria/PycharmProjects/pythonProject/"
    delta_file_path = base_path + "Reaction_Time_Delta_Coefficients.csv"

    if not os.path.exists(delta_file_path):
        print(f"File not found: {delta_file_path}")
        return

    delta_df = pd.read_csv(delta_file_path)

    delta_df['Type'] = delta_df['Variable'].apply(get_dkt_type)

    melted_data = delta_df.melt(id_vars=['Dataset', 'Variable', 'Type'], value_vars=['Delta_Coefficient'], var_name='Metric', value_name='Coefficient')

    plt.figure(figsize=(12, 8))
    sns.swarmplot(x='Type', y='Coefficient', hue='Dataset', data=melted_data, dodge=True)
    plt.title('Reaction Time ∆ Coefficients Across Models')
    plt.xlabel('Brain Measurement Type')
    plt.ylabel('∆ Coefficient (model with motion control - model without motion control)')
    plt.legend(title='Effect of')
    plt.show()

visualize_delta_coefficients()

def plot_all_dkt_coefficients(df, dkt_type, title, output_path=None):
    df_filtered = df[df['DKT Type'] == dkt_type]

    print(f"Filtered DataFrame for {dkt_type}:")
    print(df_filtered.head())
    print(f"Columns in df_filtered: {df_filtered.columns.tolist()}")

    fig, ax = plt.subplots(figsize=(20, 15))

    for idx, row in df_filtered.iterrows():
        if 'Coefficient_with_head_motion' not in row or 'Coefficient_without_head_motion' not in row:
            raise KeyError(f"'Coefficient_with_head_motion' or 'Coefficient_without_head_motion' column is missing in the DataFrame for row: {row}")

        with_hm_coef = row['Coefficient_with_head_motion']
        without_hm_coef = row['Coefficient_without_head_motion']
        description = dkt_descriptions_dict.get(row['Variable'], 'No Description')

        print(f"Plotting: Variable={row['Variable']}, with_hm_coef={with_hm_coef}, without_hm_coef={without_hm_coef}, description={description}")

        if pd.notnull(with_hm_coef) and pd.notnull(without_hm_coef):
            if row['Significant_with_head_motion'] and not row['Significant_without_head_motion']:
                color = model_colors['Exclusive to Model with Head Motion']
            elif not row['Significant_with_head_motion'] and row['Significant_without_head_motion']:
                color = model_colors['Exclusive to Model without Head Motion']
            else:
                color = model_colors['Shared Between Both Models']
            jitter = np.random.uniform(-0.02, 0.02, 2)
            ax.plot([0 + jitter[0], 1 + jitter[1]], [with_hm_coef, without_hm_coef], marker='o', markersize=4, linestyle='-', linewidth=1, color=color, alpha=0.7)
            ax.text(1.05, without_hm_coef, description, fontsize=6, verticalalignment='center', color=color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['With Head Motion', 'Without Head Motion'])
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Coefficient")
    ax.axhline(y=0, color='black', linewidth=1, linestyle='-')
    ax.legend(handles=legend_elements, loc='best')

    if output_path:
        plt.savefig(output_path)
    plt.show()

# Create combined dataframe
combined_results = []

for include_head_motion in ['T1', 'rfMRI', 'all']:
    file_with_hm = f"/Users/Cabria/PycharmProjects/pythonProject/Reaction_Time_results_{include_head_motion}_head_motion_FDR.csv"
    file_no_hm = "/Users/Cabria/PycharmProjects/pythonProject/Reaction_Time_results_no_head_motion_FDR.csv"

    df_with_hm = pd.read_csv(file_with_hm)
    df_no_hm = pd.read_csv(file_no_hm)

    df_with_hm['Significant_with_head_motion'] = df_with_hm['Significant After FDR']
    df_no_hm['Significant_without_head_motion'] = df_no_hm['Significant After FDR']

    df_combined = pd.merge(df_with_hm, df_no_hm, on='Variable', suffixes=('_with_head_motion', '_without_head_motion'))
    df_combined['DKT Type'] = df_combined['Variable'].apply(get_dkt_type)

    combined_results.append(df_combined)

df_combined = pd.concat(combined_results)

# Plotting all coefficients
plot_all_dkt_coefficients(df_combined, 'volume', 'All Reaction Time Coefficients for Cortical Volume', "/Users/Cabria/PycharmProjects/pythonProject/Reaction_Time_all_volume_coefficients_plot.png")
plot_all_dkt_coefficients(df_combined, 'thickness', 'All Reaction Time Coefficients for Cortical Thickness', "/Users/Cabria/PycharmProjects/pythonProject/Reaction_Time_all_thickness_coefficients_plot.png")
plot_all_dkt_coefficients(df_combined, 'area', 'All Reaction Time Coefficients for Cortical Area', "/Users/Cabria/PycharmProjects/pythonProject/Reaction_Time_all_area_coefficients_plot.png")

# Plotting only significant coefficients
#plot_significant_dkt_coefficients(df_combined, 'volume', 'Significant Reaction Time Coefficients for Cortical Volume', "/Users/Cabria/PycharmProjects/pythonProject/Reaction_Time_significant_volume_coefficients_plot.png")
#plot_significant_dkt_coefficients(df_combined, 'thickness', 'Significant Reaction Time Coefficients for Cortical Thickness', "/Users/Cabria/PycharmProjects/pythonProject/Reaction_Time_significant_thickness_coefficients_plot.png")
#plot_significant_dkt_coefficients(df_combined, 'area', 'Significant Reaction Time Coefficients for Cortical Area', "/Users/Cabria/PycharmProjects/pythonProject/Reaction_Time_significant_area_coefficients_plot.png")



# Updated IQ Linear Regression Code with 4 different levels
# Importing packages
import pandas as pd
import numpy as np
from scipy.stats import zscore
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from adjustText import adjust_text
import os
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Helper Functions
def is_binary_or_categorical(column, df):
    return df[column].dtype == 'object' or df[column].nunique() == 2

def remove_outliers(df, columns_for_outlier_detection, std_threshold=6):
    df_outlier_detection = df[columns_for_outlier_detection]
    mean = df_outlier_detection.mean()
    std = df_outlier_detection.std()
    outliers = (df_outlier_detection > mean + std_threshold * std) | (df_outlier_detection < mean - std_threshold * std)
    df_clean = df[~outliers.any(axis=1)]

    num_excluded = df.shape[0] - df_clean.shape[0]
    num_remaining = df_clean.shape[0]

    print(f"Excluded {num_excluded} subjects due to outliers.")
    print(f"Remaining subjects: {num_remaining}")

    return df_clean

# Preparing and Loading IQ Data for Precise Linear Regression
df_main = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset3.csv", low_memory=False)
df_main = df_main.dropna()

# IQ calculation
df_main['IQ'] = df_main['20016-2.0']

# Specify columns for outlier detection
columns_for_outlier_detection = [col for col in df_main.columns if not is_binary_or_categorical(col, df_main) and col not in ['IQ']]
df_main = remove_outliers(df_main, columns_for_outlier_detection, std_threshold=6)

# Apply z-score normalization after outlier removal
df_main = df_main.apply(zscore)

# Ensure all required columns are present
required_columns = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0', '25741-2.0', '27143-2.0', '27327-2.0']
missing_columns = [col for col in required_columns if col not in df_main.columns]
if missing_columns:
    raise KeyError(f"The following required columns are missing: {missing_columns}")

# Performing Linear Regression Analysis and Saving Results
def perform_analysis_and_save(df_main, head_motion_columns):
    results_dict = {'T1_head_motion': [], 'rfMRI_head_motion': [], 'all_head_motion': [], 'no_head_motion': []}
    start_index = df_main.columns.get_loc('27143-2.0')
    end_index = df_main.columns.get_loc('27327-2.0')

    for dkt_var in df_main.columns[start_index:end_index + 1]:
        df_main['DKT'] = df_main[dkt_var]

        if df_main['DKT'].isnull().any() or np.isinf(df_main['DKT']).any():
            print(f"NaNs or infinite values detected in DKT calculations for variable {dkt_var}.")
            continue

        for include_head_motion in ['T1', 'rfMRI', 'all', 'none']:
            covariates = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0']
            if include_head_motion == 'T1':
                covariates.append('24419-2.0')
            elif include_head_motion == 'rfMRI':
                covariates.append('25741-2.0')
            elif include_head_motion == 'all':
                covariates.extend(head_motion_columns)

            all_vars = ['DKT'] + covariates
            X = df_main[all_vars]

            if X.isnull().any().any() or np.isinf(X).any().any():
                print(f"NaNs or infinite values detected in X for variable {dkt_var}.")
                continue

            Y = df_main['IQ']
            model = sm.OLS(Y, X).fit()
            result = {
                'Variable': dkt_var,
                'Coefficient': model.params['DKT'],
                'p-value': model.pvalues['DKT']
            }

            key = f"{include_head_motion}_head_motion" if include_head_motion != 'none' else 'no_head_motion'
            results_dict[key].append(result)

    for key in results_dict:
        if results_dict[key]:  # Only save if there are results to save
            results_df = pd.DataFrame(results_dict[key])
            file_name = f"/Users/Cabria/PycharmProjects/pythonProject/IQ_results_{key}.csv"
            results_df.to_csv(file_name, index=False)
            apply_fdr(results_df, key)

    return results_dict

# Applying False Discovery Rate (FDR) Correction to P-values
def apply_fdr(results_df, key):
    p_values = results_df['p-value'].values
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    results_df['p-value_corrected'] = pvals_corrected
    results_df['Significant After FDR'] = reject
    file_name = f"/Users/Cabria/PycharmProjects/pythonProject/IQ_results_{key}_FDR.csv"
    results_df.to_csv(file_name, index=False)
    return results_df

head_motion_columns = [
    '24419-2.0', '24438-2.0', '24439-2.0', '24440-2.0', '24441-2.0',
    '24442-2.0', '24443-2.0', '25741-2.0', '24450-2.0', '24451-2.0',
    '24452-2.0', '24453-2.0', '24454-2.0', '24455-2.0', '24444-2.0',
    '24445-2.0', '24446-2.0', '24447-2.0', '24448-2.0', '24449-2.0',
    '25742-2.0'
]

results_dict = perform_analysis_and_save(df_main, head_motion_columns)
print("Results and FDR-corrected results saved for both models")

# Calculate Delta (Difference) of Coefficients and Save to CSV
def calculate_and_save_delta():
    base_path = "/Users/Cabria/PycharmProjects/pythonProject/"
    file_name_template = "IQ_results_{}_head_motion_FDR.csv"

    delta_list = []

    for include_head_motion in ['T1', 'rfMRI', 'all']:
        file_with_hm = base_path + file_name_template.format(include_head_motion)
        file_no_hm = base_path + "IQ_results_no_head_motion_FDR.csv"

        if not os.path.exists(file_with_hm) or not os.path.exists(file_no_hm):
            print(f"File not found: {file_with_hm} or {file_no_hm}")
            continue

        df_with_hm = pd.read_csv(file_with_hm)
        df_no_hm = pd.read_csv(file_no_hm)

        print(f"Columns in df_with_hm: {df_with_hm.columns.tolist()}")
        print(f"Columns in df_no_hm: {df_no_hm.columns.tolist()}")

        if 'Coefficient' not in df_with_hm.columns or 'Coefficient' not in df_no_hm.columns:
            raise KeyError("'Coefficient' column is missing in one of the dataframes.")

        for variable in df_with_hm['Variable']:
            with_hm_coef = df_with_hm.loc[df_with_hm['Variable'] == variable, 'Coefficient'].values[0]
            without_hm_coef = df_no_hm.loc[df_no_hm['Variable'] == variable, 'Coefficient'].values[0] if variable in df_no_hm['Variable'].values else np.nan
            delta = abs(with_hm_coef) - abs(without_hm_coef)
            delta_list.append({
                'Variable': variable,
                'Coefficient_with_head_motion': with_hm_coef,
                'Coefficient_without_head_motion': without_hm_coef,
                'Delta_Coefficient': delta,
                'Dataset': include_head_motion + ' motion'
            })

    delta_df = pd.DataFrame(delta_list)
    delta_df.to_csv(base_path + "IQ_Delta_Coefficients.csv", index=False)

calculate_and_save_delta()
print("Delta coefficients saved.")

# Combining Results and Creating Plots
# Load the descriptions dictionary
dkt_descriptions_df = pd.read_csv("C:/Users/Cabria/Downloads/DKT_descriptions_dictionary.csv")
dkt_descriptions_df.columns = [col.strip() for col in dkt_descriptions_df.columns]  # Strip any leading/trailing whitespace
dkt_descriptions_df['DKT Variable'] = dkt_descriptions_df['DKT Variable'].astype(str)
dkt_descriptions_dict = pd.Series(dkt_descriptions_df['DKT Description'].values, index=dkt_descriptions_df['DKT Variable']).to_dict()

# Function to classify DKT variables based on description
def get_dkt_type(dkt_variable):
    dkt_description = dkt_descriptions_dict.get(dkt_variable, '').lower()
    if "thickness" in dkt_description:
        return 'thickness'
    elif "area" in dkt_description:
        return 'area'
    elif "volume" in dkt_description:
        return 'volume'
    return 'other'

# Define colors for each group
model_colors = {
    'Exclusive to Model with Head Motion': 'red',
    'Exclusive to Model without Head Motion': 'blue',
    'Shared Between Both Models': 'green'
}

legend_elements = [
    Line2D([0], [0], color='red', lw=4, label='Exclusive to Model with Head Motion'),
    Line2D([0], [0], color='blue', lw=4, label='Exclusive to Model without Head Motion'),
    Line2D([0], [0], color='green', lw=4, label='Shared Between Both Models')
]

# Plotting Significant Coefficients
def plot_significant_dkt_coefficients(df, dkt_type, title, output_path=None):
    df_filtered = df[df['DKT Type'] == dkt_type]
    significant_with_hm = df_filtered['Significant_with_head_motion'] == True
    significant_without_hm = df_filtered['Significant_without_head_motion'] == True
    significant_indices = df_filtered[significant_with_hm | significant_without_hm].index

    fig, ax = plt.subplots(figsize=(20, 15))

    for idx in significant_indices:
        row = df_filtered.loc[idx]
        with_hm_coef = row['Coefficient_with_head_motion']
        without_hm_coef = row['Coefficient_without_head_motion']
        description = dkt_descriptions_dict.get(row['Variable'], 'No Description')

        if significant_with_hm[idx] and not significant_without_hm[idx]:
            color = model_colors['Exclusive to Model with Head Motion']
        elif not significant_with_hm[idx] and significant_without_hm[idx]:
            color = model_colors['Exclusive to Model without Head Motion']
        else:
            color = model_colors['Shared Between Both Models']

        jitter = np.random.uniform(-0.02, 0.02, 2)

        if pd.notnull(with_hm_coef) and pd.notnull(without_hm_coef):
            ax.plot([0 + jitter[0], 1 + jitter[1]], [with_hm_coef, without_hm_coef], marker='o', markersize=4, linestyle='-', linewidth=1, color=color, alpha=0.7)
            ax.text(1.05, without_hm_coef, description, fontsize=6, verticalalignment='center', color=color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['With Head Motion', 'Without Head Motion'])
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Coefficient")
    ax.axhline(y=0, color='black', linewidth=1, linestyle='-')
    ax.legend(handles=legend_elements, loc='best')

    if output_path:
        plt.savefig(output_path)
    plt.show()

# Visualization
def visualize_delta_coefficients():
    base_path = "/Users/Cabria/PycharmProjects/pythonProject/"
    delta_file_path = base_path + "IQ_Delta_Coefficients.csv"

    if not os.path.exists(delta_file_path):
        print(f"File not found: {delta_file_path}")
        return

    delta_df = pd.read_csv(delta_file_path)

    delta_df['Type'] = delta_df['Variable'].apply(get_dkt_type)

    melted_data = delta_df.melt(id_vars=['Dataset', 'Variable', 'Type'], value_vars=['Delta_Coefficient'], var_name='Metric', value_name='Coefficient')

    plt.figure(figsize=(12, 8))
    sns.swarmplot(x='Type', y='Coefficient', hue='Dataset', data=melted_data, dodge=True)
    plt.title('IQ ∆ Coefficients Across Models')
    plt.xlabel('Brain Measurement Type')
    plt.ylabel('∆ Coefficient (model with motion control - model without motion control)')
    plt.legend(title='Effect of')
    plt.show()

visualize_delta_coefficients()

def plot_all_dkt_coefficients(df, dkt_type, title, output_path=None):
    df_filtered = df[df['DKT Type'] == dkt_type]

    print(f"Filtered DataFrame for {dkt_type}:")
    print(df_filtered.head())
    print(f"Columns in df_filtered: {df_filtered.columns.tolist()}")

    fig, ax = plt.subplots(figsize=(20, 15))

    for idx, row in df_filtered.iterrows():
        if 'Coefficient_with_head_motion' not in row or 'Coefficient_without_head_motion' not in row:
            raise KeyError(f"'Coefficient_with_head_motion' or 'Coefficient_without_head_motion' column is missing in the DataFrame for row: {row}")

        with_hm_coef = row['Coefficient_with_head_motion']
        without_hm_coef = row['Coefficient_without_head_motion']
        description = dkt_descriptions_dict.get(row['Variable'], 'No Description')

        print(f"Plotting: Variable={row['Variable']}, with_hm_coef={with_hm_coef}, without_hm_coef={without_hm_coef}, description={description}")

        if pd.notnull(with_hm_coef) and pd.notnull(without_hm_coef):
            if row['Significant_with_head_motion'] and not row['Significant_without_head_motion']:
                color = model_colors['Exclusive to Model with Head Motion']
            elif not row['Significant_with_head_motion'] and row['Significant_without_head_motion']:
                color = model_colors['Exclusive to Model without Head Motion']
            else:
                color = model_colors['Shared Between Both Models']
            jitter = np.random.uniform(-0.02, 0.02, 2)
            ax.plot([0 + jitter[0], 1 + jitter[1]], [with_hm_coef, without_hm_coef], marker='o', markersize=4, linestyle='-', linewidth=1, color=color, alpha=0.7)
            ax.text(1.05, without_hm_coef, description, fontsize=6, verticalalignment='center', color=color)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['With Head Motion', 'Without Head Motion'])
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Coefficient")
    ax.axhline(y=0, color='black', linewidth=1, linestyle='-')
    ax.legend(handles=legend_elements, loc='best')

    if output_path:
        plt.savefig(output_path)
    plt.show()

# Create combined dataframe
combined_results = []

for include_head_motion in ['T1', 'rfMRI', 'all']:
    file_with_hm = f"/Users/Cabria/PycharmProjects/pythonProject/IQ_results_{include_head_motion}_head_motion_FDR.csv"
    file_no_hm = "/Users/Cabria/PycharmProjects/pythonProject/IQ_results_no_head_motion_FDR.csv"

    df_with_hm = pd.read_csv(file_with_hm)
    df_no_hm = pd.read_csv(file_no_hm)

    df_with_hm['Significant_with_head_motion'] = df_with_hm['Significant After FDR']
    df_no_hm['Significant_without_head_motion'] = df_no_hm['Significant After FDR']

    df_combined = pd.merge(df_with_hm, df_no_hm, on='Variable', suffixes=('_with_head_motion', '_without_head_motion'))
    df_combined['DKT Type'] = df_combined['Variable'].apply(get_dkt_type)

    combined_results.append(df_combined)

df_combined = pd.concat(combined_results)

# Plotting all coefficients
plot_all_dkt_coefficients(df_combined, 'volume', 'All IQ Coefficients for Cortical Volume', "/Users/Cabria/PycharmProjects/pythonProject/IQ_all_volume_coefficients_plot.png")
plot_all_dkt_coefficients(df_combined, 'thickness', 'All IQ Coefficients for Cortical Thickness', "/Users/Cabria/PycharmProjects/pythonProject/IQ_all_thickness_coefficients_plot.png")
plot_all_dkt_coefficients(df_combined, 'area', 'All IQ Coefficients for Cortical Area', "/Users/Cabria/PycharmProjects/pythonProject/IQ_all_area_coefficients_plot.png")

# Plotting only significant coefficients
#plot_significant_dkt_coefficients(df_combined, 'volume', 'Significant IQ Coefficients for Cortical Volume', "/Users/Cabria/PycharmProjects/pythonProject/IQ_significant_volume_coefficients_plot.png")
#plot_significant_dkt_coefficients(df_combined, 'thickness', 'Significant IQ Coefficients for Cortical Thickness', "/Users/Cabria/PycharmProjects/pythonProject/IQ_significant_thickness_coefficients_plot.png")
#plot_significant_dkt_coefficients(df_combined, 'area', 'Significant IQ Coefficients for Cortical Area', "/Users/Cabria/PycharmProjects/pythonProject/IQ_significant_area_coefficients_plot.png")





#New IQ Swarm Plot w/o seperating on brain measures
# Updated IQ Linear Regression Code with 4 different levels
# Importing packages
import pandas as pd
import numpy as np
from scipy.stats import zscore
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from adjustText import adjust_text
import os
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Helper Functions
def is_binary_or_categorical(column, df):
    return df[column].dtype == 'object' or df[column].nunique() == 2

def remove_outliers(df, columns_for_outlier_detection, std_threshold=6):
    df_outlier_detection = df[columns_for_outlier_detection]
    mean = df_outlier_detection.mean()
    std = df_outlier_detection.std()
    outliers = (df_outlier_detection > mean + std_threshold * std) | (df_outlier_detection < mean - std_threshold * std)
    df_clean = df[~outliers.any(axis=1)]

    num_excluded = df.shape[0] - df_clean.shape[0]
    num_remaining = df_clean.shape[0]

    print(f"Excluded {num_excluded} subjects due to outliers.")
    print(f"Remaining subjects: {num_remaining}")

    return df_clean

# Preparing and Loading IQ Data for Precise Linear Regression
df_main = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset3.csv", low_memory=False)
df_main = df_main.dropna()

# IQ calculation
df_main['IQ'] = df_main['20016-2.0']

# Specify columns for outlier detection
columns_for_outlier_detection = [col for col in df_main.columns if not is_binary_or_categorical(col, df_main) and col not in ['IQ']]
df_main = remove_outliers(df_main, columns_for_outlier_detection, std_threshold=6)

# Apply z-score normalization after outlier removal
df_main = df_main.apply(zscore)

# Ensure all required columns are present
required_columns = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0', '25741-2.0', '27143-2.0', '27327-2.0']
missing_columns = [col for col in required_columns if col not in df_main.columns]
if missing_columns:
    raise KeyError(f"The following required columns are missing: {missing_columns}")

# Performing Linear Regression Analysis and Saving Results
def perform_analysis_and_save(df_main, head_motion_columns):
    results_dict = {'T1_head_motion': [], 'rfMRI_head_motion': [], 'all_head_motion': [], 'no_head_motion': []}
    start_index = df_main.columns.get_loc('27143-2.0')
    end_index = df_main.columns.get_loc('27327-2.0')

    for dkt_var in df_main.columns[start_index:end_index + 1]:
        df_main['DKT'] = df_main[dkt_var]

        if df_main['DKT'].isnull().any() or np.isinf(df_main['DKT']).any():
            print(f"NaNs or infinite values detected in DKT calculations for variable {dkt_var}.")
            continue

        for include_head_motion in ['T1', 'rfMRI', 'all', 'none']:
            covariates = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0']
            if include_head_motion == 'T1':
                covariates.append('24419-2.0')
            elif include_head_motion == 'rfMRI':
                covariates.append('25741-2.0')
            elif include_head_motion == 'all':
                covariates.extend(head_motion_columns)

            all_vars = ['DKT'] + covariates
            X = df_main[all_vars]

            if X.isnull().any().any() or np.isinf(X).any().any():
                print(f"NaNs or infinite values detected in X for variable {dkt_var}.")
                continue

            Y = df_main['IQ']
            model = sm.OLS(Y, X).fit()
            result = {
                'Variable': dkt_var,
                'Coefficient': model.params['DKT'],
                'p-value': model.pvalues['DKT']
            }

            key = f"{include_head_motion}_head_motion" if include_head_motion != 'none' else 'no_head_motion'
            results_dict[key].append(result)

    for key in results_dict:
        if results_dict[key]:  # Only save if there are results to save
            results_df = pd.DataFrame(results_dict[key])
            file_name = f"/Users/Cabria/PycharmProjects/pythonProject/IQ_results_{key}.csv"
            results_df.to_csv(file_name, index=False)
            apply_fdr(results_df, key)

    return results_dict

# Applying False Discovery Rate (FDR) Correction to P-values
def apply_fdr(results_df, key):
    p_values = results_df['p-value'].values
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    results_df['p-value_corrected'] = pvals_corrected
    results_df['Significant After FDR'] = reject
    file_name = f"/Users/Cabria/PycharmProjects/pythonProject/IQ_results_{key}_FDR.csv"
    results_df.to_csv(file_name, index=False)
    return results_df

head_motion_columns = [
    '24419-2.0', '24438-2.0', '24439-2.0', '24440-2.0', '24441-2.0',
    '24442-2.0', '24443-2.0', '25741-2.0', '24450-2.0', '24451-2.0',
    '24452-2.0', '24453-2.0', '24454-2.0', '24455-2.0', '24444-2.0',
    '24445-2.0', '24446-2.0', '24447-2.0', '24448-2.0', '24449-2.0',
    '25742-2.0'
]

results_dict = perform_analysis_and_save(df_main, head_motion_columns)
print("Results and FDR-corrected results saved for both models")

# Calculate Delta (Difference) of Coefficients and Save to CSV
def calculate_and_save_delta():
    base_path = "/Users/Cabria/PycharmProjects/pythonProject/"
    file_name_template = "IQ_results_{}_head_motion_FDR.csv"

    delta_list = []

    for include_head_motion in ['T1', 'rfMRI', 'all']:
        file_with_hm = base_path + file_name_template.format(include_head_motion)
        file_no_hm = base_path + "IQ_results_no_head_motion_FDR.csv"

        if not os.path.exists(file_with_hm) or not os.path.exists(file_no_hm):
            print(f"File not found: {file_with_hm} or {file_no_hm}")
            continue

        df_with_hm = pd.read_csv(file_with_hm)
        df_no_hm = pd.read_csv(file_no_hm)

        print(f"Columns in df_with_hm: {df_with_hm.columns.tolist()}")
        print(f"Columns in df_no_hm: {df_no_hm.columns.tolist()}")

        if 'Coefficient' not in df_with_hm.columns or 'Coefficient' not in df_no_hm.columns:
            raise KeyError("'Coefficient' column is missing in one of the dataframes.")

        for variable in df_with_hm['Variable']:
            with_hm_coef = df_with_hm.loc[df_with_hm['Variable'] == variable, 'Coefficient'].values[0]
            without_hm_coef = df_no_hm.loc[df_no_hm['Variable'] == variable, 'Coefficient'].values[0] if variable in df_no_hm['Variable'].values else np.nan
            delta = abs(with_hm_coef) - abs(without_hm_coef)
            delta_list.append({
                'Variable': variable,
                'Coefficient_with_head_motion': with_hm_coef,
                'Coefficient_without_head_motion': without_hm_coef,
                'Delta_Coefficient': delta,
                'Dataset': include_head_motion + ' motion'
            })

    delta_df = pd.DataFrame(delta_list)
    delta_df.to_csv(base_path + "IQ_Delta_Coefficients.csv", index=False)

calculate_and_save_delta()
print("Delta coefficients saved.")

# Visualization
def visualize_delta_coefficients():
    base_path = "/Users/Cabria/PycharmProjects/pythonProject/"
    delta_file_path = base_path + "IQ_Delta_Coefficients.csv"

    if not os.path.exists(delta_file_path):
        print(f"File not found: {delta_file_path}")
        return

    delta_df = pd.read_csv(delta_file_path)

    plt.figure(figsize=(12, 8))
    sns.swarmplot(x='Dataset', y='Delta_Coefficient', data=delta_df, palette=['blue', 'red', 'orange'])
    plt.title('IQ ∆ Coefficients Across Models')
    plt.xlabel('Which head motion variable is controlled for')
    plt.ylabel('∆ Coefficient abs(model with motion control) - abs(model without motion control)')
    plt.ylim(-0.01, 0.01)
    plt.show()

visualize_delta_coefficients()




# Updated Depression Linear Regression Code with 4 different levels
# Importing packages
import pandas as pd
import numpy as np
from scipy.stats import zscore
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from adjustText import adjust_text
import os
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Helper Functions
def is_binary_or_categorical(column, df):
    return df[column].dtype == 'object' or df[column].nunique() == 2

def remove_outliers(df, columns_for_outlier_detection, std_threshold=6):
    df_outlier_detection = df[columns_for_outlier_detection]
    mean = df_outlier_detection.mean()
    std = df_outlier_detection.std()
    outliers = (df_outlier_detection > mean + std_threshold * std) | (df_outlier_detection < mean - std_threshold * std)
    df_clean = df[~outliers.any(axis=1)]

    num_excluded = df.shape[0] - df_clean.shape[0]
    num_remaining = df_clean.shape[0]

    print(f"Excluded {num_excluded} subjects due to outliers.")
    print(f"Remaining subjects: {num_remaining}")

    return df_clean

# Preparing and Loading RDS-4 Data for Precise Linear Regression
df_main = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset3.csv", low_memory=False)
df_main = df_main.dropna()

df_main['RDS'] = df_main[['2050-2.0', '2060-2.0', '2070-2.0', '2080-2.0']].sum(axis=1)

# Specify columns for outlier detection
columns_for_outlier_detection = [col for col in df_main.columns if not is_binary_or_categorical(col, df_main) and col not in ['RDS']]
df_main = remove_outliers(df_main, columns_for_outlier_detection, std_threshold=6)

# Apply z-score normalization after outlier removal
df_main = df_main.apply(zscore)

# Ensure all required columns are present
required_columns = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0', '25741-2.0', '27143-2.0', '27327-2.0']
missing_columns = [col for col in required_columns if col not in df_main.columns]
if missing_columns:
    raise KeyError(f"The following required columns are missing: {missing_columns}")

# Performing Linear Regression Analysis and Saving Results
def perform_analysis_and_save(df_main, head_motion_columns):
    results_dict = {'T1_head_motion': [], 'rfMRI_head_motion': [], 'all_head_motion': [], 'no_head_motion': []}
    start_index = df_main.columns.get_loc('27143-2.0')
    end_index = df_main.columns.get_loc('27327-2.0')

    for dkt_var in df_main.columns[start_index:end_index + 1]:
        df_main['DKT'] = df_main[dkt_var]

        if df_main['DKT'].isnull().any() or np.isinf(df_main['DKT']).any():
            print(f"NaNs or infinite values detected in DKT calculations for variable {dkt_var}.")
            continue

        for include_head_motion in ['T1', 'rfMRI', 'all', 'none']:
            covariates = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0']
            if include_head_motion == 'T1':
                covariates.append('24419-2.0')
            elif include_head_motion == 'rfMRI':
                covariates.append('25741-2.0')
            elif include_head_motion == 'all':
                covariates.extend(head_motion_columns)

            all_vars = ['DKT'] + covariates
            X = df_main[all_vars]

            if X.isnull().any().any() or np.isinf(X).any().any():
                print(f"NaNs or infinite values detected in X for variable {dkt_var}.")
                continue

            Y = df_main['RDS']
            model = sm.OLS(Y, X).fit()
            result = {
                'Variable': dkt_var,
                'Coefficient': model.params['DKT'],
                'p-value': model.pvalues['DKT']
            }

            key = f"{include_head_motion}_head_motion" if include_head_motion != 'none' else 'no_head_motion'
            results_dict[key].append(result)

    for key in results_dict:
        if results_dict[key]:  # Only save if there are results to save
            results_df = pd.DataFrame(results_dict[key])
            file_name = f"/Users/Cabria/PycharmProjects/pythonProject/RDS_results_{key}.csv"
            results_df.to_csv(file_name, index=False)
            apply_fdr(results_df, key)

    return results_dict

# Applying False Discovery Rate (FDR) Correction to P-values
def apply_fdr(results_df, key):
    p_values = results_df['p-value'].values
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    results_df['p-value_corrected'] = pvals_corrected
    results_df['Significant After FDR'] = reject
    file_name = f"/Users/Cabria/PycharmProjects/pythonProject/RDS_results_{key}_FDR.csv"
    results_df.to_csv(file_name, index=False)
    return results_df

head_motion_columns = [
    '24419-2.0', '24438-2.0', '24439-2.0', '24440-2.0', '24441-2.0',
    '24442-2.0', '24443-2.0', '25741-2.0', '24450-2.0', '24451-2.0',
    '24452-2.0', '24453-2.0', '24454-2.0', '24455-2.0', '24444-2.0',
    '24445-2.0', '24446-2.0', '24447-2.0', '24448-2.0', '24449-2.0',
    '25742-2.0'
]

results_dict = perform_analysis_and_save(df_main, head_motion_columns)
print("Results and FDR-corrected results saved for both models")

# Calculate Delta (Difference) of Coefficients and Save to CSV
def calculate_and_save_delta():
    base_path = "/Users/Cabria/PycharmProjects/pythonProject/"
    file_name_template = "RDS_results_{}_head_motion_FDR.csv"

    delta_list = []

    for include_head_motion in ['T1', 'rfMRI', 'all']:
        file_with_hm = base_path + file_name_template.format(include_head_motion)
        file_no_hm = base_path + "RDS_results_no_head_motion_FDR.csv"

        if not os.path.exists(file_with_hm) or not os.path.exists(file_no_hm):
            print(f"File not found: {file_with_hm} or {file_no_hm}")
            continue

        df_with_hm = pd.read_csv(file_with_hm)
        df_no_hm = pd.read_csv(file_no_hm)

        print(f"Columns in df_with_hm: {df_with_hm.columns.tolist()}")
        print(f"Columns in df_no_hm: {df_no_hm.columns.tolist()}")

        if 'Coefficient' not in df_with_hm.columns or 'Coefficient' not in df_no_hm.columns:
            raise KeyError("'Coefficient' column is missing in one of the dataframes.")

        for variable in df_with_hm['Variable']:
            with_hm_coef = df_with_hm.loc[df_with_hm['Variable'] == variable, 'Coefficient'].values[0]
            without_hm_coef = df_no_hm.loc[df_no_hm['Variable'] == variable, 'Coefficient'].values[0] if variable in df_no_hm['Variable'].values else np.nan
            delta = abs(with_hm_coef) - abs(without_hm_coef)
            delta_list.append({
                'Variable': variable,
                'Coefficient_with_head_motion': with_hm_coef,
                'Coefficient_without_head_motion': without_hm_coef,
                'Delta_Coefficient': delta,
                'Dataset': include_head_motion + ' motion'
            })

    delta_df = pd.DataFrame(delta_list)
    delta_df.to_csv(base_path + "RDS_Delta_Coefficients.csv", index=False)

calculate_and_save_delta()
print("Delta coefficients saved.")

# Visualization
def visualize_delta_coefficients():
    base_path = "/Users/Cabria/PycharmProjects/pythonProject/"
    delta_file_path = base_path + "RDS_Delta_Coefficients.csv"

    if not os.path.exists(delta_file_path):
        print(f"File not found: {delta_file_path}")
        return

    delta_df = pd.read_csv(delta_file_path)

    plt.figure(figsize=(12, 8))
    sns.swarmplot(x='Dataset', y='Delta_Coefficient', data=delta_df, palette=['blue', 'red', 'orange'])
    plt.title('Depression (RDS)-4 ∆ Coefficients Across Models')
    plt.xlabel('Which head motion variable is controlled for')
    plt.ylabel('∆ Coefficient abs(model with motion control) - abs(model without motion control)')
    plt.ylim(-0.01, 0.01)
    plt.show()

visualize_delta_coefficients()





# Updated Neuroticism (N-12) Linear Regression Code with 4 different levels
# Importing packages
import pandas as pd
import numpy as np
from scipy.stats import zscore
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from adjustText import adjust_text
import os
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Helper Functions
def is_binary_or_categorical(column, df):
    return df[column].dtype == 'object' or df[column].nunique() == 2

def remove_outliers(df, columns_for_outlier_detection, std_threshold=6):
    df_outlier_detection = df[columns_for_outlier_detection]
    mean = df_outlier_detection.mean()
    std = df_outlier_detection.std()
    outliers = (df_outlier_detection > mean + std_threshold * std) | (df_outlier_detection < mean - std_threshold * std)
    df_clean = df[~outliers.any(axis=1)]

    num_excluded = df.shape[0] - df_clean.shape[0]
    num_remaining = df_clean.shape[0]

    print(f"Excluded {num_excluded} subjects due to outliers.")
    print(f"Remaining subjects: {num_remaining}")

    return df_clean

# Preparing and Loading N-12 Data for Precise Linear Regression
df_main = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset3.csv", low_memory=False)
df_main = df_main.dropna()

# N-12 calculation and demeaning
n_12_columns = ['1920-2.0', '1930-2.0', '1940-2.0', '1950-2.0', '1960-2.0',
                '1970-2.0', '1980-2.0', '1990-2.0', '2000-2.0', '2010-2.0',
                '2020-2.0', '2030-2.0']
df_main['N_12'] = df_main[n_12_columns].sum(axis=1)

# Specify columns for outlier detection
columns_for_outlier_detection = [col for col in df_main.columns if not is_binary_or_categorical(col, df_main) and col not in ['N_12']]
df_main = remove_outliers(df_main, columns_for_outlier_detection, std_threshold=6)

# Apply z-score normalization after outlier removal
df_main = df_main.apply(zscore)

# Ensure all required columns are present
required_columns = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0', '25741-2.0', '27143-2.0', '27327-2.0']
missing_columns = [col for col in required_columns if col not in df_main.columns]
if missing_columns:
    raise KeyError(f"The following required columns are missing: {missing_columns}")

# Performing Linear Regression Analysis and Saving Results
def perform_analysis_and_save(df_main, head_motion_columns):
    results_dict = {'T1_head_motion': [], 'rfMRI_head_motion': [], 'all_head_motion': [], 'no_head_motion': []}
    start_index = df_main.columns.get_loc('27143-2.0')
    end_index = df_main.columns.get_loc('27327-2.0')

    for dkt_var in df_main.columns[start_index:end_index + 1]:
        df_main['DKT'] = df_main[dkt_var]

        if df_main['DKT'].isnull().any() or np.isinf(df_main['DKT']).any():
            print(f"NaNs or infinite values detected in DKT calculations for variable {dkt_var}.")
            continue

        for include_head_motion in ['T1', 'rfMRI', 'all', 'none']:
            covariates = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0']
            if include_head_motion == 'T1':
                covariates.append('24419-2.0')
            elif include_head_motion == 'rfMRI':
                covariates.append('25741-2.0')
            elif include_head_motion == 'all':
                covariates.extend(head_motion_columns)

            all_vars = ['DKT'] + covariates
            X = df_main[all_vars]

            if X.isnull().any().any() or np.isinf(X).any().any():
                print(f"NaNs or infinite values detected in X for variable {dkt_var}.")
                continue

            Y = df_main['N_12']
            model = sm.OLS(Y, X).fit()
            result = {
                'Variable': dkt_var,
                'Coefficient': model.params['DKT'],
                'p-value': model.pvalues['DKT']
            }

            key = f"{include_head_motion}_head_motion" if include_head_motion != 'none' else 'no_head_motion'
            results_dict[key].append(result)

    for key in results_dict:
        if results_dict[key]:  # Only save if there are results to save
            results_df = pd.DataFrame(results_dict[key])
            file_name = f"/Users/Cabria/PycharmProjects/pythonProject/N_12_results_{key}.csv"
            results_df.to_csv(file_name, index=False)
            apply_fdr(results_df, key)

    return results_dict

# Applying False Discovery Rate (FDR) Correction to P-values
def apply_fdr(results_df, key):
    p_values = results_df['p-value'].values
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    results_df['p-value_corrected'] = pvals_corrected
    results_df['Significant After FDR'] = reject
    file_name = f"/Users/Cabria/PycharmProjects/pythonProject/N_12_results_{key}_FDR.csv"
    results_df.to_csv(file_name, index=False)
    return results_df

head_motion_columns = [
    '24419-2.0', '24438-2.0', '24439-2.0', '24440-2.0', '24441-2.0',
    '24442-2.0', '24443-2.0', '25741-2.0', '24450-2.0', '24451-2.0',
    '24452-2.0', '24453-2.0', '24454-2.0', '24455-2.0', '24444-2.0',
    '24445-2.0', '24446-2.0', '24447-2.0', '24448-2.0', '24449-2.0',
    '25742-2.0'
]

results_dict = perform_analysis_and_save(df_main, head_motion_columns)
print("Results and FDR-corrected results saved for both models")

# Calculate Delta (Difference) of Coefficients and Save to CSV
def calculate_and_save_delta():
    base_path = "/Users/Cabria/PycharmProjects/pythonProject/"
    file_name_template = "N_12_results_{}_head_motion_FDR.csv"

    delta_list = []

    for include_head_motion in ['T1', 'rfMRI', 'all']:
        file_with_hm = base_path + file_name_template.format(include_head_motion)
        file_no_hm = base_path + "N_12_results_no_head_motion_FDR.csv"

        if not os.path.exists(file_with_hm) or not os.path.exists(file_no_hm):
            print(f"File not found: {file_with_hm} or {file_no_hm}")
            continue

        df_with_hm = pd.read_csv(file_with_hm)
        df_no_hm = pd.read_csv(file_no_hm)

        print(f"Columns in df_with_hm: {df_with_hm.columns.tolist()}")
        print(f"Columns in df_no_hm: {df_no_hm.columns.tolist()}")

        if 'Coefficient' not in df_with_hm.columns or 'Coefficient' not in df_no_hm.columns:
            raise KeyError("'Coefficient' column is missing in one of the dataframes.")

        for variable in df_with_hm['Variable']:
            with_hm_coef = df_with_hm.loc[df_with_hm['Variable'] == variable, 'Coefficient'].values[0]
            without_hm_coef = df_no_hm.loc[df_no_hm['Variable'] == variable, 'Coefficient'].values[0] if variable in df_no_hm['Variable'].values else np.nan
            delta = abs(with_hm_coef) - abs(without_hm_coef)
            delta_list.append({
                'Variable': variable,
                'Coefficient_with_head_motion': with_hm_coef,
                'Coefficient_without_head_motion': without_hm_coef,
                'Delta_Coefficient': delta,
                'Dataset': include_head_motion + ' motion'
            })

    delta_df = pd.DataFrame(delta_list)
    delta_df.to_csv(base_path + "N_12_Delta_Coefficients.csv", index=False)

calculate_and_save_delta()
print("Delta coefficients saved.")

# Visualization
def visualize_delta_coefficients():
    base_path = "/Users/Cabria/PycharmProjects/pythonProject/"
    delta_file_path = base_path + "N_12_Delta_Coefficients.csv"

    if not os.path.exists(delta_file_path):
        print(f"File not found: {delta_file_path}")
        return

    delta_df = pd.read_csv(delta_file_path)

    plt.figure(figsize=(12, 8))
    sns.swarmplot(x='Dataset', y='Delta_Coefficient', data=delta_df, palette=['blue', 'red', 'orange'])
    plt.title('Neuroticism (N-12) ∆ Coefficients Across Models')
    plt.xlabel('Which head motion variable is controlled for')
    plt.ylabel('∆ Coefficient abs(model with motion control) - abs(model without motion control)')
    plt.ylim(-0.01, 0.01)
    plt.show()

visualize_delta_coefficients()




# Updated Reaction Time Linear Regression Code with 4 different levels
# Importing packages
import pandas as pd
import numpy as np
from scipy.stats import zscore
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from adjustText import adjust_text
import os
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Helper Functions
def is_binary_or_categorical(column, df):
    return df[column].dtype == 'object' or df[column].nunique() == 2

def remove_outliers(df, columns_for_outlier_detection, std_threshold=6):
    df_outlier_detection = df[columns_for_outlier_detection]
    mean = df_outlier_detection.mean()
    std = df_outlier_detection.std()
    outliers = (df_outlier_detection > mean + std_threshold * std) | (df_outlier_detection < mean - std_threshold * std)
    df_clean = df[~outliers.any(axis=1)]

    num_excluded = df.shape[0] - df_clean.shape[0]
    num_remaining = df_clean.shape[0]

    print(f"Excluded {num_excluded} subjects due to outliers.")
    print(f"Remaining subjects: {num_remaining}")

    return df_clean

# Preparing and Loading Reaction Time Data for Precise Linear Regression
df_main = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset3.csv", low_memory=False)
df_main = df_main.dropna()

# Reaction Time calculation
df_main['Reaction_Time'] = df_main['20023-2.0']

# Specify columns for outlier detection
columns_for_outlier_detection = [col for col in df_main.columns if not is_binary_or_categorical(col, df_main) and col not in ['Reaction_Time']]
df_main = remove_outliers(df_main, columns_for_outlier_detection, std_threshold=6)

# Apply z-score normalization after outlier removal
df_main = df_main.apply(zscore)

# Ensure all required columns are present
required_columns = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0', '25741-2.0', '27143-2.0', '27327-2.0']
missing_columns = [col for col in required_columns if col not in df_main.columns]
if missing_columns:
    raise KeyError(f"The following required columns are missing: {missing_columns}")

# Performing Linear Regression Analysis and Saving Results
def perform_analysis_and_save(df_main, head_motion_columns):
    results_dict = {'T1_head_motion': [], 'rfMRI_head_motion': [], 'all_head_motion': [], 'no_head_motion': []}
    start_index = df_main.columns.get_loc('27143-2.0')
    end_index = df_main.columns.get_loc('27327-2.0')

    for dkt_var in df_main.columns[start_index:end_index + 1]:
        df_main['DKT'] = df_main[dkt_var]

        if df_main['DKT'].isnull().any() or np.isinf(df_main['DKT']).any():
            print(f"NaNs or infinite values detected in DKT calculations for variable {dkt_var}.")
            continue

        for include_head_motion in ['T1', 'rfMRI', 'all', 'none']:
            covariates = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0']
            if include_head_motion == 'T1':
                covariates.append('24419-2.0')
            elif include_head_motion == 'rfMRI':
                covariates.append('25741-2.0')
            elif include_head_motion == 'all':
                covariates.extend(head_motion_columns)

            all_vars = ['DKT'] + covariates
            X = df_main[all_vars]

            if X.isnull().any().any() or np.isinf(X).any().any():
                print(f"NaNs or infinite values detected in X for variable {dkt_var}.")
                continue

            Y = df_main['Reaction_Time']
            model = sm.OLS(Y, X).fit()
            result = {
                'Variable': dkt_var,
                'Coefficient': model.params['DKT'],
                'p-value': model.pvalues['DKT']
            }

            key = f"{include_head_motion}_head_motion" if include_head_motion != 'none' else 'no_head_motion'
            results_dict[key].append(result)

    for key in results_dict:
        if results_dict[key]:  # Only save if there are results to save
            results_df = pd.DataFrame(results_dict[key])
            file_name = f"/Users/Cabria/PycharmProjects/pythonProject/Reaction_Time_results_{key}.csv"
            results_df.to_csv(file_name, index=False)
            apply_fdr(results_df, key)

    return results_dict

# Applying False Discovery Rate (FDR) Correction to P-values
def apply_fdr(results_df, key):
    p_values = results_df['p-value'].values
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    results_df['p-value_corrected'] = pvals_corrected
    results_df['Significant After FDR'] = reject
    file_name = f"/Users/Cabria/PycharmProjects/pythonProject/Reaction_Time_results_{key}_FDR.csv"
    results_df.to_csv(file_name, index=False)
    return results_df

head_motion_columns = [
    '24419-2.0', '24438-2.0', '24439-2.0', '24440-2.0', '24441-2.0',
    '24442-2.0', '24443-2.0', '25741-2.0', '24450-2.0', '24451-2.0',
    '24452-2.0', '24453-2.0', '24454-2.0', '24455-2.0', '24444-2.0',
    '24445-2.0', '24446-2.0', '24447-2.0', '24448-2.0', '24449-2.0',
    '25742-2.0'
]

results_dict = perform_analysis_and_save(df_main, head_motion_columns)
print("Results and FDR-corrected results saved for both models")

# Calculate Delta (Difference) of Coefficients and Save to CSV
def calculate_and_save_delta():
    base_path = "/Users/Cabria/PycharmProjects/pythonProject/"
    file_name_template = "Reaction_Time_results_{}_head_motion_FDR.csv"

    delta_list = []

    for include_head_motion in ['T1', 'rfMRI', 'all']:
        file_with_hm = base_path + file_name_template.format(include_head_motion)
        file_no_hm = base_path + "Reaction_Time_results_no_head_motion_FDR.csv"

        if not os.path.exists(file_with_hm) or not os.path.exists(file_no_hm):
            print(f"File not found: {file_with_hm} or {file_no_hm}")
            continue

        df_with_hm = pd.read_csv(file_with_hm)
        df_no_hm = pd.read_csv(file_no_hm)

        print(f"Columns in df_with_hm: {df_with_hm.columns.tolist()}")
        print(f"Columns in df_no_hm: {df_no_hm.columns.tolist()}")

        if 'Coefficient' not in df_with_hm.columns or 'Coefficient' not in df_no_hm.columns:
            raise KeyError("'Coefficient' column is missing in one of the dataframes.")

        for variable in df_with_hm['Variable']:
            with_hm_coef = df_with_hm.loc[df_with_hm['Variable'] == variable, 'Coefficient'].values[0]
            without_hm_coef = df_no_hm.loc[df_no_hm['Variable'] == variable, 'Coefficient'].values[0] if variable in df_no_hm['Variable'].values else np.nan
            delta = abs(with_hm_coef) - abs(without_hm_coef)
            delta_list.append({
                'Variable': variable,
                'Coefficient_with_head_motion': with_hm_coef,
                'Coefficient_without_head_motion': without_hm_coef,
                'Delta_Coefficient': delta,
                'Dataset': include_head_motion + ' motion'
            })

    delta_df = pd.DataFrame(delta_list)
    delta_df.to_csv(base_path + "Reaction_Time_Delta_Coefficients.csv", index=False)

calculate_and_save_delta()
print("Delta coefficients saved.")

# Visualization
def visualize_delta_coefficients():
    base_path = "/Users/Cabria/PycharmProjects/pythonProject/"
    delta_file_path = base_path + "Reaction_Time_Delta_Coefficients.csv"

    if not os.path.exists(delta_file_path):
        print(f"File not found: {delta_file_path}")
        return

    delta_df = pd.read_csv(delta_file_path)

    plt.figure(figsize=(12, 8))
    sns.swarmplot(x='Dataset', y='Delta_Coefficient', data=delta_df, palette=['blue', 'red', 'orange'])
    plt.title('Reaction Time ∆ Coefficients Across Models')
    plt.xlabel('Which head motion variable is controlled for')
    plt.ylabel('∆ Coefficient abs(model with motion control) - abs(model without motion control)')
    plt.ylim(-0.01, 0.01)
    plt.show()

visualize_delta_coefficients()





# RDS-4 Relative Change with Absolute Value
# Importing packages
import pandas as pd
import numpy as np
from scipy.stats import zscore
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import os
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Helper Functions
def is_binary_or_categorical(column, df):
    return df[column].dtype == 'object' or df[column].nunique() == 2

def remove_outliers(df, columns_for_outlier_detection, std_threshold=6):
    df_outlier_detection = df[columns_for_outlier_detection]
    mean = df_outlier_detection.mean()
    std = df_outlier_detection.std()
    outliers = (df_outlier_detection > mean + std_threshold * std) | (df_outlier_detection < mean - std_threshold * std)
    df_clean = df[~outliers.any(axis=1)]

    num_excluded = df.shape[0] - df_clean.shape[0]
    num_remaining = df_clean.shape[0]

    print(f"Excluded {num_excluded} subjects due to outliers.")
    print(f"Remaining subjects: {num_remaining}")

    return df_clean

# Preparing and Loading RDS-4 Data for Precise Linear Regression
df_main = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset3.csv", low_memory=False)
df_main = df_main.dropna()

df_main['RDS'] = df_main[['2050-2.0', '2060-2.0', '2070-2.0', '2080-2.0']].sum(axis=1)

# Specify columns for outlier detection
columns_for_outlier_detection = [col for col in df_main.columns if not is_binary_or_categorical(col, df_main) and col not in ['RDS']]
df_main = remove_outliers(df_main, columns_for_outlier_detection, std_threshold=6)

# Apply z-score normalization after outlier removal
df_main = df_main.apply(zscore)

# Ensure all required columns are present
required_columns = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0', '25741-2.0', '27143-2.0', '27327-2.0']
missing_columns = [col for col in required_columns if col not in df_main.columns]
if missing_columns:
    raise KeyError(f"The following required columns are missing: {missing_columns}")

# Performing Linear Regression Analysis and Saving Results
def perform_analysis_and_save(df_main, head_motion_columns):
    results_dict = {'T1_head_motion': [], 'rfMRI_head_motion': [], 'all_head_motion': [], 'no_head_motion': []}
    start_index = df_main.columns.get_loc('27143-2.0')
    end_index = df_main.columns.get_loc('27327-2.0')

    total_dkt_variables = 0
    for dkt_var in df_main.columns[start_index:end_index + 1]:
        total_dkt_variables += 1
        df_main['DKT'] = df_main[dkt_var]

        if df_main['DKT'].isnull().any() or np.isinf(df_main['DKT']).any():
            print(f"NaNs or infinite values detected in DKT calculations for variable {dkt_var}. Skipping.")
            continue

        for include_head_motion in ['T1', 'rfMRI', 'all', 'none']:
            covariates = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0']
            if include_head_motion == 'T1':
                covariates.append('24419-2.0')
            elif include_head_motion == 'rfMRI':
                covariates.append('25741-2.0')
            elif include_head_motion == 'all':
                covariates.extend(head_motion_columns)

            all_vars = ['DKT'] + covariates
            X = df_main[all_vars]

            if X.isnull().any().any() or np.isinf(X).any().any():
                print(f"NaNs or infinite values detected in X for variable {dkt_var} and model {include_head_motion}. Skipping.")
                continue

            Y = df_main['RDS']
            model = sm.OLS(Y, X).fit()
            result = {
                'Variable': dkt_var,
                'Coefficient': model.params['DKT'],
                'p-value': model.pvalues['DKT']
            }

            key = f"{include_head_motion}_head_motion" if include_head_motion != 'none' else 'no_head_motion'
            results_dict[key].append(result)

    # Output debug information
    print(f"Total DKT variables expected: 186")
    print(f"Total DKT variables processed: {total_dkt_variables}")

    for key in results_dict:
        print(f"Variables processed for {key}: {len(results_dict[key])}")
        if results_dict[key]:  # Only save if there are results to save
            results_df = pd.DataFrame(results_dict[key])
            file_name = f"/Users/Cabria/PycharmProjects/pythonProject/RDS_results_{key}.csv"
            results_df.to_csv(file_name, index=False)
            apply_fdr(results_df, key)

    return results_dict

# Applying False Discovery Rate (FDR) Correction to P-values
def apply_fdr(results_df, key):
    p_values = results_df['p-value'].values
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    results_df['p-value_corrected'] = pvals_corrected
    results_df['Significant After FDR'] = reject
    file_name = f"/Users/Cabria/PycharmProjects/pythonProject/RDS_results_{key}_FDR.csv"
    results_df.to_csv(file_name, index=False)
    return results_df

head_motion_columns = [
    '24419-2.0', '24438-2.0', '24439-2.0', '24440-2.0', '24441-2.0',
    '24442-2.0', '24443-2.0', '25741-2.0', '24450-2.0', '24451-2.0',
    '24452-2.0', '24453-2.0', '24454-2.0', '24455-2.0', '24444-2.0',
    '24445-2.0', '24446-2.0', '24447-2.0', '24448-2.0', '24449-2.0',
    '25742-2.0'
]

results_dict = perform_analysis_and_save(df_main, head_motion_columns)
print("Results and FDR-corrected results saved for both models")

# Calculate Relative Change of Coefficients (with absolute values) and Save to CSV
def calculate_and_save_delta_abs():
    base_path = "/Users/Cabria/PycharmProjects/pythonProject/"
    file_name_template = "RDS_results_{}_head_motion_FDR.csv"

    delta_list = []
    opposite_signs_dict = {'T1_head_motion': 0, 'rfMRI_head_motion': 0, 'all_head_motion': 0}

    for include_head_motion in ['T1', 'rfMRI', 'all']:
        file_with_hm = base_path + file_name_template.format(include_head_motion)
        file_no_hm = base_path + "RDS_results_no_head_motion_FDR.csv"

        if not os.path.exists(file_with_hm) or not os.path.exists(file_no_hm):
            print(f"File not found: {file_with_hm} or {file_no_hm}")
            continue

        df_with_hm = pd.read_csv(file_with_hm)
        df_no_hm = pd.read_csv(file_no_hm)

        print(f"Columns in df_with_hm: {df_with_hm.columns.tolist()}")
        print(f"Columns in df_no_hm: {df_no_hm.columns.tolist()}")

        if 'Coefficient' not in df_with_hm.columns or 'Coefficient' not in df_no_hm.columns:
            raise KeyError("'Coefficient' column is missing in one of the dataframes.")

        for variable in df_with_hm['Variable']:
            with_hm_coef = df_with_hm.loc[df_with_hm['Variable'] == variable, 'Coefficient'].values[0]
            without_hm_coef = df_no_hm.loc[df_no_hm['Variable'] == variable, 'Coefficient'].values[0] if variable in df_no_hm['Variable'].values else np.nan

            # Check for opposite signs after delta calculation
            if np.sign(with_hm_coef) != np.sign(without_hm_coef):
                print(f"Opposite signs detected for variable {variable} in model {include_head_motion}.")
                opposite_signs_dict[f"{include_head_motion}_head_motion"] += 1
                continue  # Skip this variable if coefficients have opposite signs

            if without_hm_coef != 0 and not np.isnan(without_hm_coef):
                delta = (abs(with_hm_coef) - abs(without_hm_coef)) / abs(with_hm_coef)
            else:
                delta = np.nan
            delta_list.append({
                'Variable': variable,
                'Coefficient_with_head_motion': with_hm_coef,
                'Coefficient_without_head_motion': without_hm_coef,
                'Delta_Coefficient': delta,
                'Dataset': include_head_motion + ' motion'
            })

    delta_df = pd.DataFrame(delta_list)
    delta_df.to_csv(base_path + "RDS_Delta_Coefficients_abs.csv", index=False)

    # Save the count of opposite sign variables
    opposite_signs_df = pd.DataFrame(list(opposite_signs_dict.items()), columns=['Model', 'Opposite_Signs_Count'])
    opposite_signs_df.to_csv("/Users/Cabria/PycharmProjects/pythonProject/RDS_Opposite_Signs_Count.csv", index=False)

    print(f"Delta coefficients with abs saved.")

calculate_and_save_delta_abs()
print("Delta coefficients with abs saved.")

# Visualization
def visualize_delta_coefficients_abs():
    base_path = "/Users/Cabria/PycharmProjects/pythonProject/"
    delta_file_path = base_path + "RDS_Delta_Coefficients_abs.csv"

    if not os.path.exists(delta_file_path):
        print(f"File not found: {delta_file_path}")
        return

    delta_df = pd.read_csv(delta_file_path)

    plt.figure(figsize=(12, 8))
    sns.swarmplot(x='Dataset', y='Delta_Coefficient', data=delta_df, palette=['blue', 'red', 'orange'])
    plt.title('Depression (RDS)-4 Relative ∆ Coefficients Across Models')
    plt.xlabel('Which head motion variable is controlled for')
    plt.ylabel('∆ Coefficient (abs(model with motion control) - abs(model without motion control)) / abs(model with motion control)', fontsize=9)
    plt.show()
    plt.ylim(-10, 3)

visualize_delta_coefficients_abs()




# N-12 Relative Change with Absolute Value
# Importing packages
import pandas as pd
import numpy as np
from scipy.stats import zscore
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import os
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Helper Functions
def is_binary_or_categorical(column, df):
    return df[column].dtype == 'object' or df[column].nunique() == 2

def remove_outliers(df, columns_for_outlier_detection, std_threshold=6):
    df_outlier_detection = df[columns_for_outlier_detection]
    mean = df_outlier_detection.mean()
    std = df_outlier_detection.std()
    outliers = (df_outlier_detection > mean + std_threshold * std) | (df_outlier_detection < mean - std_threshold * std)
    df_clean = df[~outliers.any(axis=1)]

    num_excluded = df.shape[0] - df_clean.shape[0]
    num_remaining = df_clean.shape[0]

    print(f"Excluded {num_excluded} subjects due to outliers.")
    print(f"Remaining subjects: {num_remaining}")

    return df_clean

# Preparing and Loading N-12 Data for Precise Linear Regression
df_main = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset3.csv", low_memory=False)
df_main = df_main.dropna()

# N-12 calculation
n_12_columns = ['1920-2.0', '1930-2.0', '1940-2.0', '1950-2.0', '1960-2.0',
                '1970-2.0', '1980-2.0', '1990-2.0', '2000-2.0', '2010-2.0',
                '2020-2.0', '2030-2.0']
df_main['N_12'] = df_main[n_12_columns].sum(axis=1)

# Specify columns for outlier detection
columns_for_outlier_detection = [col for col in df_main.columns if not is_binary_or_categorical(col, df_main) and col not in ['N_12']]
df_main = remove_outliers(df_main, columns_for_outlier_detection, std_threshold=6)

# Apply z-score normalization after outlier removal
df_main = df_main.apply(zscore)

# Ensure all required columns are present
required_columns = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0', '25741-2.0', '27143-2.0', '27327-2.0']
missing_columns = [col for col in required_columns if col not in df_main.columns]
if missing_columns:
    raise KeyError(f"The following required columns are missing: {missing_columns}")

# Performing Linear Regression Analysis and Saving Results
def perform_analysis_and_save(df_main, head_motion_columns):
    results_dict = {'T1_head_motion': [], 'rfMRI_head_motion': [], 'all_head_motion': [], 'no_head_motion': []}
    start_index = df_main.columns.get_loc('27143-2.0')
    end_index = df_main.columns.get_loc('27327-2.0')

    total_dkt_variables = 0
    for dkt_var in df_main.columns[start_index:end_index + 1]:
        total_dkt_variables += 1
        df_main['DKT'] = df_main[dkt_var]

        if df_main['DKT'].isnull().any() or np.isinf(df_main['DKT']).any():
            print(f"NaNs or infinite values detected in DKT calculations for variable {dkt_var}. Skipping.")
            continue

        for include_head_motion in ['T1', 'rfMRI', 'all', 'none']:
            covariates = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0']
            if include_head_motion == 'T1':
                covariates.append('24419-2.0')
            elif include_head_motion == 'rfMRI':
                covariates.append('25741-2.0')
            elif include_head_motion == 'all':
                covariates.extend(head_motion_columns)

            all_vars = ['DKT'] + covariates
            X = df_main[all_vars]

            if X.isnull().any().any() or np.isinf(X).any().any():
                print(f"NaNs or infinite values detected in X for variable {dkt_var} and model {include_head_motion}. Skipping.")
                continue

            Y = df_main['N_12']
            model = sm.OLS(Y, X).fit()
            result = {
                'Variable': dkt_var,
                'Coefficient': model.params['DKT'],
                'p-value': model.pvalues['DKT']
            }

            key = f"{include_head_motion}_head_motion" if include_head_motion != 'none' else 'no_head_motion'
            results_dict[key].append(result)

    # Output debug information
    print(f"Total DKT variables expected: 186")
    print(f"Total DKT variables processed: {total_dkt_variables}")

    for key in results_dict:
        print(f"Variables processed for {key}: {len(results_dict[key])}")
        if results_dict[key]:  # Only save if there are results to save
            results_df = pd.DataFrame(results_dict[key])
            file_name = f"/Users/Cabria/PycharmProjects/pythonProject/N_12_results_{key}.csv"
            results_df.to_csv(file_name, index=False)
            apply_fdr(results_df, key)

    return results_dict

# Applying False Discovery Rate (FDR) Correction to P-values
def apply_fdr(results_df, key):
    p_values = results_df['p-value'].values
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    results_df['p-value_corrected'] = pvals_corrected
    results_df['Significant After FDR'] = reject
    file_name = f"/Users/Cabria/PycharmProjects/pythonProject/N_12_results_{key}_FDR.csv"
    results_df.to_csv(file_name, index=False)
    return results_df

head_motion_columns = [
    '24419-2.0', '24438-2.0', '24439-2.0', '24440-2.0', '24441-2.0',
    '24442-2.0', '24443-2.0', '25741-2.0', '24450-2.0', '24451-2.0',
    '24452-2.0', '24453-2.0', '24454-2.0', '24455-2.0', '24444-2.0',
    '24445-2.0', '24446-2.0', '24447-2.0', '24448-2.0', '24449-2.0',
    '25742-2.0'
]

results_dict = perform_analysis_and_save(df_main, head_motion_columns)
print("Results and FDR-corrected results saved for both models")

# Calculate Relative Change of Coefficients (with absolute values) and Save to CSV
def calculate_and_save_delta_abs():
    base_path = "/Users/Cabria/PycharmProjects/pythonProject/"
    file_name_template = "N_12_results_{}_head_motion_FDR.csv"

    delta_list = []
    opposite_signs_dict = {'T1_head_motion': 0, 'rfMRI_head_motion': 0, 'all_head_motion': 0}

    for include_head_motion in ['T1', 'rfMRI', 'all']:
        file_with_hm = base_path + file_name_template.format(include_head_motion)
        file_no_hm = base_path + "N_12_results_no_head_motion_FDR.csv"

        if not os.path.exists(file_with_hm) or not os.path.exists(file_no_hm):
            print(f"File not found: {file_with_hm} or {file_no_hm}")
            continue

        df_with_hm = pd.read_csv(file_with_hm)
        df_no_hm = pd.read_csv(file_no_hm)

        print(f"Columns in df_with_hm: {df_with_hm.columns.tolist()}")
        print(f"Columns in df_no_hm: {df_no_hm.columns.tolist()}")

        if 'Coefficient' not in df_with_hm.columns or 'Coefficient' not in df_no_hm.columns:
            raise KeyError("'Coefficient' column is missing in one of the dataframes.")

        for variable in df_with_hm['Variable']:
            with_hm_coef = df_with_hm.loc[df_with_hm['Variable'] == variable, 'Coefficient'].values[0]
            without_hm_coef = df_no_hm.loc[df_no_hm['Variable'] == variable, 'Coefficient'].values[0] if variable in df_no_hm['Variable'].values else np.nan

            # Check for opposite signs after delta calculation
            if np.sign(with_hm_coef) != np.sign(without_hm_coef):
                print(f"Opposite signs detected for variable {variable} in model {include_head_motion}.")
                opposite_signs_dict[f"{include_head_motion}_head_motion"] += 1
                continue  # Skip this variable if coefficients have opposite signs

            if without_hm_coef != 0 and not np.isnan(without_hm_coef):
                delta = (abs(with_hm_coef) - abs(without_hm_coef)) / abs(with_hm_coef)
            else:
                delta = np.nan
            delta_list.append({
                'Variable': variable,
                'Coefficient_with_head_motion': with_hm_coef,
                'Coefficient_without_head_motion': without_hm_coef,
                'Delta_Coefficient': delta,
                'Dataset': include_head_motion + ' motion'
            })

    delta_df = pd.DataFrame(delta_list)
    delta_df.to_csv(base_path + "N_12_Delta_Coefficients_abs.csv", index=False)

    # Save the count of opposite sign variables
    opposite_signs_df = pd.DataFrame(list(opposite_signs_dict.items()), columns=['Model', 'Opposite_Signs_Count'])
    opposite_signs_df.to_csv("/Users/Cabria/PycharmProjects/pythonProject/N_12_Opposite_Signs_Count.csv", index=False)

    print(f"Delta coefficients with abs saved.")

calculate_and_save_delta_abs()
print("Delta coefficients with abs saved.")

# Visualization
def visualize_delta_coefficients_abs():
    base_path = "/Users/Cabria/PycharmProjects/pythonProject/"
    delta_file_path = base_path + "N_12_Delta_Coefficients_abs.csv"

    if not os.path.exists(delta_file_path):
        print(f"File not found: {delta_file_path}")
        return

    delta_df = pd.read_csv(delta_file_path)

    plt.figure(figsize=(12, 8))
    sns.swarmplot(x='Dataset', y='Delta_Coefficient', data=delta_df, palette=['blue', 'red', 'orange'])
    plt.title('Neuroticism (N-12) Relative ∆ Coefficients Across Models')
    plt.xlabel('Which head motion variable is controlled for')
    plt.ylabel('∆ Coefficient (abs(model with motion control) - abs(model without motion control)) / abs(model with motion control)', fontsize=9)
    plt.show()

visualize_delta_coefficients_abs()





# IQ Relative Change with Absolute Value
# Importing packages
import pandas as pd
import numpy as np
from scipy.stats import zscore
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import os
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Helper Functions
def is_binary_or_categorical(column, df):
    return df[column].dtype == 'object' or df[column].nunique() == 2

def remove_outliers(df, columns_for_outlier_detection, std_threshold=6):
    df_outlier_detection = df[columns_for_outlier_detection]
    mean = df_outlier_detection.mean()
    std = df_outlier_detection.std()
    outliers = (df_outlier_detection > mean + std_threshold * std) | (df_outlier_detection < mean - std_threshold * std)
    df_clean = df[~outliers.any(axis=1)]

    num_excluded = df.shape[0] - df_clean.shape[0]
    num_remaining = df_clean.shape[0]

    print(f"Excluded {num_excluded} subjects due to outliers.")
    print(f"Remaining subjects: {num_remaining}")

    return df_clean

# Preparing and Loading IQ Data for Precise Linear Regression
df_main = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset3.csv", low_memory=False)
df_main = df_main.dropna()

# IQ calculation
df_main['IQ'] = df_main['20016-2.0']

# Specify columns for outlier detection
columns_for_outlier_detection = [col for col in df_main.columns if not is_binary_or_categorical(col, df_main) and col not in ['IQ']]
df_main = remove_outliers(df_main, columns_for_outlier_detection, std_threshold=6)

# Apply z-score normalization after outlier removal
df_main = df_main.apply(zscore)

# Ensure all required columns are present
required_columns = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0', '25741-2.0', '27143-2.0', '27327-2.0']
missing_columns = [col for col in required_columns if col not in df_main.columns]
if missing_columns:
    raise KeyError(f"The following required columns are missing: {missing_columns}")

# Performing Linear Regression Analysis and Saving Results
def perform_analysis_and_save(df_main, head_motion_columns):
    results_dict = {'T1_head_motion': [], 'rfMRI_head_motion': [], 'all_head_motion': [], 'no_head_motion': []}
    start_index = df_main.columns.get_loc('27143-2.0')
    end_index = df_main.columns.get_loc('27327-2.0')

    total_dkt_variables = 0
    for dkt_var in df_main.columns[start_index:end_index + 1]:
        total_dkt_variables += 1
        df_main['DKT'] = df_main[dkt_var]

        if df_main['DKT'].isnull().any() or np.isinf(df_main['DKT']).any():
            print(f"NaNs or infinite values detected in DKT calculations for variable {dkt_var}. Skipping.")
            continue

        for include_head_motion in ['T1', 'rfMRI', 'all', 'none']:
            covariates = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0']
            if include_head_motion == 'T1':
                covariates.append('24419-2.0')
            elif include_head_motion == 'rfMRI':
                covariates.append('25741-2.0')
            elif include_head_motion == 'all':
                covariates.extend(head_motion_columns)

            all_vars = ['DKT'] + covariates
            X = df_main[all_vars]

            if X.isnull().any().any() or np.isinf(X).any().any():
                print(f"NaNs or infinite values detected in X for variable {dkt_var} and model {include_head_motion}. Skipping.")
                continue

            Y = df_main['IQ']
            model = sm.OLS(Y, X).fit()
            result = {
                'Variable': dkt_var,
                'Coefficient': model.params['DKT'],
                'p-value': model.pvalues['DKT']
            }

            key = f"{include_head_motion}_head_motion" if include_head_motion != 'none' else 'no_head_motion'
            results_dict[key].append(result)

    # Output debug information
    print(f"Total DKT variables expected: 186")
    print(f"Total DKT variables processed: {total_dkt_variables}")

    for key in results_dict:
        print(f"Variables processed for {key}: {len(results_dict[key])}")
        if results_dict[key]:  # Only save if there are results to save
            results_df = pd.DataFrame(results_dict[key])
            file_name = f"/Users/Cabria/PycharmProjects/pythonProject/IQ_results_{key}.csv"
            results_df.to_csv(file_name, index=False)
            apply_fdr(results_df, key)

    return results_dict

# Applying False Discovery Rate (FDR) Correction to P-values
def apply_fdr(results_df, key):
    p_values = results_df['p-value'].values
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    results_df['p-value_corrected'] = pvals_corrected
    results_df['Significant After FDR'] = reject
    file_name = f"/Users/Cabria/PycharmProjects/pythonProject/IQ_results_{key}_FDR.csv"
    results_df.to_csv(file_name, index=False)
    return results_df

head_motion_columns = [
    '24419-2.0', '24438-2.0', '24439-2.0', '24440-2.0', '24441-2.0',
    '24442-2.0', '24443-2.0', '25741-2.0', '24450-2.0', '24451-2.0',
    '24452-2.0', '24453-2.0', '24454-2.0', '24455-2.0', '24444-2.0',
    '24445-2.0', '24446-2.0', '24447-2.0', '24448-2.0', '24449-2.0',
    '25742-2.0'
]

results_dict = perform_analysis_and_save(df_main, head_motion_columns)
print("Results and FDR-corrected results saved for both models")

# Calculate Relative Change of Coefficients (with absolute values) and Save to CSV
def calculate_and_save_delta_abs():
    base_path = "/Users/Cabria/PycharmProjects/pythonProject/"
    file_name_template = "IQ_results_{}_head_motion_FDR.csv"

    delta_list = []
    opposite_signs_dict = {'T1_head_motion': 0, 'rfMRI_head_motion': 0, 'all_head_motion': 0}

    for include_head_motion in ['T1', 'rfMRI', 'all']:
        file_with_hm = base_path + file_name_template.format(include_head_motion)
        file_no_hm = base_path + "IQ_results_no_head_motion_FDR.csv"

        if not os.path.exists(file_with_hm) or not os.path.exists(file_no_hm):
            print(f"File not found: {file_with_hm} or {file_no_hm}")
            continue

        df_with_hm = pd.read_csv(file_with_hm)
        df_no_hm = pd.read_csv(file_no_hm)

        print(f"Columns in df_with_hm: {df_with_hm.columns.tolist()}")
        print(f"Columns in df_no_hm: {df_no_hm.columns.tolist()}")

        if 'Coefficient' not in df_with_hm.columns or 'Coefficient' not in df_no_hm.columns:
            raise KeyError("'Coefficient' column is missing in one of the dataframes.")

        for variable in df_with_hm['Variable']:
            with_hm_coef = df_with_hm.loc[df_with_hm['Variable'] == variable, 'Coefficient'].values[0]
            without_hm_coef = df_no_hm.loc[df_no_hm['Variable'] == variable, 'Coefficient'].values[0] if variable in df_no_hm['Variable'].values else np.nan

            # Check for opposite signs after delta calculation
            if np.sign(with_hm_coef) != np.sign(without_hm_coef):
                print(f"Opposite signs detected for variable {variable} in model {include_head_motion}.")
                opposite_signs_dict[f"{include_head_motion}_head_motion"] += 1
                continue  # Skip this variable if coefficients have opposite signs

            if without_hm_coef != 0 and not np.isnan(without_hm_coef):
                delta = (abs(with_hm_coef) - abs(without_hm_coef)) / abs(with_hm_coef)
            else:
                delta = np.nan
            delta_list.append({
                'Variable': variable,
                'Coefficient_with_head_motion': with_hm_coef,
                'Coefficient_without_head_motion': without_hm_coef,
                'Delta_Coefficient': delta,
                'Dataset': include_head_motion + ' motion'
            })

    delta_df = pd.DataFrame(delta_list)
    delta_df.to_csv(base_path + "IQ_Delta_Coefficients_abs.csv", index=False)

    # Save the count of opposite sign variables
    opposite_signs_df = pd.DataFrame(list(opposite_signs_dict.items()), columns=['Model', 'Opposite_Signs_Count'])
    opposite_signs_df.to_csv("/Users/Cabria/PycharmProjects/pythonProject/IQ_Opposite_Signs_Count.csv", index=False)

    print(f"Delta coefficients with abs saved.")

calculate_and_save_delta_abs()
print("Delta coefficients with abs saved.")

# Visualization
def visualize_delta_coefficients_abs():
    base_path = "/Users/Cabria/PycharmProjects/pythonProject/"
    delta_file_path = base_path + "IQ_Delta_Coefficients_abs.csv"

    if not os.path.exists(delta_file_path):
        print(f"File not found: {delta_file_path}")
        return

    delta_df = pd.read_csv(delta_file_path)

    plt.figure(figsize=(12, 8))
    sns.swarmplot(x='Dataset', y='Delta_Coefficient', data=delta_df, palette=['blue', 'red', 'orange'])
    plt.title('IQ Relative ∆ Coefficients Across Models')
    plt.xlabel('Which head motion variable is controlled for')
    plt.ylabel('∆ Coefficient (abs(model with motion control) - abs(model without motion control)) / abs(model with motion control)', fontsize=9)
    plt.show()

visualize_delta_coefficients_abs()




# RT Relative Change with Absolute Value
# Importing packages
import pandas as pd
import numpy as np
from scipy.stats import zscore
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import os
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Helper Functions
def is_binary_or_categorical(column, df):
    return df[column].dtype == 'object' or df[column].nunique() == 2

def remove_outliers(df, columns_for_outlier_detection, std_threshold=6):
    df_outlier_detection = df[columns_for_outlier_detection]
    mean = df_outlier_detection.mean()
    std = df_outlier_detection.std()
    outliers = (df_outlier_detection > mean + std_threshold * std) | (df_outlier_detection < mean - std_threshold * std)
    df_clean = df[~outliers.any(axis=1)]

    num_excluded = df.shape[0] - df_clean.shape[0]
    num_remaining = df_clean.shape[0]

    print(f"Excluded {num_excluded} subjects due to outliers.")
    print(f"Remaining subjects: {num_remaining}")

    return df_clean

# Preparing and Loading RT Data for Precise Linear Regression
df_main = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset3.csv", low_memory=False)
df_main = df_main.dropna()

# RT calculation
df_main['RT'] = df_main['20023-2.0']

# Specify columns for outlier detection
columns_for_outlier_detection = [col for col in df_main.columns if not is_binary_or_categorical(col, df_main) and col not in ['RT']]
df_main = remove_outliers(df_main, columns_for_outlier_detection, std_threshold=6)

# Apply z-score normalization after outlier removal
df_main = df_main.apply(zscore)

# Ensure all required columns are present
required_columns = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0', '25741-2.0', '27143-2.0', '27327-2.0']
missing_columns = [col for col in required_columns if col not in df_main.columns]
if missing_columns:
    raise KeyError(f"The following required columns are missing: {missing_columns}")

# Performing Linear Regression Analysis and Saving Results
def perform_analysis_and_save(df_main, head_motion_columns):
    results_dict = {'T1_head_motion': [], 'rfMRI_head_motion': [], 'all_head_motion': [], 'no_head_motion': []}
    start_index = df_main.columns.get_loc('27143-2.0')
    end_index = df_main.columns.get_loc('27327-2.0')

    total_dkt_variables = 0
    for dkt_var in df_main.columns[start_index:end_index + 1]:
        total_dkt_variables += 1
        df_main['DKT'] = df_main[dkt_var]

        if df_main['DKT'].isnull().any() or np.isinf(df_main['DKT']).any():
            print(f"NaNs or infinite values detected in DKT calculations for variable {dkt_var}. Skipping.")
            continue

        for include_head_motion in ['T1', 'rfMRI', 'all', 'none']:
            covariates = ['21003-2.0', '31-0.0', '54-2.0', '26521-2.0']
            if include_head_motion == 'T1':
                covariates.append('24419-2.0')
            elif include_head_motion == 'rfMRI':
                covariates.append('25741-2.0')
            elif include_head_motion == 'all':
                covariates.extend(head_motion_columns)

            all_vars = ['DKT'] + covariates
            X = df_main[all_vars]

            if X.isnull().any().any() or np.isinf(X).any().any():
                print(f"NaNs or infinite values detected in X for variable {dkt_var} and model {include_head_motion}. Skipping.")
                continue

            Y = df_main['RT']
            model = sm.OLS(Y, X).fit()
            result = {
                'Variable': dkt_var,
                'Coefficient': model.params['DKT'],
                'p-value': model.pvalues['DKT']
            }

            key = f"{include_head_motion}_head_motion" if include_head_motion != 'none' else 'no_head_motion'
            results_dict[key].append(result)

    # Output debug information
    print(f"Total DKT variables expected: 186")
    print(f"Total DKT variables processed: {total_dkt_variables}")

    for key in results_dict:
        print(f"Variables processed for {key}: {len(results_dict[key])}")
        if results_dict[key]:  # Only save if there are results to save
            results_df = pd.DataFrame(results_dict[key])
            file_name = f"/Users/Cabria/PycharmProjects/pythonProject/RT_results_{key}.csv"
            results_df.to_csv(file_name, index=False)
            apply_fdr(results_df, key)

    return results_dict

# Applying False Discovery Rate (FDR) Correction to P-values
def apply_fdr(results_df, key):
    p_values = results_df['p-value'].values
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    results_df['p-value_corrected'] = pvals_corrected
    results_df['Significant After FDR'] = reject
    file_name = f"/Users/Cabria/PycharmProjects/pythonProject/RT_results_{key}_FDR.csv"
    results_df.to_csv(file_name, index=False)
    return results_df

head_motion_columns = [
    '24419-2.0', '24438-2.0', '24439-2.0', '24440-2.0', '24441-2.0',
    '24442-2.0', '24443-2.0', '25741-2.0', '24450-2.0', '24451-2.0',
    '24452-2.0', '24453-2.0', '24454-2.0', '24455-2.0', '24444-2.0',
    '24445-2.0', '24446-2.0', '24447-2.0', '24448-2.0', '24449-2.0',
    '25742-2.0'
]

results_dict = perform_analysis_and_save(df_main, head_motion_columns)
print("Results and FDR-corrected results saved for both models")

# Calculate Relative Change of Coefficients (with absolute values) and Save to CSV
def calculate_and_save_delta_abs():
    base_path = "/Users/Cabria/PycharmProjects/pythonProject/"
    file_name_template = "RT_results_{}_head_motion_FDR.csv"

    delta_list = []
    opposite_signs_dict = {'T1_head_motion': 0, 'rfMRI_head_motion': 0, 'all_head_motion': 0}

    for include_head_motion in ['T1', 'rfMRI', 'all']:
        file_with_hm = base_path + file_name_template.format(include_head_motion)
        file_no_hm = base_path + "RT_results_no_head_motion_FDR.csv"

        if not os.path.exists(file_with_hm) or not os.path.exists(file_no_hm):
            print(f"File not found: {file_with_hm} or {file_no_hm}")
            continue

        df_with_hm = pd.read_csv(file_with_hm)
        df_no_hm = pd.read_csv(file_no_hm)

        print(f"Columns in df_with_hm: {df_with_hm.columns.tolist()}")
        print(f"Columns in df_no_hm: {df_no_hm.columns.tolist()}")

        if 'Coefficient' not in df_with_hm.columns or 'Coefficient' not in df_no_hm.columns:
            raise KeyError("'Coefficient' column is missing in one of the dataframes.")

        for variable in df_with_hm['Variable']:
            with_hm_coef = df_with_hm.loc[df_with_hm['Variable'] == variable, 'Coefficient'].values[0]
            without_hm_coef = df_no_hm.loc[df_no_hm['Variable'] == variable, 'Coefficient'].values[0] if variable in df_no_hm['Variable'].values else np.nan

            # Check for opposite signs after delta calculation
            if np.sign(with_hm_coef) != np.sign(without_hm_coef):
                print(f"Opposite signs detected for variable {variable} in model {include_head_motion}.")
                opposite_signs_dict[f"{include_head_motion}_head_motion"] += 1
                continue  # Skip this variable if coefficients have opposite signs

            if without_hm_coef != 0 and not np.isnan(without_hm_coef):
                delta = (abs(with_hm_coef) - abs(without_hm_coef)) / abs(with_hm_coef)
            else:
                delta = np.nan
            delta_list.append({
                'Variable': variable,
                'Coefficient_with_head_motion': with_hm_coef,
                'Coefficient_without_head_motion': without_hm_coef,
                'Delta_Coefficient': delta,
                'Dataset': include_head_motion + ' motion'
            })

    delta_df = pd.DataFrame(delta_list)
    delta_df.to_csv(base_path + "RT_Delta_Coefficients_abs.csv", index=False)

    # Save the count of opposite sign variables
    opposite_signs_df = pd.DataFrame(list(opposite_signs_dict.items()), columns=['Model', 'Opposite_Signs_Count'])
    opposite_signs_df.to_csv("/Users/Cabria/PycharmProjects/pythonProject/RT_Opposite_Signs_Count.csv", index=False)

    print(f"Delta coefficients with abs saved.")

calculate_and_save_delta_abs()
print("Delta coefficients with abs saved.")

# Visualization
def visualize_delta_coefficients_abs():
    base_path = "/Users/Cabria/PycharmProjects/pythonProject/"
    delta_file_path = base_path + "RT_Delta_Coefficients_abs.csv"

    if not os.path.exists(delta_file_path):
        print(f"File not found: {delta_file_path}")
        return

    delta_df = pd.read_csv(delta_file_path)

    plt.figure(figsize=(12, 8))
    sns.swarmplot(x='Dataset', y='Delta_Coefficient', data=delta_df, palette=['blue', 'red', 'orange'])
    plt.title('Reaction Time (RT) Relative ∆ Coefficients Across Models')
    plt.xlabel('Which head motion variable is controlled for')
    plt.ylabel('∆ Coefficient (abs(model with motion control) - abs(model without motion control)) / abs(model with motion control)', fontsize=9)
    plt.show()

visualize_delta_coefficients_abs()





try:
    import matplotlib
    matplotlib.use('TkAgg')  # Try TkAgg instead of Qt5Agg
    import matplotlib.pyplot as plt

    print("Current backend:", matplotlib.get_backend())
    plt.plot([1, 2, 3, 4])
    plt.title("Test Plot")
    plt.show()

except KeyboardInterrupt:
    print("Process interrupted. Performing cleanup...")
    # Add any cleanup logic here
    # Then exit gracefully








#Testing correlation of T1 vs rfMRI
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('TkAgg')


df_main = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset3.csv", low_memory=False)
df_main = df_main.dropna()

# Define the column names for T1 and rfMRI motion metrics
t1_motion_column = '24419-2.0'  # T1 Structural Motion
rfmri_motion_column = '25741-2.0'  # rfMRI Head Motion

# Extract the relevant columns
t1_motion = df_main[t1_motion_column]
rfmri_motion = df_main[rfmri_motion_column]

# Calculate the Pearson correlation between T1 structural motion and rfMRI head motion
correlation, p_value = pearsonr(t1_motion, rfmri_motion)

# Print correlation and p-value
print(f"Pearson correlation between T1 structural motion and rfMRI head motion: {correlation}")
print(f"P-value: {p_value}")

# Generate a scatterplot to visualize the relationship
plt.figure(figsize=(8, 6))
sns.scatterplot(x=t1_motion, y=rfmri_motion)
plt.title('Scatterplot of T1 Structural Motion vs rfMRI Motion')
plt.xlabel('T1 Structural Motion (24419-2.0)')
plt.ylabel('rfMRI Head Motion (25741-2.0)')
plt.grid(True)
plt.show()




#Lexi's code for propensity matching based on age, sex, and head motion
import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')


# function to match case and control (perfect matching)
def match(df_code, df_health, df_all):
    df_code = pd.merge(df_code, df_all, how='inner', on='eid')
    df_m = pd.merge(df_code, df_health, how='inner', on='sex')
    df_m['diff_age'] = abs(df_m['age_x'] - df_m['age_y'])
    df_m['diff_rfMRI'] = abs(df_m['rfMRI_x'] - df_m['rfMRI_y'])
    df_m = df_m.fillna(100)
    df_m = df_m.sort_values(by=['sex', 'age_x', 'diff_age', 'diff_rfMRI', 'eid_x', 'eid_y'])
    df_m = df_m.loc[:, ['eid_x', 'eid_y', 'sex', 'diff_age', 'diff_rfMRI']]

    df1 = df_m.drop_duplicates(subset=['eid_x'])
    df2 = df_m.drop_duplicates(subset=['eid_x']).drop_duplicates(subset=['eid_y'])
    df = df_m
    while len(df1) != len(df2):
        df_rep = df1.duplicated(subset=['eid_y'])
        df_rep = df1[df_rep]
        df = df.append(df_rep)
        df = df.drop_duplicates(subset=['eid_x', 'eid_y'], keep=False)
        df1 = df.drop_duplicates(subset=['eid_x'])
        df2 = df.drop_duplicates(subset=['eid_x']).drop_duplicates(subset=['eid_y'])
    return df1


code_list = ['F0', 'F10', 'F17', 'F32', 'F41', 'G2', 'G40', 'G43', 'G45', 'G47', 'G55', 'G56', 'G57', 'G62', 'G8',
             'G93', 'G35_37']
df_all = pd.read_csv('/Users/yuanyuanxiaowang/Desktop/RA/wapiaawphenos_universe_ses-01_anatfunc_subs.csv',
                     index_col=None, header=0, usecols=[0, 1, 10, 14], names=['eid', 'sex', 'age', 'rfMRI'])
df_health = pd.read_csv('/Users/yuanyuanxiaowang/PycharmProjects/pythonProject/WAPIAW/Gigascience/health_uni.csv',
                        index_col=None, header=None, names=['eid'])
df_health = pd.merge(df_health, df_all, how='inner', on='eid')
root_path = '/Users/yuanyuanxiaowang/PycharmProjects/pythonProject/WAPIAW/Gigascience/patient_eid_unique_resample'
for code in code_list:
    filename = code + '_unique'
    path = os.path.join(root_path, filename)
    df_code = pd.read_csv(path, names=['eid'])
    print(len(df_code))
    df_match = match(df_code, df_health, df_all)
    df_match = df_match['eid_y']
    print(len(df_match))
    name = 'match' + '_' + filename
    df_code = df_code
    df_match.to_csv(os.path.join('/Users/yuanyuanxiaowang/PycharmProjects/pythonProject/WAPIAW/Gigascience/match_unique', name), index=False, header=False)
    df1 = pd.read_csv(os.path.join(root_path, filename), header=None)
    df2 = pd.read_csv(os.path.join('/Users/yuanyuanxiaowang/PycharmProjects/pythonProject/WAPIAW/Gigascience/match_unique', name), header=None)
    df = pd.concat([df1, df2], axis=0)
    df.to_csv(os.path.join("/Users/yuanyuanxiaowang/PycharmProjects/pythonProject/WAPIAW/Gigascience/combined_match_unique", filename), index=False, header=False)
    print(len(df))
    print(filename)


#This one works better
import pandas as pd

def calculate_mean_difference(file_path_no_hm, file_path_with_hm):
    try:
        # Load the data
        df_no_hm = pd.read_csv('/Users/Cabria/PycharmProjects/pythonProject/RDS_results_no_head_motion_FDR.csv')
        df_with_hm = pd.read_csv('/Users/Cabria/PycharmProjects/pythonProject/RDS_results_rfMRI_head_motion_FDR.csv')

        # Print column names to verify
        print("Columns in no HM file:", df_no_hm.columns)
        print("Columns in with HM file:", df_with_hm.columns)

        # Merge dataframes on the 'Variable' column for comparison
        merged_df = pd.merge(df_no_hm, df_with_hm, on='Variable', suffixes=('_no_hm', '_with_hm'))

        # Calculate the difference in coefficients for each variable
        merged_df['Coefficient_Difference'] = merged_df['Coefficient_with_hm'] - merged_df['Coefficient_no_hm']

        # Calculate the mean of these differences
        mean_difference = merged_df['Coefficient_Difference'].mean()

        # Output the results
        print("Coefficient differences for all variables:")
        print(merged_df[['Variable', 'Coefficient_no_hm', 'Coefficient_with_hm', 'Coefficient_Difference']])
        print(f"Mean coefficient difference across all variables: {mean_difference}")

    except Exception as e:
        print(f"An error occurred: {e}")

# usage of the function with file paths
calculate_mean_difference(
    '/Users/Cabria/PycharmProjects/pythonProject/RDS_results_no_head_motion_FDR.csv',
    '/Users/Cabria/PycharmProjects/pythonProject/RDS_results_rfMRI_head_motion_FDR.csv'
)




#New and I hope more correct high vs low matrix (T1 and rfMRI)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib

# Switch to the Agg backend
matplotlib.use('Agg')

# Load datasets
subjects_df = pd.read_csv("C:/Users/Cabria/Downloads/Clean_SubjecsWithT1orRfMRIand_NO_GorFinICD10atBaseline.csv", low_memory=False)
df_main = pd.read_csv("/Users/Cabria/PycharmProjects/pythonProject/CombinedDataset3.csv", low_memory=False)
dkt_descriptions = pd.read_csv("C:/Users/Cabria/Downloads/DKT_descriptions_dictionary.csv")

# Step 1: Filter the main dataset to include only subjects present in subjects_df
filtered_df_main = pd.merge(subjects_df, df_main, on='eid', how='inner')
filtered_df_main = filtered_df_main.dropna()  # Remove rows with any missing values

# Print sample size after filtering and cleaning
print(f"Sample size after merging and cleaning: {filtered_df_main.shape[0]}")

# Step 2: Extract DKT variables/IDPs based on specified index range
start_index = df_main.columns.get_loc('27143-2.0')
end_index = df_main.columns.get_loc('27327-2.0')
dkt_variables = filtered_df_main.iloc[:, start_index:end_index + 1]

# Step 3: Use DKT descriptions to identify IDP types (thickness, volume, area)
thickness_columns = dkt_descriptions[dkt_descriptions['DKT Type'] == 'thickness']['DKT Variable'].tolist()
volume_columns = dkt_descriptions[dkt_descriptions['DKT Type'] == 'volume']['DKT Variable'].tolist()
area_columns = dkt_descriptions[dkt_descriptions['DKT Type'] == 'area']['DKT Variable'].tolist()

# Filter DKT variables to match the DKT descriptions
thickness_columns = [col for col in dkt_variables.columns if col in thickness_columns]
volume_columns = [col for col in dkt_variables.columns if col in volume_columns]
area_columns = [col for col in dkt_variables.columns if col in area_columns]

# Step 4: Stratify subjects based on the T1 and rfMRI head motion variables
filtered_df_main = filtered_df_main.copy()
filtered_df_main['T1_quartile'] = pd.qcut(filtered_df_main['24419-2.0'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
filtered_df_main['rfMRI_quartile'] = pd.qcut(filtered_df_main['25741-2.0'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# Define IDP types for analysis
idp_types = {
    'thickness': thickness_columns,
    'volume': volume_columns,
    'area': area_columns
}

# Define the updated titles for each IDP type
heatmap_titles = {
    'thickness': "Comparing Mean Cortical Brain Thickness (mm) by High vs Low T1 and rfMRI Head Motion",
    'area': "Comparing Mean Cortical Brain Area (mm²) by High vs Low T1 and rfMRI Head Motion",
    'volume': "Comparing Mean Cortical Brain Volume (mm³) by High vs Low T1 and rfMRI Head Motion"
}

# Step 5: Propensity score calculation with nearest neighbor matching
def nearest_neighbor_matching(target_df, comparison_df, tolerance=0.05):
    X_target = target_df[['propensity_score']].values
    X_comparison = comparison_df[['propensity_score']].values

    nn = NearestNeighbors(n_neighbors=1, radius=tolerance)
    nn.fit(X_comparison)
    distances, indices = nn.kneighbors(X_target)

    matched_target = target_df.reset_index(drop=True)
    matched_comparison = comparison_df.iloc[indices.flatten()].reset_index(drop=True)

    return matched_target, matched_comparison

# Step 6: Iterate through each IDP type and generate heatmaps
descriptive_table = []

for idp_type, columns in idp_types.items():
    print(f"Processing {idp_type} IDPs...")

    # Prepare the dataset for the current IDP type
    idp_df = filtered_df_main[['eid', 'T1_quartile', 'rfMRI_quartile', '21003-2.0', '31-0.0'] + columns].copy()
    idp_df = idp_df.dropna(subset=['T1_quartile', 'rfMRI_quartile'] + columns)

    heatmap_matrix = pd.DataFrame(
        index=['Q1', 'Q2', 'Q3', 'Q4'],
        columns=['Q1', 'Q2', 'Q3', 'Q4'],
        dtype=float
    )

    # Iterate through T1 and rfMRI quartiles
    for t1_q in ['Q1', 'Q2', 'Q3', 'Q4']:
        for rf_q in ['Q1', 'Q2', 'Q3', 'Q4']:
            target_df = idp_df[(idp_df['T1_quartile'] == t1_q) & (idp_df['rfMRI_quartile'] == rf_q)]
            comparison_df = idp_df[(idp_df['T1_quartile'] != t1_q) | (idp_df['rfMRI_quartile'] != rf_q)]

            # Ensure sufficient samples
            if len(target_df) < 2 or len(comparison_df) < 2:
                heatmap_matrix.loc[t1_q, rf_q] = np.nan
                continue

            # Calculate propensity scores for matching
            X = idp_df[['21003-2.0', '31-0.0']]
            y = (idp_df['T1_quartile'] == t1_q).astype(int)

            model = LogisticRegression(solver='liblinear')
            model.fit(X, y)
            idp_df['propensity_score'] = model.predict_proba(X)[:, 1]

            target_df['propensity_score'] = idp_df.loc[target_df.index, 'propensity_score']
            comparison_df['propensity_score'] = idp_df.loc[comparison_df.index, 'propensity_score']

            # Perform nearest neighbor matching
            matched_target, matched_comparison = nearest_neighbor_matching(target_df, comparison_df)

            # Calculate mean difference in IDP variables
            mean_diff = (matched_target[columns].mean() - matched_comparison[columns].mean()).mean()
            heatmap_matrix.loc[t1_q, rf_q] = mean_diff

            # Collect additional descriptive information
            avg_age = matched_target['21003-2.0'].mean()
            sex_distribution = matched_target['31-0.0'].mean()
            num_matches = len(matched_target)

            descriptive_table.append([idp_type, t1_q, rf_q, avg_age, sex_distribution, num_matches])

    # Plot the heatmap with the updated title
    plt.figure(figsize=(14, 10))
    sns.heatmap(heatmap_matrix, annot=True, cmap='YlGnBu', fmt=".2f", cbar=True)
    plt.title(heatmap_titles[idp_type], fontsize=12)
    plt.xlabel('rfMRI Head Motion Quartiles')
    plt.ylabel('T1 Head Motion Quartiles')
    plt.tight_layout()
    plt.savefig(f"{idp_type}_IDP_heatmap_final.png")
    plt.close()

# Create a descriptive table of results
descriptive_df = pd.DataFrame(
    descriptive_table,
    columns=['IDP Type', 'T1 Quartile', 'rfMRI Quartile', 'Avg Age', 'Sex Distribution (M)', 'Number of Matches']
)
descriptive_df.to_csv("descriptive_table.csv", index=False)
print("Heatmap generation and descriptive table complete. Check the saved files for results.")





#Visualing racial demographics data distribution in Emotional Regulation project
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

df_emotions = pd.read_csv("C:/Users/Cabria/PycharmProjects/pythonProject/Emotional_Regulation_Project_filtered.csv", low_memory=False)
df_emotions = df_emotions.dropna()
ethnic_groups = "21000-2.0"



# Mapping codes to categories
ethnic_mapping = {
    1: 'White',
    1001: 'White - British',
    1002: 'White - Irish',
    1003: 'White - Other',
    2: 'Mixed',
    2001: 'Mixed - White and Black Caribbean',
    2002: 'Mixed - White and Black African',
    2003: 'Mixed - White and Asian',
    2004: 'Mixed - Other',
    3: 'Asian or Asian British',
    3001: 'Asian - Indian',
    3002: 'Asian - Pakistani',
    3003: 'Asian - Bangladeshi',
    3004: 'Asian - Other',
    4: 'Black or Black British',
    4001: 'Black - Caribbean',
    4002: 'Black - African',
    4003: 'Black - Other',
    5: 'Chinese',
    6: 'Other ethnic group',
    -1: 'Do not know',
    -3: 'Prefer not to answer'
}

# Replace codes with labels
df_emotions['Ethnic Background'] = df_emotions[ethnic_groups].map(ethnic_mapping)

# Count occurrences of each category
ethnic_counts = df_emotions['Ethnic Background'].value_counts()

# Plot the histogram
plt.figure(figsize=(12, 8))
bars = ethnic_counts.plot(kind='bar', color='skyblue', edgecolor='k', alpha=0.7)
# Add counts above each bar
for bar in bars.patches:
    bar_height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,  # X-coordinate
        bar_height,  # Y-coordinate
        f'{int(bar_height)}',  # Text (count)
        ha='center',  # Horizontal alignment
        va='bottom',  # Vertical alignment
        fontsize=9  # Font size
    )
plt.title('Distribution of Ethnic Backgrounds in UKB (with complete data)', fontsize=12)
#(with complete data)
plt.xlabel('Ethnic Background', fontsize=12)
plt.ylabel('Number of Participants', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()





#Visualing Townsend index distribution in Emotional Regulation project + sample size
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

df_emotions = pd.read_csv("C:/Users/Cabria/PycharmProjects/pythonProject/Emotional_Regulation_Project_filtered.csv", low_memory=False)
df_emotions = df_emotions.dropna()

# Calculate the number of unique participant IDs in the cleaned dataset
#sample_size = df_emotions['eid'].nunique()
#print(f"Sample size for emotional regulation project: {sample_size}")

# Assuming the column is named 'Townsend_Index'
townsend_data = df_emotions['22189-0.0']
townsend_data = df_emotions['22189-0.0'].dropna()
# Plotting the histogram
plt.figure(figsize=(12, 6))

# Create the histogram
plt.hist(townsend_data, bins=30, color='skyblue', edgecolor='black')

# Add labels and title
plt.xlabel('Townsend Index')
plt.ylabel('Frequency')
plt.title('Histogram of Townsend Deprivation Index (UK Biobank Cohort)')

# Add text box for mean, median, standard deviation, and sample size
mean = townsend_data.mean()
median = townsend_data.median()
std_dev = townsend_data.std()
sample_size = townsend_data.count()
plt.text(0.95, 0.95, f'Mean = {mean:.1f} (SD = {std_dev:.1f})\nMedian = {median:.1f}\nSample Size = {sample_size}',
         transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(facecolor='white', alpha=0.5))

# Show the plot
plt.show()




#Visualing age distribution in Emotional Regulation project
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

df_emotions = pd.read_csv("C:/Users/Cabria/PycharmProjects/pythonProject/Emotional_Regulation_Project_filtered.csv", low_memory=False)
df_emotions.dropna()
age_data = df_emotions['21003-2.0']

# Plotting the histogram for Age
plt.figure(figsize=(12, 6))

# Create the histogram
plt.hist(age_data, bins=30, color='lightgreen', edgecolor='black')

# Add labels and title
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Histogram of Age in UKB Cohort')

# Add text box for mean, median, standard deviation, and sample size
mean_age = age_data.mean()
median_age = age_data.median()
std_dev_age = age_data.std()
sample_size_age = age_data.count()
plt.text(0.95, 0.95, f'Mean = {mean_age:.1f} (SD = {std_dev_age:.1f})\nMedian = {median_age:.1f}\nSample Size = {sample_size_age}',
         transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(facecolor='white', alpha=0.5))

# Show the plot
plt.show()



# Correlation matrix: age and tfMRI activation variables
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df_emotions = pd.read_csv("C:/Users/Cabria/PycharmProjects/pythonProject/Emotional_Regulation_Project_filtered.csv", low_memory=False)

# Extract relevant data for age and Hariri tfMRI variables
age_data = df_emotions['21003-2.0']  # Age variable
amygdala_faces_shapes = df_emotions['25054-2.0']  # Faces-Shapes contrast (Amygdala)
faces_activation = df_emotions['25046-2.0']  # Faces activation (Whole Mask)
shapes_activation = df_emotions['25042-2.0']  # Shapes activation (Whole Mask)
faces_shapes_whole = df_emotions['25050-2.0']  # Faces-Shapes contrast (Whole Mask)
age_column = '21003-2.0'
# Create a new DataFrame for correlation
df_corr = pd.DataFrame({
    'Age': age_data,
    'Amygdala Contrast': amygdala_faces_shapes,
    'Faces (Whole Mask)': faces_activation,
    'Shapes (Whole Mask)': shapes_activation,
    'Whole Brain Contrast': faces_shapes_whole
})

# Drop rows with missing values to avoid NaNs in correlation
df_corr.dropna(inplace=True)

# Filter out participants below the age of 50 and count removed rows
initial_count = len(df_emotions)
df_emotions = df_emotions[df_emotions[age_column] >= 50]
removed_count = initial_count - len(df_emotions)
print(f"Number of participants removed for being below 50 years old: {removed_count}")

# Calculate the correlation matrix
correlation_matrix = df_corr.corr()

# Print the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

# Visualize the correlation matrix using seaborn heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='BuPu', linewidths=0.5)
plt.title('Correlation between Age and tfMRI Activation Variables')
plt.show()



# removing those who are below 50
# Linear regression #1 with SES (Townsend deprivation index) moderating the strength

import pandas as pd
import numpy as np
from scipy.stats import zscore
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

file_path = "C:/Users/Cabria/PycharmProjects/pythonProject/Emotional_Regulation_Project_filtered.csv"
df_emotions = pd.read_csv(file_path, low_memory=False)

# making sure all required columns are present
required_columns = ['21003-2.0', '25054-2.0', '25046-2.0', '25042-2.0', '25050-2.0',
                    '22189-0.0', '31-0.0', '54-0.0', '26521-2.0', '25742-2.0']
missing_columns = [col for col in required_columns if col not in df_emotions.columns]
if missing_columns:
    raise KeyError(f"The following required columns are missing: {missing_columns}")

df_emotions = df_emotions[required_columns].dropna()

# Filter out participants below the age of 50 and count removed rows
initial_count = len(df_emotions)
df_emotions = df_emotions[df_emotions['21003-2.0'] >= 50]
removed_count = initial_count - len(df_emotions)
print(f"Number of participants removed for being below 50 years old: {removed_count}")
print(f"Total sample size after filtering: {len(df_emotions)}")

# Rename columns
df_emotions.rename(columns={
    '21003-2.0': 'Age',
    '25054-2.0': 'Amygdala_Contrast',
    '25046-2.0': 'Faces_WholeMask',
    '25042-2.0': 'Shapes_WholeMask',
    '25050-2.0': 'WholeBrain_Contrast',
    '22189-0.0': 'Townsend_Index',
    '31-0.0': 'Sex',
    '54-0.0': 'Site',
    '26521-2.0': 'Intracranial_Volume',
    '25742-2.0': 'Head_Motion'
}, inplace=True)

# Z-score normalization
df_emotions = df_emotions.apply(zscore)

# Create interaction term (moderation)
df_emotions['Age_Townsend_Interaction'] = df_emotions['Age'] * df_emotions['Townsend_Index']

# Define independent variables
independent_vars = ['Age', 'Townsend_Index', 'Sex', 'Site', 'Intracranial_Volume', 'Head_Motion', 'Age_Townsend_Interaction']

# Define dependent variables
dependent_variables = ['Amygdala_Contrast', 'Faces_WholeMask', 'Shapes_WholeMask', 'WholeBrain_Contrast']

# Run separate regressions for each dependent variable
results_dict = []

for dv in dependent_variables:
    Y = df_emotions[dv]
    X = df_emotions[independent_vars]
    model = sm.OLS(Y, X).fit()

    # save results
    for var in independent_vars:
        results_dict.append({
            'Dependent Variable': dv,
            'Independent Variable': var,
            'Coefficient': model.params[var],
            'P-Value': model.pvalues[var]
        })

    # Print model summary
    print(f"\nRegression Results for {dv}:\n")
    print(model.summary())

# Convert results to df
results_df = pd.DataFrame(results_dict)

# Save results to CSV
output_file_path = "C:/Users/Cabria/PycharmProjects/pythonProject/Emotion_Regulation_Results.csv"
results_df.to_csv(output_file_path, index=False)
print(f"\nAnalysis complete. Results saved to {output_file_path}.")

# Visualization: swarm plot
plt.figure(figsize=(12, 8))
sns.swarmplot(
    x='Dependent Variable',
    y='Coefficient',
    hue='Independent Variable',
    data=results_df,
    palette='Set2',
    size= 10
)
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
plt.title('Swarm Plot of SES Regression Coefficients by Hariri tfMRI')
plt.xlabel('Dependent Variable')
plt.ylabel('Coefficient')
plt.legend(title='Independent Variable', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



# Linear regression #1 with SES (Townsend deprivation index) moderating the strength

import pandas as pd
import numpy as np
from scipy.stats import zscore
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


file_path = "C:/Users/Cabria/PycharmProjects/pythonProject/Emotional_Regulation_Project_filtered.csv"
df_emotions = pd.read_csv(file_path, low_memory=False)

# making sure all required columns are present
required_columns = ['21003-2.0', '25054-2.0', '25046-2.0', '25042-2.0', '25050-2.0',
                    '22189-0.0', '31-0.0', '54-0.0', '26521-2.0', '25742-2.0']
missing_columns = [col for col in required_columns if col not in df_emotions.columns]
if missing_columns:
    raise KeyError(f"The following required columns are missing: {missing_columns}")

df_emotions = df_emotions[required_columns].dropna()


# Filter out participants below the age of 50 and count removed rows
initial_count = len(df_emotions)
df_emotions = df_emotions[df_emotions['21003-2.0'] >= 50]
removed_count = initial_count - len(df_emotions)
print(f"Number of participants removed for being below 50 years old: {removed_count}")
print(f"Total sample size after filtering: {len(df_emotions)}")

# Rename columns
df_emotions.rename(columns={
    '21003-2.0': 'Age',
    '25054-2.0': 'Amygdala_Contrast',
    '25046-2.0': 'Faces_WholeMask',
    '25042-2.0': 'Shapes_WholeMask',
    '25050-2.0': 'WholeBrain_Contrast',
    '22189-0.0': 'Townsend_Index',
    '31-0.0': 'Sex',
    '54-0.0': 'Site',
    '26521-2.0': 'Intracranial_Volume',
    '25742-2.0': 'Head_Motion'
}, inplace=True)

# Z-score normalization
df_emotions = df_emotions.apply(zscore)

# Create interaction term (moderation)
df_emotions['Age_Townsend_Interaction'] = df_emotions['Age'] * df_emotions['Townsend_Index']

# Define independent variables
independent_vars = ['Age', 'Townsend_Index', 'Sex', 'Site', 'Intracranial_Volume', 'Head_Motion', 'Age_Townsend_Interaction']

# Define dependent variables
dependent_variables = ['Amygdala_Contrast', 'Faces_WholeMask', 'Shapes_WholeMask', 'WholeBrain_Contrast']

# Run separate regressions for each dependent variable
results_dict = []

for dv in dependent_variables:
    Y = df_emotions[dv]
    X = df_emotions[independent_vars]
    model = sm.OLS(Y, X).fit()

    # save results
    for var in independent_vars:
        results_dict.append({
            'Dependent Variable': dv,
            'Independent Variable': var,
            'Coefficient': model.params[var],
            'P-Value': model.pvalues[var]
        })

    # Print model summary
    print(f"\nRegression Results for {dv}:\n")
    print(model.summary())

# Convert results to df
results_df = pd.DataFrame(results_dict)

# Save results to CSV
output_file_path = "C:/Users/Cabria/PycharmProjects/pythonProject/Emotion_Regulation_Results.csv"
results_df.to_csv(output_file_path, index=False)
print(f"\nAnalysis complete. Results saved to {output_file_path}.")

# Visualization: swarm plot
plt.figure(figsize=(12, 8))
sns.swarmplot(
    x='Dependent Variable',
    y='Coefficient',
    hue='Independent Variable',
    data=results_df,
    palette='Set2',
    size= 10
)
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
plt.title('Swarm Plot of SES Regression Coefficients by Hariri tfMRI')
plt.xlabel('Dependent Variable')
plt.ylabel('Coefficient')
plt.legend(title='Independent Variable', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()





# Linear regression #1 with simplified dummy variables for ethnicity (does not account for disparity in group sizes)
import pandas as pd
import numpy as np
from scipy.stats import zscore
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

# File path
file_path = "C:/Users/Cabria/PycharmProjects/pythonProject/Emotional_Regulation_Project_filtered.csv"
df_emotions = pd.read_csv(file_path, low_memory=False)

# Ensure all required columns are present
required_columns = ['21003-2.0', '25054-2.0', '25046-2.0', '25042-2.0', '25050-2.0',
                    '21000-2.0', '31-0.0', '54-0.0', '26521-2.0', '25742-2.0']
missing_columns = [col for col in required_columns if col not in df_emotions.columns]
if missing_columns:
    raise KeyError(f"The following required columns are missing: {missing_columns}")

# Keep only the required columns and drop rows with missing values
df_emotions = df_emotions[required_columns].dropna()

# Rename columns
column_mapping = {
    '21003-2.0': 'Age',
    '25054-2.0': 'Amygdala_Contrast',
    '25046-2.0': 'Faces_WholeMask',
    '25042-2.0': 'Shapes_WholeMask',
    '25050-2.0': 'WholeBrain_Contrast',
    '21000-2.0': 'Race_Ethnicity',
    '31-0.0': 'Sex',
    '54-0.0': 'Site',
    '26521-2.0': 'Intracranial_Volume',
    '25742-2.0': 'Head_Motion'
}
df_emotions.rename(columns=column_mapping, inplace=True)

# Map Race_Ethnicity to subcategories
ethnicity_mapping = {
    1: 'White', 1001: 'British', 1002: 'Irish', 1003: 'Other White',
    2: 'Mixed', 2001: 'White-Black Caribbean', 2002: 'White-Black African',
    2003: 'White-Asian', 2004: 'Other Mixed',
    3: 'Asian', 3001: 'Indian', 3002: 'Pakistani',
    3003: 'Bangladeshi', 3004: 'Other Asian',
    4: 'Black', 4001: 'Caribbean', 4002: 'African',
    4003: 'Other Black', 5: 'Chinese', 6: 'Other'
}
df_emotions['Ethnicity_Subcategory'] = df_emotions['Race_Ethnicity'].map(ethnicity_mapping)

# Filter invalid Ethnicity_Subcategories (-1 and -3 Don't know and prefer not to answer)
df_emotions = df_emotions[df_emotions['Ethnicity_Subcategory'].notnull()]

# Define combined ethnic groups mapping
combined_ethnicity_mapping = {
    'British': 'White',
    'Irish': 'White',
    'Other White': 'White',
    'Caribbean': 'Non-White',
    'African': 'Non-White',
    'Indian': 'Non-White',
    'Chinese': 'Non-White',
    'Pakistani': 'Non-White',
    'Other Asian': 'Non-White',
    'Asian or Asian British': 'Non-White',
    'Bangladeshi': 'Non-White',
    'White-Black African': 'Non-White',
    'White-Black Caribbean': 'Non-White',
    'White-Asian': 'Non-White',
    'Mixed': 'Non-White'
}

# Map Ethnicity_Subcategory to simplified groups
df_emotions['Combined_Ethnicity'] = df_emotions['Ethnicity_Subcategory'].map(combined_ethnicity_mapping)

# Remove rows with missing Combined_Ethnicity values
df_emotions = df_emotions[df_emotions['Combined_Ethnicity'].notnull()]

# Create dummy variable for Non-White group
df_emotions['Non_White'] = (df_emotions['Combined_Ethnicity'] == 'Non-White').astype(int)

# Print counts for the Non_White variable
print(df_emotions['Non_White'].value_counts())

# Convert Combined_Ethnicity into dummy variables
df_emotions = pd.get_dummies(df_emotions, columns=['Combined_Ethnicity'], drop_first=True)

# Print a preview of the Ethnicity_Subcategory, Combined_Ethnicity, and Non_White columns
print(df_emotions[['Ethnicity_Subcategory', 'Non_White'] + [col for col in df_emotions.columns if 'Combined_Ethnicity_' in col]].head())

# Z-score normalization (only for continuous variables)
continuous_vars = ['Age', 'Amygdala_Contrast', 'Faces_WholeMask', 'Shapes_WholeMask',
                   'WholeBrain_Contrast', 'Intracranial_Volume', 'Head_Motion']
df_emotions[continuous_vars] = df_emotions[continuous_vars].apply(zscore)

# Validate Race_Ethnicity interpretation in interaction term
unique_race_ethnicity = df_emotions['Race_Ethnicity'].unique()
print("Unique values in Race_Ethnicity:", unique_race_ethnicity)

# Interaction term for Age and Race_Ethnicity
df_emotions['Age_Race_Interaction'] = df_emotions['Age'] * df_emotions['Race_Ethnicity']

# Define independent and dependent variables
independent_vars = ['Age', 'Sex', 'Site', 'Intracranial_Volume', 'Head_Motion', 'Age_Race_Interaction', 'Non_White']
dependent_variables = ['Amygdala_Contrast', 'Faces_WholeMask', 'Shapes_WholeMask', 'WholeBrain_Contrast']

# Run regression on the entire dataset
results = []
for dv in dependent_variables:
    try:
        Y = df_emotions[dv]
        X = df_emotions[independent_vars]
        model = sm.OLS(Y, X).fit()

        for var in independent_vars:
            results.append({
                'Dependent Variable': dv,
                'Independent Variable': var,
                'Coefficient': model.params[var],
                'P-Value': model.pvalues[var]
            })

        # Print summary for debugging
        print(f"Results for {dv}:")
        print(model.summary())
    except Exception as e:
        print(f"Error processing {dv}: {e}")

# Save results to DataFrame and CSV
if results:
    results_df = pd.DataFrame(results)  # All coefficients are saved to the output file
    output_file = "C:/Users/Cabria/PycharmProjects/pythonProject/Emotion_Regulation_Results_Simplified_Ethnicity.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
else:
    print("No results to save.")

# Visualization: Swarm Plot
if not results_df.empty:
    plt.figure(figsize=(12, 8))
    sns.swarmplot(
        x='Dependent Variable',
        y='Coefficient',
        hue='Independent Variable',
        data=results_df[results_df['Independent Variable'].isin(['Age', 'Age_Race_Interaction', 'Non_White'])],
        palette='Set2',
        size=10
    )
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
plt.title('Swarm Plot of Race/Ethnicity Regression Coefficients by Hariri tfMRI')
plt.xlabel('Dependent Variable')
plt.ylabel('Coefficient')
plt.legend(title='Independent Variable', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("C:/Users/Cabria/PycharmProjects/pythonProject/Swarm_Plot2.png")  # Save plot as file
print("Swarm plot saved.")






# Linear regression with bootstrapping for balanced groups
import pandas as pd
import numpy as np
from scipy.stats import zscore
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import matplotlib
from sklearn.utils import resample

# Set backend to Agg for non-interactive rendering
matplotlib.use('Agg')

warnings.filterwarnings('ignore')

# File path
file_path = "C:/Users/Cabria/PycharmProjects/pythonProject/Emotional_Regulation_Project_filtered.csv"
df_emotions = pd.read_csv(file_path, low_memory=False)

# Ensure all required columns are present
required_columns = ['21003-2.0', '25054-2.0', '25046-2.0', '25042-2.0', '25050-2.0',
                    '21000-2.0', '31-0.0', '54-0.0', '26521-2.0', '25742-2.0']
missing_columns = [col for col in required_columns if col not in df_emotions.columns]
if missing_columns:
    raise KeyError(f"The following required columns are missing: {missing_columns}")

# Keep only the required columns and drop rows with missing values
df_emotions = df_emotions[required_columns].dropna()

# Rename columns
column_mapping = {
    '21003-2.0': 'Age',
    '25054-2.0': 'Amygdala_Contrast',
    '25046-2.0': 'Faces_WholeMask',
    '25042-2.0': 'Shapes_WholeMask',
    '25050-2.0': 'WholeBrain_Contrast',
    '21000-2.0': 'Race_Ethnicity',
    '31-0.0': 'Sex',
    '54-0.0': 'Site',
    '26521-2.0': 'Intracranial_Volume',
    '25742-2.0': 'Head_Motion'
}
df_emotions.rename(columns=column_mapping, inplace=True)

# Map Race_Ethnicity to subcategories
ethnicity_mapping = {
    1: 'White', 1001: 'British', 1002: 'Irish', 1003: 'Other White',
    2: 'Mixed', 2001: 'White-Black Caribbean', 2002: 'White-Black African',
    2003: 'White-Asian', 2004: 'Other Mixed',
    3: 'Asian', 3001: 'Indian', 3002: 'Pakistani',
    3003: 'Bangladeshi', 3004: 'Other Asian',
    4: 'Black', 4001: 'Caribbean', 4002: 'African',
    4003: 'Other Black', 5: 'Chinese', 6: 'Other'
}
df_emotions['Ethnicity_Subcategory'] = df_emotions['Race_Ethnicity'].map(ethnicity_mapping)

# Filter invalid Ethnicity_Subcategories (-1 and -3 Don't know and prefer not to answer)
df_emotions = df_emotions[df_emotions['Ethnicity_Subcategory'].notnull()]

# Define combined ethnic groups mapping
combined_ethnicity_mapping = {
    'British': 'White',
    'Irish': 'White',
    'Other White': 'White',
    'Caribbean': 'Non-White',
    'African': 'Non-White',
    'Indian': 'Non-White',
    'Chinese': 'Non-White',
    'Pakistani': 'Non-White',
    'Other Asian': 'Non-White',
    'Asian or Asian British': 'Non-White',
    'Bangladeshi': 'Non-White',
    'White-Black African': 'Non-White',
    'White-Black Caribbean': 'Non-White',
    'White-Asian': 'Non-White',
    'Mixed': 'Non-White'
}

# Map Ethnicity_Subcategory to simplified groups
df_emotions['Combined_Ethnicity'] = df_emotions['Ethnicity_Subcategory'].map(combined_ethnicity_mapping)

# Remove rows with missing Combined_Ethnicity values
df_emotions = df_emotions[df_emotions['Combined_Ethnicity'].notnull()]

# Create dummy variable for Non-White group
df_emotions['Non_White'] = (df_emotions['Combined_Ethnicity'] == 'Non-White').astype(int)

# Separate the data into Non-White and White groups
non_white = df_emotions[df_emotions['Non_White'] == 1]
white = df_emotions[df_emotions['Non_White'] == 0]

# Bootstrap 199 samples from the White group to match the size of the Non-White group
white_bootstrap = resample(white, replace=False, n_samples=199, random_state=42)

# Combine the Non-White group with the bootstrapped White group
balanced_df = pd.concat([non_white, white_bootstrap])

# Z-score normalization (only for continuous variables)
continuous_vars = ['Age', 'Amygdala_Contrast', 'Faces_WholeMask', 'Shapes_WholeMask',
                   'WholeBrain_Contrast', 'Intracranial_Volume', 'Head_Motion']
balanced_df[continuous_vars] = balanced_df[continuous_vars].apply(zscore)

# Interaction term for Age and Race_Ethnicity
balanced_df['Age_Race_Interaction'] = balanced_df['Age'] * balanced_df['Race_Ethnicity']

# Define independent and dependent variables
independent_vars = ['Age', 'Sex', 'Site', 'Intracranial_Volume', 'Head_Motion', 'Age_Race_Interaction', 'Non_White']
dependent_variables = ['Amygdala_Contrast', 'Faces_WholeMask', 'Shapes_WholeMask', 'WholeBrain_Contrast']

# Run regression on the balanced dataset
results = []
for dv in dependent_variables:
    try:
        Y = balanced_df[dv]
        X = balanced_df[independent_vars]
        model = sm.OLS(Y, X).fit()

        for var in independent_vars:
            results.append({
                'Dependent Variable': dv,
                'Independent Variable': var,
                'Coefficient': model.params[var],
                'P-Value': model.pvalues[var]
            })

        # Print summary for debugging
        print(f"Results for {dv}:")
        print(model.summary())
    except Exception as e:
        print(f"Error processing {dv}: {e}")

# Save results to DataFrame and CSV
if results:
    results_df = pd.DataFrame(results)  # All coefficients are saved to the output file
    output_file = "C:/Users/Cabria/PycharmProjects/pythonProject/Emotion_Regulation_Results_Bootstrap.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
else:
    print("No results to save.")

# Visualization: Swarm Plot
if not results_df.empty:
    plt.figure(figsize=(12, 8))
    sns.swarmplot(
        x='Dependent Variable',
        y='Coefficient',
        hue='Independent Variable',
        data=results_df[results_df['Independent Variable'].isin(['Age', 'Age_Race_Interaction', 'Non_White'])],
        palette='Set2',
        size=10
    )
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
plt.title('Swarm Plot of Race/Ethnicity Regression Coefficients (Bootstrapped)')
plt.xlabel('Dependent Variable')
plt.ylabel('Coefficient')
plt.legend(title='Independent Variable', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("C:/Users/Cabria/PycharmProjects/pythonProject/Swarm_Plot.png")  # Save plot as file
print("Swarm plot saved.")






# Removing subjects in dataset based on new UKB update
import pandas as pd

# Load datasets
df_emotions = pd.read_csv("C:/Users/Cabria/PycharmProjects/pythonProject/Emotional_Regulation_Project.csv", low_memory=False)
remove_subjects = pd.read_csv("C:/Users/Cabria/Downloads/w47267_20241217.csv", header=None, names=['eid'], low_memory=False)

# Check the initial size of the dataset
initial_count = len(df_emotions)

# Remove subjects from df_emotions using 'eid' column
filtered_df = df_emotions[~df_emotions['eid'].isin(remove_subjects['eid'])]

# Check the final size of the dataset
final_count = len(filtered_df)

# Calculate the number of subjects removed
removed_count = initial_count - final_count

# Print summary
print(f"Initial number of subjects: {initial_count}")
print(f"Number of subjects removed: {removed_count}")
print(f"Final number of subjects: {final_count}")

# Save the updated dataset to a new file
filtered_df.to_csv("C:/Users/Cabria/PycharmProjects/pythonProject/Emotional_Regulation_Project_filtered.csv", index=False)
print("Filtered dataset saved successfully.")




# Becker Library Intro Python Course Practice
import pandas as pd
Practice_DATA = pd.read_csv("C:/Users/Cabria/Downloads/study_data.csv")

# Data Exploration
print(type(Practice_DATA))  # Returns the type of object
print(Practice_DATA.shape)  # Returns the dimensions of the object
print(Practice_DATA.head())  # Returns the first several rows of the dataframe
print(Practice_DATA.tail())  # Returns the last several rows of the dataframe
print(Practice_DATA.describe())  # Returns summary statistics of numeric columns

# Extracting Single or Multiple Elements from a DataFrame
# ------------------------------------------------------
# There are various methods to extract data elements from a DataFrame in Pandas.
# Two common methods are `iloc` and `loc`, which are used to access rows and columns.

# Syntax for extracting elements using iloc:
# dataframe.iloc[row, column]
# - `iloc` is primarily used for positional indexing.
# - Both the row and column indices must be integers.

# Syntax for extracting elements using loc:
# dataframe.loc[row, column]
# - `loc` is primarily used for label-based indexing.
# - Rows and columns are accessed using their labels (e.g., row/column names).

# Note:
# - Python uses zero-based indexing, meaning the first row or column is index 0,
#   the second is index 1, and so on.

print(Practice_DATA.iloc[1,0]) # extract element from second row, first column
print(Practice_DATA.iloc[0:2, 1:3]) # extract elements from row 1 to 2 and columns 2 to 3, a new dataframe is created
print(Practice_DATA.iloc[:,[3]]) # extract all elements in the 4th column
print(Practice_DATA.loc[:,["BMI"]]) # here, we use loc method and column name to extract all rows of BMI column

#How to extract non-continuous items
subsetDATA = Practice_DATA.loc[:,["ID", "Exercise", "BMI"]] #select all the rows of data for these columns only
print(subsetDATA)
subsetDATA.shape
Practice_DATA.loc[Practice_DATA["BMI"]>25,:] # select rows for participants with BMI greater than 25
Practice_DATA.loc[Practice_DATA['Exercise'] == 'Daily',:] # select rows for participants who exercise daily

# Adding a new column using data in existing column
Practice_DATA['Weight_kg'] = Practice_DATA['Weight']/2.2046
print(Practice_DATA)
Practice_DATA['Weight_kg'] = round(Practice_DATA['Weight']/2.2046, 1) # round output to 1 decimal place
print(Practice_DATA)

# Delete rows and columns using the .drop() function
# Syntax to delete column: dataframe.drop(columns='columnname')
# Syntax to delete row: dataframe.drop(dataframe.index[row_index])
delWeight_kg = Practice_DATA.drop(columns='Weight_kg') # delete weight_kg column
print(delWeight_kg)
del_IDWeight_kg = Practice_DATA.drop(columns=['ID','Weight_kg']) # delete multiple columns - ID and Weight_kg columns
print(del_IDWeight_kg)
del_index4 = Practice_DATA.drop(Practice_DATA.index[4]) # delete a single row, row 5
del_indexes4to7 = Practice_DATA.drop(Practice_DATA.index[4:7]) # delete multiple rows - rows 5 to 8
print(del_index4)
print(del_indexes4to7)

# Export Data
subsetDATA.to_csv('subsetDATA.csv')# export subsetDATA

# Data Visualization
# Using Matplotlib and Seaborn libraries you can save plots as a png, jpeg, or pdf using plt.savefig('filename.png/pdf/jpeg')
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import plotly.express as px
matplotlib.use('TkAgg')

Practice_DATA = pd.read_csv("C:/Users/Cabria/Downloads/study_data.csv")
# Basic histogram
# histogram of the Distribution of Practice_DATA BMI
plt.hist(Practice_DATA['BMI'])
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.title('Distribution of Practice_DATA BMI values')
plt.savefig('hist1.png')
plt.show()

# Basic histogram
# customize plot - add color and adjust the number of bins
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import plotly.express as px
matplotlib.use('TkAgg')

Practice_DATA = pd.read_csv("C:/Users/Cabria/Downloads/study_data.csv")
plt.hist(Practice_DATA['BMI'],color='purple', edgecolor='red', bins=6)
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.title('Distribution of PracticeDATA BMI values')
plt.savefig('hist2.png')
plt.show()


# basic scatter plot
# scatter plot of BMI by Weight
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import plotly.express as px
matplotlib.use('TkAgg')
plt.scatter(Practice_DATA['Weight'],Practice_DATA['BMI'])
plt.xlabel('Weight(lb)')
plt.ylabel('BMI')
plt.title('BMI by Weight')
plt.savefig('scatter1.png')
plt.show()


# scatter plot
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import plotly.express as px
matplotlib.use('TkAgg')
# customize plot - add color, adjust shape and size of points
# size is specified by the s parameter and shape is specified by the marker parameter
# marker and s options are defined here: https://matplotlib.org/3.3.3/api/markers_api.html#module-matplotlib.markers
plt.scatter(Practice_DATA['Weight'],Practice_DATA['BMI'], color='red', marker='*', s=200)
plt.xlabel('Weight(lb)')
plt.ylabel('BMI')
plt.title('BMI by Weight')
plt.savefig('redscatterplot70.pdf')
plt.savefig('scatter89.png')
plt.show()

# Seaborn scatter plot - customize plot - add color based on a variable
# Can use Matplotlib, but requires several lines of code
# with Seaborn, requires only one line of code, axis labels and legend are also generated

ax=sns.scatterplot(data=Practice_DATA, x='Weight', y='BMI',hue='Diet')

# add color based on Diet variable - assign Diet column to hue parameter, it will map diet levels to color of points
# the colors are assigned to values based on the order of indexing

ax=sns.scatterplot(data=Practice_DATA, x='Weight', y='BMI',hue='Diet')
plt.xlabel('Weight(lb)')
plt.ylabel('BMI')
plt.title('BMI by Weight')
plt.savefig('scatter3.png')

# choosing color palettes
# run sns.color_palette() to check what the current palette is
# then run sns.palplot(current_palette) like below, to display the current color palette
# the sns.palplot () function displays the palette as an array of colors

current_palette = sns.color_palette()
sns.palplot(current_palette)

# scatter plot - customize plot and add color based on a variable, altering the default point colors
# to alter the default colors, can change the current color palette
# can choose a built-in color palette from here: https://seaborn.pydata.org/tutorial/color_palettes.html
# use sns.color_palette()function to build the palette and sns.set_palette() function to set it as current palette

# not interested in built-in palettes? Create your own palette with your colors of choice, like this:

colors = ['red', 'green']# create an array with your colors of choice
sns.set_palette(sns.color_palette(colors))# build the palette and set it as the current color palette
ax=sns.scatterplot(data=Practice_DATA, x='Weight', y='BMI',hue='Diet', s=150)
plt.xlabel('Weight(lb)')
plt.ylabel('BMI')
plt.title('BMI by Weight')
plt.savefig('scatter4.png')

# reverting to the Seaborn default color palette
# the Seaborn default colors are ordered as the default matplotlib color palette, "tab10"
# to access default colors, set "tab10" as the current palette

current_palette = sns.color_palette('tab10')# build the palette
sns.set_palette(current_palette)# set the palette
sns.palplot(current_palette)# display the palette as an array of colors

# basic box plot
# Seaborn's boxplot() function generates box plot, requires only one line of code
# boxplot of BMI and Diet frequency
ax=sns.boxplot(data=Practice_DATA, x='Diet', y='BMI')
plt.xlabel('Diet')
plt.ylabel('BMI')
plt.title('Box plot of Diet by BMI Frequency')
plt.savefig('box1.png')

# basic boxplot
# customize plot - add points on the plot
# use seaborn's sns.swarmplot() function to add points on box plot
ax=sns.boxplot(data=Practice_DATA, x='Diet', y='BMI')
ax=sns.swarmplot(data=Practice_DATA, x='Diet', y='BMI', color='black') # use swarmplot function to add points to categorical plots
plt.xlabel('Diet')
plt.ylabel('BMI')
plt.title('Box plot of Diet by BMI Frequency')
plt.savefig('box2.png')

# Plotly Express (interactive plots)
# histogram of Distribution of BMI values Among Study Participants
# customizations are added within the plot function
# below, we adjust the number of bins to 15, add the color violet, and label both the plot axes and title of the plot

hist = px.histogram(Practice_DATA, x="BMI", color_discrete_sequence = ["violet"], nbins=15,
        title="Distribution of BMI Among Study Participants", labels={"BMI":"BMI"})
hist.show()

# scatter plot of Weight by Height with points colored red
scatter = px.scatter(Practice_DATA, x="Height", y="Weight", title="Weight by Height", labels={"Weight":"Weight (lb)",
        "Height":"Height (in)"}, color_discrete_sequence = ["red"])
scatter.show()

#box plot of Weight and Exercise Frequency
box = px.box(Practice_DATA, x="Exercise", y="Weight", title="Weight and Exercise Frequency",
    labels={"Weight":"Weight (lb)","Exercise":"Exercise"}, color_discrete_sequence = ["purple"], points="all")
box.show()


# Image export using the "kaleido" engine requires the kaleido package, which can be installed using pip:
# $ pip install -U kaleido
box.write_image("box.jpeg") # export box plot as a jpeg image file
scatter.write_image("scatter.pdf") # export scatter plot as a pdf file
hist.write_image("hist.png") # export histogram as a png image file
