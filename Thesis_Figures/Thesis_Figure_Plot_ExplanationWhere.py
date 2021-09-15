import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Explanation "Where": Plot for explanation in fundamentals chapter

# 1. Generate data with gaussian, uniform and mixed distribution
n = 3000
var = 0.12
# Dimension 1
dim1_sequence_100percent_gaussian = np.random.normal(0.5, var, n)
dim1_sequence_100percent_uniform = np.random.uniform(-0.5, 1.5, n)
dim1_sequence_mixed = np.append(np.random.normal(0.5, var, int(n/2)), np.random.uniform(-0.5, 1.5, n))
# Dimension 2
dim2_sequence_100percent_gaussian = np.random.normal(0.5, var, n)
dim2_sequence_100percent_uniform = np.random.uniform(-0.5, 1.5, n)
dim2_sequence_mixed = np.append(np.random.normal(0.5, var, int(n/2)), np.random.uniform(-0.5, 1.5, n))
# Shuffle data
random.shuffle(dim1_sequence_100percent_gaussian)
random.shuffle(dim1_sequence_100percent_uniform)
random.shuffle(dim1_sequence_mixed)
random.shuffle(dim2_sequence_100percent_gaussian)
random.shuffle(dim2_sequence_100percent_uniform)
random.shuffle(dim2_sequence_mixed)

# 2. Generate 2-dimensional dataset
# Gaussian-uniform
df_gaussian_uniform = pd.DataFrame()
df_gaussian_uniform['Dim 1']=pd.Series(dim1_sequence_100percent_gaussian)
df_gaussian_uniform['Dim 2']=pd.Series(dim2_sequence_100percent_uniform)
# Gaussian-mixed
df_gaussian_mixed = pd.DataFrame()
df_gaussian_mixed['Dim 1']=pd.Series(dim1_sequence_100percent_gaussian)
df_gaussian_mixed['Dim 2']=pd.Series(dim2_sequence_mixed)
# Uniform-mixed
df_uniform_mixed = pd.DataFrame()
df_uniform_mixed['Dim 1']=pd.Series(dim1_sequence_100percent_uniform)
df_uniform_mixed['Dim 2']=pd.Series(dim2_sequence_mixed)
# Gaussian-gaussian
df_gaussian_gaussian = pd.DataFrame()
df_gaussian_gaussian['Dim 1']=pd.Series(dim1_sequence_100percent_gaussian)
df_gaussian_gaussian['Dim 2']=pd.Series(dim2_sequence_100percent_gaussian)
# Uniform-uniform
df_uniform_uniform = pd.DataFrame()
df_uniform_uniform['Dim 1']=pd.Series(dim1_sequence_100percent_uniform)
df_uniform_uniform['Dim 2']=pd.Series(dim2_sequence_100percent_uniform)

# 3. Prepare plot
plt.style.use("ggplot")
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=22)

# 3.1 Gaussian-Gaussian Plot
JP1 = sns.JointGrid(data=df_gaussian_gaussian, x="Dim 1", y="Dim 2", xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
# add scatterplot layer
JP1.plot_joint(sns.scatterplot)
# add marginal density plot layer
JP1.plot_marginals(sns.kdeplot)
plt.savefig("Figures/Plot_Where_2D_Part1.pdf")

# 3.2 Gaussian-Mixed Plot
JP2 = sns.JointGrid(data=df_gaussian_mixed, x="Dim 1", y="Dim 2", xlim=[-0.05, 1.05], ylim=[-0.05,  1.05])
# add scatterplot layer
JP2.plot_joint(sns.scatterplot)
# add marginal density plot layer
JP2.plot_marginals(sns.kdeplot)
plt.savefig("Figures/Plot_Where_2D_Part2.pdf")

# 3.3 Gaussian-Uniform Plot
JP3 = sns.JointGrid(data=df_gaussian_uniform, x="Dim 1", y="Dim 2", xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
# add scatterplot layer
JP3.plot_joint(sns.scatterplot)
# add marginal density plot layer
JP3.plot_marginals(sns.kdeplot)
plt.savefig("Figures/Plot_Where_2D_Part3.pdf")
