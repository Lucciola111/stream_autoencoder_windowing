import numpy as np
import matplotlib.pyplot as plt

# Use LaTex Font
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.tight_layout()

fontsize = 15
params = {'axes.labelsize': fontsize, 'axes.titlesize': fontsize, 'legend.fontsize': fontsize,
          'xtick.labelsize': fontsize, 'ytick.labelsize': fontsize}
plt.rcParams.update(params)

k_values = np.arange(1, 21)
samples_needed = 10*2**k_values

plt.figure(figsize=(10, 4))
plt.plot(k_values, samples_needed)
plt.xlabel('Number of dimensions', fontsize=16)
plt.ylabel('Number of samples needed', fontsize=16)
plt.xticks(k_values, fontsize=12)
plt.yticks(np.arange(0, 1.2*10**7, .2*10**7), fontsize=12)
plt.axhline(10**6, linestyle='--', color='k', alpha=0.75)
plt.annotate('1 Million', (1, .15*10**7), fontsize=16)
plt.axhline(10**7, linestyle='--', color='k', alpha=0.75)
plt.annotate('10 Million', (1, .9*10**7), fontsize=16)
plt.savefig("./Figures/Figure_2_DataSparsity.pdf", bbox_inches='tight')
plt.show()
