import numpy as np
from collections import defaultdict
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

def bootstrapped(data, n_resamples=50):
    data = (data,)
    bootstrap_ci = bootstrap(data, np.mean, confidence_level=0.95, n_resamples=n_resamples,
                             random_state=1, method='percentile')
    return bootstrap_ci.confidence_interval

color_2 = {
    'paper': '#bf5700'
}

# mapping fields of papers
M = {
    'computer science': 14
}
MP = {M[i]: i for i in M}
CS = [14]  # CS

FM = {}
for f in CS:
    FM[f] = 'CS'
FN = {
    'CS': 'Computer Science'
}

p_y = {}
Dis_mean = {}
TS = {}
p_d = {}
p_f = {}
with open('filtered_data.txt', 'r') as f:
    for line in f:
        line = line.strip('\n').split('\t')
        p = int(line[0])
        p_y[p] = int(line[1])
        Dis_mean[p] = float(line[4])
        TS[p] = int(line[3])
        if line[-1] != 'nan':
            p_d[p] = float(line[-1])
        if line[2] != '-1':
            p_f[p] = M[line[2]]

Dbins = defaultdict(list)
for p in p_d:
    if Dis_mean[p] <= 0:
        d = 0
    if 0 < Dis_mean[p] <= 600:
        d = int(Dis_mean[p] / 200 + 1) * 200
    if Dis_mean[p] > 600:
        d = 10000
    Dbins[d].append(p_d[p])

thdpa = 0

x0 = [i for i in sorted(list(Dbins.keys()))]
for distance, d_scores in Dbins.items():
    if len(d_scores) > 0:
        d_positive_count = sum(1 for d_score in d_scores if d_score > thdpa)
        total_count = len(d_scores)
        probability = d_positive_count / total_count
        print(f"Distance: {distance} km, Probability of D>0: {probability:.4f}")
    else:
        print(f"No data for distance: {distance} km")

blp = np.sum([int(p_d[p] > 0) for p in p_d]) / len(p_d)

y0 = [len(np.array(Dbins[i])[np.array(Dbins[i]) > thdpa]) / len(Dbins[i]) for i in x0 if len(Dbins[i]) > 0]
y0ci = [bootstrapped(list(map(lambda x: int(x > 0), Dbins[i])), 100) for i in x0 if len(Dbins[i]) > 0]

fig, ax2 = plt.subplots(1, 1, figsize=[6, 6])

ax2.plot(list(range(len(x0))), np.array(y0) / blp, '-', c=color_2['paper'], lw=8, label='Papers', zorder=100)

for i in range(len(x0)):
    y0ci_ = np.array(y0ci) / blp
    ax2.plot([i, i], [y0ci_[i][0], y0ci_[i][1]], lw=2, color=color_2['paper'], zorder=10)
    ax2.plot([i - 0.1, i + 0.1], [y0ci_[i][0], y0ci_[i][0]], lw=2, color=color_2['paper'], zorder=10)
    ax2.plot([i - 0.1, i + 0.1], [y0ci_[i][1], y0ci_[i][1]], lw=2, color=color_2['paper'], zorder=10)

ax2.axhline(1, ls='--', color='grey', lw=3)
ax2.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)
ax2.set_yticks([0.8, 0.9, 1, 1.1])
ax2.set_yticklabels(tuple(['0.80', '0.90', '1.00', '1.10']))
ax2.set_xticks([0, 1, 2, 3, 4])
ax2.set_xticklabels(tuple(['0', '200', '400', '600', '600+']))
ax2.set_xlabel('Collaboration distance (km)', size=16)
ax2.set_ylabel('Absolute probability of disruption', size=16)
ax2.legend(fontsize=14, frameon=False, loc=1)
trans = mtransforms.ScaledTranslation(-60 / 72, 15 / 72, fig.dpi_scale_trans)
ax2.text(.0, 1.0, '', transform=ax2.transAxes + trans,
         fontsize=25, va='bottom', fontfamily='arial')
ax2.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)

plt.tight_layout()
plt.show()
