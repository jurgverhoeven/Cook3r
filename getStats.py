import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
labels = ['Green beans', 'Meatballs', 'Empty pan', 'Pasta']
amount = [176, 60, 128, 82]
ax.set_xticklabels(labels)

ax.bar(labels, amount)
plt.show()

import matplotlib.pyplot as plt
import numpy as np


labels = ['Green beans', 'Meatballs', 'Empty pan', 'Pasta']
men_means = [176, 60, 128, 82]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men_means, width, label='Men')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3)

fig.tight_layout()

plt.show()