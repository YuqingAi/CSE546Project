from pylab import *
import matplotlib.pyplot as pyplot

x_pca = []
y_pca = []
x_fisher = []
y_fisher = []

i = 0

with open("pca3-20.txt") as f:
	for line in f:
		str = line.split()
		if len(str) == 2:
			x_pca.append(float(str[0]))
			y_pca.append(float(str[1]))
with open("fisher3-20.txt") as f:
	for line in f:
		str = line.split()
		if len(str) == 2:
			x_fisher.append(float(str[0]))
			y_fisher.append(float(str[1]))

fig = pyplot.figure()
ax = fig.add_subplot(1,1,1)
#ax.set_xlim([1,300])
#ax.set_ylim([0,1])

pyplot.xlabel('k')
pyplot.ylabel('Accuracy')

line1, = ax.plot(x_pca, y_pca, color='blue', lw=1)
line2, = ax.plot(x_fisher, y_fisher, color='red', lw=1)
ax.legend(['pca', 'fisher'], loc='lower right')

#ax.set_xscale('log')

show()