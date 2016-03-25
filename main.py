__author__ = 'brendan'

#import matplotlib
import numpy as np
from colors import custom_colors
import matplotlib.pyplot as plt
import matplotlib.patches as patches

####################
# Styles
####################
#"""
#plt.style.use(['marker_size','linewidth'])
plt.style.use('custom1')
#List of available styles
#print plt.style.available
#"""



####################
# Basic
####################
"""
plt.plot([1,2,3,4], linewidth=5.0)          #Single list interpreted as y values
plt.ylabel('some numbers')
plt.plot([1,2,3,4], [1,4,9,16])       #(x's, ys, marker format)
plt.plot([0, 6, 0, 20], marker='o')                     #[xmin, xmax, ymin, ymax]
plt.axis([0,5,0,25])
plt.plot([1,7,1,21])
plt.plot([3,10,3,23])
plt.plot([4,11,4,24])
plt.plot([5,12,5,25])
plt.title("Title!")
plt.show()
plt.close()
"""


####################
# Multiple plots on one figure
####################
"""
t = np.arange(0., 5., 0.2)
# red dashes, blue squares and green triangles
lines = plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^') #Several plots as functions, returns 3 line objects
plt.setp(lines[0], color='r',label='Red Stuff')
plt.setp(lines[1], color='b',markevery=4)
plt.setp(lines[2], color='g',marker='o',linestyle='--',markeredgecolor='r',markerfacecolor='b',markersize=20)

plt.show()
plt.close()
"""

####################
# Multiple subplots
####################
"""
RowsOfSubplots = 2
ColsOfSubplots = 1
t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)
plt.figure(1)
which_subplot = 1
plt.subplot(RowsOfSubplots,ColsOfSubplots,which_subplot)
plt.ylabel('Number of licks')
plt.xlabel('Years Since LHS')
plt.plot(t1, t1**2,label='Dave')
plt.plot(t2, t2**3,label='Pat')
plt.legend(loc='right')

which_subplot = 2
plt.subplot(RowsOfSubplots,ColsOfSubplots,which_subplot)
plt.ylabel('Gray\'s Emotional State')
plt.plot(t2, np.cos(2*np.pi*t2)*5+5)

plt.show()
plt.close()
"""

####################
# Multiple figures
####################
"""
#Figure 1
plt.figure(1) # the first figure
plt.subplot(2,1,1) # the first subplot in the first figure
plt.plot([1, 2, 3])
plt.subplot(2,1,2) # the second subplot in the first figure
plt.plot([4, 5, 6])

#Figure 2
plt.figure(2) # a second figure
plt.plot([4, 5, 6]) # creates a subplot(111) by default
plt.title('3')

#Back to Figure 1
plt.figure(1) # figure 1 current; subplot(212) still current
plt.subplot(2,1,1)
plt.title('Easy as 1')
plt.subplot(2,1,2) # make subplot(211) in figure1 current
plt.title('2') # subplot 211 title
plt.figure(2)

plt.show()
plt.close()
"""

####################
# Text
####################
"""
mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)
# the histogram of the data
n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$') #$ --> LaTeX equation!!!
plt.axis([40, 160, 0, 0.03])
plt.grid(True)

plt.show()
plt.close()
"""

####################
# Annotating
####################
"""
arrow_tip_loc = (2,1)
text_loc = (3,1.5)
pct_gap_between_arrow_and_data = .05

ax = plt.subplot(111)
t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
line, = plt.plot(t, s, lw=2)
plt.annotate('local max', xy=arrow_tip_loc, xytext=text_loc,arrowprops=dict(facecolor='red', shrink=pct_gap_between_arrow_and_data))
plt.ylim(-2,2)

plt.show()
plt.close()
"""

####################
# Text
####################
"""
fig = plt.figure()
fig.suptitle('bold figure suptitle', fontsize=24, fontweight='bold')

ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.set_title('axes title')

ax.set_xlabel('xlabel')
ax.set_ylabel('ylabel')

ax.text(3, 8, 'boxed italics text in data coords', style='italic',bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
ax.text(2, 6, r'an equation: $E=mc^2$', fontsize=15)
ax.text(3, 2, u'unicode: Institut f\374r Festk\366rperphysik')
ax.text(0.95, 0.01, 'colored text in axes coords',
    verticalalignment='bottom', horizontalalignment='right',
    transform=ax.transAxes,
    color='green', fontsize=15)

ax.plot([2], [1], 'o')
ax.annotate('annotate', xy=(2, 1), xytext=(3, 4),
arrowprops=dict(facecolor='black', shrink=0.05))
ax.axis([0, 10, 0, 10])
plt.show()
"""

####################
# Non-linear scales
####################
"""
# make up some data in the interval ]0, 1[
y = np.random.normal(loc=0.5, scale=0.4, size=1000)
y = y[(y > 0) & (y < 1)]
y.sort()
x = np.arange(len(y))
# plot with various axes scales
plt.figure(1)
# linear
plt.subplot(131)
plt.plot(x, y)
plt.yscale('linear')
plt.title('linear')
plt.grid(True)
# log
plt.subplot(132)
plt.plot(x, y)
plt.yscale('log')
plt.title('log')
plt.grid(True)
# symmetric log
plt.subplot(133)
plt.plot(x, y - y.mean())
plt.yscale('symlog', linthreshy=0.05)
plt.title('symlog')
plt.grid(True)
plt.show()
"""

####################
# Error bar
####################
"""
# example data
x = np.arange(0.1, 4, 0.1)
y = np.exp(-x)
# example variable error bar values
yerr = 0.1 + 0.1*np.sqrt(x)
# Now switch to a more OO interface to exercise more features.
fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True)
ax = axs[0]
ax.errorbar(x, y, yerr=yerr)
ax.set_title('all errorbars')
ax = axs[1]
ax.errorbar(x, y, yerr=yerr, errorevery=5,capsize=10)
ax.set_title('only every 5th errorbar')
fig.suptitle('Errorbar subsampling for better visualibility')
plt.show()
"""

####################
# Shapes
####################
"""
circle1=plt.Circle((0,0),.2,color='r')
circle2=plt.Circle((.5,.5),.2,color='b')
circle3=plt.Circle((1,1),.2,color='g',clip_on=False)
fig = plt.gcf()
fig.gca().add_artist(circle1)
fig.gca().add_artist(circle2)
fig.gca().add_artist(circle3)
plt.show()
"""

####################
# Shapes with text
####################
"""
# build a rectangle in axes coords
left, width = .25, .5
bottom, height = .25, .5

right = left + width
top = bottom + height

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

# axes coordinates are 0,0 is bottom left and 1,1 is upper right
p = patches.Rectangle(
    (left, bottom), width, height,
    fill=False, transform=ax.transAxes, clip_on=False
    )

ax.add_patch(p)

ax.text(left, bottom, 'left top',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes)

ax.text(left, bottom, 'left bottom',
        horizontalalignment='left',
        verticalalignment='bottom',
        transform=ax.transAxes)

ax.text(right, top, 'right bottom',
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes)

ax.text(right, top, 'right top',
        horizontalalignment='right',
        verticalalignment='top',
        transform=ax.transAxes)

ax.text(right, bottom, 'center top',
        horizontalalignment='center',
        verticalalignment='top',
        transform=ax.transAxes)

ax.text(left, 0.5*(bottom+top), 'right center',
        horizontalalignment='right',
        verticalalignment='center',
        rotation='vertical',
        transform=ax.transAxes)

ax.text(left, 0.5*(bottom+top), 'left center',
        horizontalalignment='left',
        verticalalignment='center',
        rotation='vertical',
        transform=ax.transAxes)

ax.text(0.5*(left+right), 0.5*(bottom+top), 'middle',
        horizontalalignment='center',
        verticalalignment='center',fontsize=20, color='red',
        transform=ax.transAxes)

ax.text(right, 0.5*(bottom+top), 'centered',
        horizontalalignment='center',
        verticalalignment='center',
        rotation='vertical',
        transform=ax.transAxes)

ax.text(left, top, 'rotated\nwith newlines',
        horizontalalignment='center',
        verticalalignment='center',
        rotation=45,
        transform=ax.transAxes)

ax.set_axis_off() #turns off grid
plt.show()

"""

########################
# Polar shapes with text
########################
#"""
fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
r = np.arange(0,1,0.001)
theta = 2*2*np.pi*r
line, = ax.plot(theta, r, linewidth=3)
ind = 800
thisr, thistheta = r[ind], theta[ind]
ax.plot([thistheta], [thisr], 'o')
ax.annotate('a polar annotation',
xy=(thistheta, thisr), # theta, radius
xytext=(0.05, 0.05), # fraction, fraction
textcoords='figure fraction',
arrowprops=dict(facecolor='black', shrink=0.05),horizontalalignment='left',
verticalalignment='bottom')
plt.show()
#"""

