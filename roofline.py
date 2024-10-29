#!/usr/bin/env python3

"""

This is based on:
- https://github.com/mohamed/roofline, Copyright 2018-2024, Mohamed A. Bamakhrama Licensed under BSD license shown in LICENSE
- https://github.com/giopaglia/rooflini/tree/master, Copyright Giopaglia
"""


import csv
import sys
import argparse
import numpy
import matplotlib.pyplot
import matplotlib
import numpy as np
# matplotlib.rc('font', family='Arial')


# Constants
# The following constants define the span of the intensity axis
START = -4
STOP = 6
N = abs(STOP - START + 1)

##########################################################
# set_size for explicitly setting axes widths/heights
# see: https://stackoverflow.com/a/44971177/5646732

def set_size(w,h, ax=None):
  """ w, h: width, height in inches """
  if not ax: ax=plt.gca()
  l = ax.figure.subplotpars.left
  r = ax.figure.subplotpars.right
  t = ax.figure.subplotpars.top
  b = ax.figure.subplotpars.bottom
  figw = float(w)/(r-l)
  figh = float(h)/(t-b)
  ax.figure.set_size_inches(figw, figh)

##########################################################


def roofline(num_platforms, peak_performance, peak_bandwidth, intensity):
    """
    Computes the roofline model for the given platforms.
    Returns The achievable performance
    """

    assert isinstance(num_platforms, int) and num_platforms > 0
    assert isinstance(peak_performance, numpy.ndarray)
    assert isinstance(peak_bandwidth, numpy.ndarray)
    assert isinstance(intensity, numpy.ndarray)
    assert (num_platforms == peak_performance.shape[0] and
            num_platforms == peak_bandwidth.shape[0])

    achievable_perf = numpy.zeros((num_platforms, len(intensity)))
    for i in range(num_platforms):
        achievable_perf[i:] = numpy.minimum(peak_performance[i],
                                            peak_bandwidth[i] * intensity)
    return achievable_perf


def process(hw_platforms, apps, xkcd):
    """
    Processes the hw_platforms and apps to plot the Roofline.
    """
    assert isinstance(hw_platforms, list)
    assert isinstance(apps, list)
    assert isinstance(xkcd, bool)

    fontsize_title = 20
    fontsize_axis = 16
    fontsize_labels = 14

    # arithmetic intensity
    arithmetic_intensity = numpy.logspace(START, STOP, num=N, base=2)

    # Compute the rooflines
    achv_perf = roofline(len(hw_platforms),
                         numpy.array([float(p[1]) for p in hw_platforms]),
                         numpy.array([float(p[2]) for p in hw_platforms]),
                         arithmetic_intensity)
    norm_achv_perf = roofline(len(hw_platforms),
                              numpy.array([(float(p[1])*1e3) / float(p[3])
                                           for p in hw_platforms]),
                              numpy.array([(float(p[2])*1e3) / float(p[3])
                                           for p in hw_platforms]),
                              arithmetic_intensity)
    # HW_platform is a list of tuples. Each tuple contains ('Name', 'Peak perf', 'Mem BW', ...)
    platform_names = [x[0] for x in hw_platforms]
    peak_performance = [float(x[1]) for x in hw_platforms]
    peak_bandwidth = [float(x[2]) for x in hw_platforms]

    # Apps
    if apps != []:
        apps_intensity = numpy.array([float(a[1]) for a in apps])

    # Plot the graphs
    if xkcd:
        matplotlib.pyplot.xkcd()
    fig, axis = matplotlib.pyplot.subplots(1, 1)
    axis.set_xscale('log', base=2)
    axis.set_yscale('log', base=2)
    axis.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=fontsize_axis)
    axis.grid(True, which='major')
    axis.grid(color="#dddddd", zorder=-1)


    fig_ratio = 2
    fig_dimension = 7

    matplotlib.pyplot.setp(axis, xticks=arithmetic_intensity,
                           yticks=numpy.logspace(-5, 20, num=26, base=2))

    axis.set_ylabel("Achieveable Performance (GFLOP/s)", fontsize=fontsize_axis)
    # axes[1].set_ylabel("Normalized Achieveable Performance (MFLOP/s/$)",
                    #    fontsize=12)

    axis.set_title('Roofline Model', fontsize=fontsize_title)

    # plot slopes (inspired by https://github.com/giopaglia/rooflini/)
    # Axis limits
    xmin, xmax, ymin, ymax = 0.001, 1500, 1, max(peak_performance)*1.5 # scale it to fit the data
    xlogsize = float(np.log10(xmax/xmin))
    ylogsize = float(np.log10(ymax/ymin))
    m = xlogsize/ylogsize
   

    for idx, val in enumerate(hw_platforms):
        # val = [name, performance, bandwidth, cost in $]
        roof = float(val[1]) # performance peak
        bandwidth = float(val[2])
        print("Roof: ", roof, ", bw: ", bandwidth)
        y = [0, roof]
        x = [float(yy)/bandwidth for yy in y]
        print(x,y)
        # plot line connecting x[0], y[0], to x[1]y[1]
        memory_bound_line = axis.loglog(x, y, linewidth=2.0,
            linestyle='-.',
            marker="2",
            zorder=10, label=val[0]) # use label to assign it to the legend
        
        
        xpos = xmin*(10**(xlogsize*0.04))
        ypos = xpos*bandwidth
        if ypos<ymin:
            ypos = ymin*(10**(ylogsize*0.02))
            xpos = ypos/bandwidth
        pos = (xpos, ypos)

        # In case of linear plotting you might need something like this: trans_angle = np.arctan(slope["val"]*m)*180/np.pi
        #trans_angle = 45*m
        # print("\t" + str(trans_angle) + "°")
        
        axis.annotate(val[0] + ": " + str(bandwidth) + " GB/s", pos,
            rotation=np.arctan(m/fig_ratio)*180/np.pi, rotation_mode='anchor',
            fontsize=fontsize_labels,
            ha="left", va='bottom',
            color=memory_bound_line[0].get_color())
         

        # plot the roof
        # Note that given x=(x0,x1) and y =(y0,y1), loglog plots a line between point (x0,y0) and (x1, y1)
        roof_xs = [x[1], xmax*10]
        roof_ys = [roof, roof]
        compute_bound_line = axis.loglog(roof_xs, roof_ys, linewidth=2.0,
            linestyle='-.',
            marker="2",
            zorder=10, color=memory_bound_line[0].get_color())
        
        axis.text(
            xmax/(10**(xlogsize*0.01)), roof*(10**(ylogsize*0.01)),
            val[0] + ": " + str(roof) + " GFLOPs",
            ha="right",
            fontsize=fontsize_labels,
            color=memory_bound_line[0].get_color())




    if apps != []:
        color = matplotlib.pyplot.cm.rainbow(numpy.linspace(0, 1, len(apps)))
        for idx, val in enumerate(apps):
           
            axis.axvline(apps_intensity[idx], label=val[0],
                            linestyle=':', color=color[idx], linewidth=2.0)
            if len(val) > 2:
                assert len(val) % 2 == 0
                for cnt in range(2, len(val), 2):
                    pair = [apps_intensity[idx], float(val[cnt+1])]
                    axis.plot(pair[0], pair[1], 'rx')
                    axis.annotate(val[cnt], xy=(pair[0], pair[1]),
                                    textcoords='data')

    # Set aspect
    axis.set_xlim([xmin, xmax])
    axis.set_ylim([ymin, ymax])
    axis.legend(fontsize=fontsize_labels)
    fig.tight_layout()
    set_size(fig_dimension*fig_ratio,fig_dimension, ax=axis)
    matplotlib.pyplot.savefig("roofline.png")
    matplotlib.pyplot.show()


def read_file(filename, row_len, csv_name, allow_variable_rows=False):
    """
    Reads CSV file and returns a list of row_len-ary tuples
    """
    assert isinstance(row_len, int)
    elements = []
    try:
        fname = filename if filename is not None else sys.stdin
        with open(fname, 'r', encoding='utf-8') as in_file:
            reader = csv.reader(in_file, dialect='excel')
            for row in reader:
                if not row[0].startswith('#'):
                    if not allow_variable_rows:
                        if len(row) != row_len:
                            print(f"Error: Each row in {csv_name} must be "
                                  f"contain exactly {row_len} entries!",
                                  file=sys.stderr)
                            sys.exit(1)
                        else:
                            assert len(row) >= row_len
                    element = tuple([row[0]] + row[1:])
                    elements.append(element)
    except IOError as ex:
        print(ex, file=sys.stderr)
        sys.exit(1)
    return elements


def main():
    """
    main function
    """
    hw_platforms = []
    apps = []
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", metavar="hw_csv",
                        help="HW platforms CSV file", type=str)
    parser.add_argument("-a", metavar="apps_csv",
                        help="applications CSV file", type=str)
    parser.add_argument("--xkcd", action='store_true', default=False)

    args = parser.parse_args()
    # HW
    print("Reading HW characteristics...")
    hw_platforms = read_file(args.i, 4, "HW CSV")
    # apps
    if args.a is None:
        print("No application file given...")
    else:
        print("Reading applications parameters...")
        apps = read_file(args.a, 2, "SW CSV", True)

    print(hw_platforms)
    print(f"Plotting using XKCD plot style is set to {args.xkcd}")
    if apps:
        print(apps)
    process(hw_platforms, apps, args.xkcd)
    sys.exit(0)


if __name__ == "__main__":
    main()
