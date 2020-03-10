from matplotlib.text import Text
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import transforms
from matplotlib.transforms import blended_transform_factory
import matplotlib.patches as mpatches
import seaborn
from mwTools.stats import pvalueAnnotation
from mwTools.stats import multiple_category_test
import pandas as pd
import numpy as np
from numpy import log10
from scipy import stats
from scipy.stats import gaussian_kde



def background_gradient(valueList, cmap='PuBu', vmin=0, vmax=1):
    """
    Example::
        >>> background_gradient([-3, 0, 3], cmap=colormap, vmin=-3, vmax=+3)
        ['background-color: #09386d',
         'background-color: #f7f6f5',
         'background-color: #730421']
    """
    norm = colors.Normalize(vmin, vmax)
    normed = norm(valueList)
    c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
    return ['background-color: %s' % color for color in c]


def convert_colormap_to_hex(cmap, x, vmin=0, vmax=1):
    """
    Example::
        >>> seaborn.palplot(seaborn.color_palette("RdBu_r", 7))
        >>> colorMapRGB = seaborn.color_palette("RdBu_r", 61)
        >>> colormap = seaborn.blend_palette(colorMapRGB, as_cmap=True, input='rgb')
        >>> [convert_colormap_to_hex(colormap, x, vmin=-2, vmax=2) for x in range(-2, 3)]
        ['#09386d', '#72b1d3', '#f7f6f5', '#e7866a', '#730421']
    """
    norm = colors.Normalize(vmin, vmax)
    color_rgb = plt.cm.get_cmap(cmap)(norm(x))
    color_hex = colors.rgb2hex(color_rgb)
    return color_hex


def get_divergent_color_map(name="RdBu_r"):
    colorMapRGB = seaborn.color_palette(name, 61)
    colormap = seaborn.blend_palette(colorMapRGB, as_cmap=True, input='rgb')
    return colormap


def update_label(old_label, exponent_text):
    """Copied from: https://ga7g08.github.io/2015/07/22/Setting-nice-axes-labels-in-matplotlib/"""
    if exponent_text == "":
        return old_label
    
    try:
        units = old_label[old_label.index("[") + 1:old_label.rindex("]")]
    except ValueError:
        units = ""
    label = old_label.replace("[{}]".format(units), "")
    
    exponent_text = exponent_text.replace("\\times", "")
    
    return "{} [{} {}]".format(label, exponent_text, units)
    

def format_label_string_with_exponent(ax, axis='both'):
    """
    Format the label string with the exponent from the ScalarFormatter
    Copied from: https://ga7g08.github.io/2015/07/22/Setting-nice-axes-labels-in-matplotlib/
    """
    ax.ticklabel_format(axis=axis, style='sci')

    axes_instances = []
    if axis in ['x', 'both']:
        axes_instances.append(ax.xaxis)
    if axis in ['y', 'both']:
        axes_instances.append(ax.yaxis)
    
    for ax in axes_instances:
        ax.major.formatter._useMathText = True
        plt.draw()     # Update the text
        exponent_text = ax.get_offset_text().get_text()
        print(exponent_text)
        label = ax.get_label().get_text()
        print(label)
        ax.offsetText.set_visible(False)
        ax.set_label_text(update_label(label, exponent_text))


def boxplot_rel_widths(df, x, y, widthScale=5):
    sampleSizeS = df.groupby(x).size()
    sampleSizeS.name = 'n'
    sampleSizeS = sampleSizeS.to_frame()
    sampleSizeS['rel_n'] = sampleSizeS['n'] / len(df)

    ax = df.boxplot(column=y, by=x, widths=(widthScale*sampleSizeS['rel_n']).tolist())
    ax.set_xticklabels(['%s\n$n$=%d' % (k, len(v)) for k, v in df.groupby(x)])
    plt.xticks(rotation='vertical')
    return ax


def multiple_category_boxplot(df, category, value, family_wise_FDR=0.05, orderList=None,
                              plotPvalues='annotations', plotPvaluesLim=None,
                              verbose=0, **kwargs):

    dfg = df.groupby(category)

    if orderList is None:
        catList = sorted(list(dfg.groups.keys()))
    else:
        if set(dfg.groups.keys()) != set(orderList):
            raise ValueError('The order list does not contain the same set of categories as the dataframe.')
        else:
            catList = orderList

    # Add another category with all the samples
    dfAll = df.copy()
    dfAll[category] = 'all'
    # Add dummy record to create empty space
    dfAll.append({category:'', value:0}, ignore_index=True)
    dfPlot = pd.concat([df, dfAll])
    catListPlot = catList + ['', 'all']

    multipleTestDf = multiple_category_test(df, category, value, orderList=catList, verbose=verbose)

    if plotPvalues == 'scatter':
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8,6),
                                 gridspec_kw={'height_ratios': [1, 3.5], 'hspace':0.05})
        ax = axes[1]
    elif plotPvalues == 'annotations':
        fig, axes = plt.subplots(1)
        ax = axes

    seaborn.boxplot(ax=ax, data=dfPlot, x=category, y=value, order=catListPlot, fliersize=2, **kwargs)

    # We keep only one row per category, the one for which the statistical test is significant,
    # in any of the two directions (smaller or greater)
    multipleTestDf2 = multipleTestDf.sort_values(['category', 'multiple_tests_correction_FDR_0.05'], ascending=False)\
                                    .drop_duplicates(subset='category')
    # We have to sort the df following the order list again
    multipleTestDf2 = multipleTestDf2.set_index('category').loc[catList].reset_index()
    if verbose >= 2: print("multipleTestDf2", multipleTestDf2)


    # We use the multiple tests correction as a mask to write pvalue annotations
    # (only positive tests are annotated)
    pvalueAnnotS = pvalueAnnotation(multipleTestDf2['WMM_pvalue'], pvalueThresholds=None,
                                    pvalueMask=multipleTestDf2['multiple_tests_correction_FDR_0.05'],
                                    directionList=multipleTestDf2['direction'])
    pvalueAnnotS.reset_index(inplace=True, drop=True)
    if verbose >= 2: print("pvalueAnnotS", pvalueAnnotS)

    if plotPvalues == 'annotations':
        tform = blended_transform_factory(ax.transData, ax.transAxes)
        for index, text in pvalueAnnotS.iteritems():
            ax.annotate(text, xy=(int(index), 1.02), xycoords=tform, ha='center', va='bottom',
                        fontweight='bold', fontsize='x-large', rotation=90, annotation_clip=False)

    if plotPvalues == 'scatter':
        pvalPlotDf = pd.DataFrame(data=np.array(range(len(pvalueAnnotS))), columns=['x'])
        pvalPlotDf['y'] = -10*log10(multipleTestDf2['WMM_pvalue'])
        directionS = multipleTestDf2['direction'].map(lambda x: +1 if x == '>' else -1)
        pvalPlotDf['y'] = directionS*pvalPlotDf['y']
        pvalPlotDf['significance'] = multipleTestDf2['multiple_tests_correction_FDR_0.05']
        current_palette = seaborn.color_palette()
        col1 = current_palette[2]
        col2 = current_palette[1]
        color = [col1 if significant else col2 for significant in
                 multipleTestDf2['multiple_tests_correction_FDR_0.05'].tolist()]
        pvalPlotDf['color'] = color
        yLabel = "-10*log(pvalue)\n*sign(test)"
        pvalPlotDf.rename(columns={'y':yLabel}, inplace=True)
        ax = axes[0]
        df = pvalPlotDf[pvalPlotDf['significance']]
        ax.scatter(df['x'], df[yLabel], color=col1, label='significant')
        df = pvalPlotDf[~pvalPlotDf['significance']]
        ax.scatter(df['x'], df[yLabel], color=col2, label='not significant')
        ax.set_ylim((1.4*pvalPlotDf[yLabel].min(), 1.4*pvalPlotDf[yLabel].max()))
        ax.set_ylabel(yLabel)
        if plotPvaluesLim is not None:
            ax.set_ylim(plotPvaluesLim)
        ax.legend(title='Multiple tests correction:', labelspacing=0.1, handletextpad=0.1)
        # seaborn.lmplot(data=pvalPlotDf, x='x', y=yLabel, fit_reg=False, hue="significance",
        #                palette=current_palette[1:3], scatter_kws={"marker": "D", "s": 100})

    return axes


def seaborn_pairplot_add_diagonal_line(g, ls="--", lw=1, c="0.3", **kwargs):
    
    nRow = g.axes.shape[0]
    nCol = g.axes.shape[1]

    for i in range(nRow):
        for j in range(nCol):
            if i != j:
                ax = g.axes[i][j]
                ax.plot(ax.get_xlim(), ax.get_ylim(), ls=ls, lw=lw, c=c, **kwargs)
    # return plt.gcf()


def pvalAnnotation_text(x, pvalueThresholds):
    singleValue = False
    if type(x) is np.array:
        x1 = x
    else:
        x1 = np.array([x])
        singleValue = True
    # Sort the threshold array
    pvalueThresholds = pd.DataFrame(pvalueThresholds).sort_values(by=0, ascending=False).values
    xAnnot = pd.Series(["" for _ in range(len(x1))])
    for i in range(0, len(pvalueThresholds)):
        if (i < len(pvalueThresholds)-1):
            condition = (x1 <= pvalueThresholds[i][0]) & (pvalueThresholds[i+1][0] < x1)
            xAnnot[condition] = pvalueThresholds[i][1]
        else:
            condition = x1 < pvalueThresholds[i][0]
            xAnnot[condition] = pvalueThresholds[i][1]

    return xAnnot if not singleValue else xAnnot.iloc[0]


# 2020.01.07 DEPRECATED by statannot package
# def seaborn_boxplot_add_statistical_test_annotation(
#         ax, df, xlabel, ylabel, catPairList, test='Mann-Whitney', order=None,
#         textFormat='full',
#         pvalueThresholds=[[1.1,""], [0.05,"*"], [1e-2,"**"], [1e-3,"***"], [1e-4,"****"]],
#         color='0.2', lineYOffsetAxesCoord=0.02, lineHeightAxesCoord=0.02, yTextOffsetPoints=2,
#         linewidth=1.5, fontsize='large', verbose=1):

#     """
#     TODO:
#         + print NS for not significant instead of nothing
#         + print pvalue legend (thresholds)
#     """

#     yStack = [-1e9]
#     g = df.groupby(xlabel)
#     catValues = df[xlabel].unique()
#     if order is not None:
#         catValues = order

#     for cat1, cat2 in catPairList:
#         if cat1 in catValues and cat2 in catValues:
#             # Get position of bars 1 and 2
#             x1 = np.where(catValues == cat1)[0][0]
#             x2 = np.where(catValues == cat2)[0][0]
#             cat1YMax = g[ylabel].max()[cat1]
#             cat2YMax = g[ylabel].max()[cat2]
#             cat1Values = g.get_group(cat1)[ylabel].values
#             cat2Values = g.get_group(cat2)[ylabel].values

#             testShortName = ''
#             if test == 'Mann-Whitney':
#                 u_stat, pval = stats.mannwhitneyu(cat1Values, cat2Values, alternative='two-sided')
#                 testShortName = 'M.W.W.'
#                 if verbose >= 2: print ("MWW RankSum P_val=", pval, "U_stat=", u_stat)
#             elif test == 't-test':
#                 stat, pval = stats.ttest_ind(a=cat1Values, b=cat2Values)
#                 testShortName = 't-test'
#                 if verbose >= 2: print ("t-test independent samples, P_val=", pval, "stat=", stat)

#             if textFormat == 'full':
#                 text = "{} p < {:.2e}".format(testShortName, pval)
#             elif textFormat is None:
#                 text = ''
#             elif textFormat is 'asterix':
#                 text = pvalAnnotation_text(pval, pvalueThresholds)
            
#             yRef = max(max(cat1YMax, cat2YMax), max(yStack))
#             yLim = ax.get_ylim()
#             yRange = yLim[1] - yLim[0]
#             yOffset = lineYOffsetAxesCoord*yRange
#             y = yRef + yOffset
#             h = lineHeightAxesCoord*yRange
#             # axisToData = ax.transAxes + ax.transData.inverted()
#             # y = axisToData.transform([(0, 1)])[0][1]
#             ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=linewidth, c=color)
#             ann = ax.annotate(text, xy=(np.mean([x1, x2]), y + h), xytext=(0, yTextOffsetPoints),
#                               textcoords='offset points', ha='center', va='bottom', fontsize=fontsize)
#             plt.gcf().canvas.draw()
#             # patch = ann.get_bbox_patch()
#             # box = patch.get_extents()
#             box = Text.get_window_extent(ann)
#             tcbox = ax.transData.inverted().transform(box)
#             yTopAnnot = tcbox[1][1]
#             yStack.append(yTopAnnot)
    
#     return ax


def seaborn_boxplot_add_number_samples(
        ax, df, xlabel, ylabel, order=None, position='top', yPosAxes=0.9,
        fontsize='large', color='black', text_rotation=0, verbose=1):

    if order is None:
        order = df[xlabel].unique()
    if True in order:
        # For some reason seaborn always sorts False, True.
        order = [False, True]

    # Calculate number of obs per group & median to position labels
    medians = df.groupby([xlabel])[ylabel].median().loc[order].values

    # There is a problem where the categories are True and False, because there is no way
    # to work around the boolean indexing in Pandas. True and False are considered as
    # row indexing selectors, not as values as python objects.
    if verbose>=2: print("order", order)
    if True in order:
        nobs = df[xlabel].value_counts()
        nobs2 = nobs.copy()
        if order[0] == True:
            nobs2.iloc[0] = nobs.loc[True]
            nobs2.iloc[1] = nobs.loc[False]
        elif order[0] == False:
            nobs2.iloc[0] = nobs.loc[False]
            nobs2.iloc[1] = nobs.loc[True]
        nobs = nobs2
    else:
        nobs = df[xlabel].value_counts().loc[order].values
    nobs = [str(x) for x in nobs.tolist()]
    nobs = ["N: " + i for i in nobs]

    # Add it to the plot
    pos = range(len(nobs))
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

    for tick, label in zip(pos, ax.get_xticklabels()):
        if position == 'median':
            ax.annotate(nobs[tick], xy=(pos[tick], medians[tick]), xycoords='data',
                        xytext=(0, 2), textcoords='offset points',
                        ha='center', fontsize=fontsize, color=color, rotation=text_rotation)
        elif position == 'top':
            ax.annotate(nobs[tick], xy=(pos[tick], yPosAxes), xycoords=trans, ha='center',
                        color=color, fontsize=fontsize, rotation=text_rotation)

    return ax


def compute_scatter_density(plotDf, xlabel, ylabel, ax):

    x = plotDf[xlabel].values
    y = plotDf[ylabel].values

    # Calculate the point density, IN THE SCALING SPACE
    x_transform = ax.xaxis.get_transform().transform
    x2 = x_transform(x)
    y_transform = ax.yaxis.get_transform().transform
    y2 = y_transform(y)
    xy = np.array([x2, y2])

    plotDf['density'] = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    plotDf = plotDf.sort_values(by='density')
    # Recompute x, y, and z with new order
    x = plotDf[xlabel].values
    y = plotDf[ylabel].values
    xyPoints = np.array(list(zip(x, y)))
    z = plotDf['density']

    return plotDf, x, y, z


def draw_color_gradient(ax, start_color, end_color, path, direction='down', hide_axis=True, extent=(0, 1, 0, 1), alpha=1):
    """Example:

    start_color = mcolors.hex2color(mcolors.cnames["dodgerblue"])
    end_color = mcolors.to_rgb('1')
    fig, ax = plt.subplots()
    path = mpatches.Path([[0,0],[0,1],[1,0],[0,0]])
    draw_color_gradient(ax, start_color, end_color, path, direction='down', hide_axis=True, alpha=0.5)
    """
    if direction == 'down':
        img = [[start_color], [end_color]]
    elif direction == 'up':
        img = [[end_color], [start_color]]
    elif direction == 'right':
        img = [[start_color, end_color]]
    elif direction == 'left':
        img = [[end_color, start_color]]

    patch = mpatches.PathPatch(path, facecolor='none', ec='none', clip_on=False)
    ax.add_patch(patch)
    ax.imshow(img, interpolation='bilinear', extent=extent, clip_path=patch, clip_on=True,
              origin='upper', aspect='auto', alpha=alpha)
    plt.gcf().set_facecolor('white')
    if hide_axis:
        ax.axis('off')
    return ax
