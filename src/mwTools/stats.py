# Author: Marc Weber
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.sandbox.stats.multicomp import multipletests


def weighted_sample_mean_with_reliability_weights(x, sigma, verbose=0):
    """
    Compute the weighted mean and the standard deviation of the weighted
    mean from a list of data of potentially unequal reliability. We assume that
    the data $x_i$ are drawn from independent normal distribution with the same
    mean $\mu$ but different variance $\sigma_i^2$. In this case, it can be shown
    that the maximum likelihood estimator of the mean probability distribution [1]
    is given by the weighted mean

    $$ \bar x = \sum_{i=1}^{N} w_i x_i $$

    with the weights equal to the inverse of the variance of each data
    distribution,

    $$ w_i = \frac{\sum_j \sigma_j^2}{\sigma_i^2} $$

    where we normalize weights to unity. These weights are called *reliability
    weights* and intuitively give more weight to data point with smaller
    variance.

    The theoretical variance of the weighted mean is given by the variance of
    a linear combination of independent normal distributions,

    $$ \sigma_{\bar x}^2 = \sum_i w_i^2 \sigma_i^2 = \frac{1}{ \sum_i
    $$ \\frac{1}{\sigma_i^2} } $$

    where in this case the coefficient of the linear combination are
    themselves a function of the variances $\sigma_i$. However, in practice,
    the weighted mean is used to find the best estimator for the expected
    value and variance of a physical observable, based on independent
    measurements with variable precision and reliability. In such,
    experimental errors can be underestimated when calculating the error of
    each data point. Taking the expression above of the variance based on the
    values of the errors can thus underestimated the true variance of the
    observable. This can be easily illustrated if we take 3 measurements with
    standard deviations

    $$(x_i, \sigma_i) = (2, 0.1), (4.5, 2.0), (8, 0.1)$$

    The true value of the observable is 5, and while the error of the second
    measurement was correctly estimated with a standard error of 2, the errors
    of the first and last measurements were clearly underestimated at 0.1. The
    probability of obtaining both extreme measurements drawn from a normal
    distribution with mean 5.0 and standard deviation of 0.1 is extremely
    small. The theoretical standard deviation of the combined normal
    distributions is only 0.07, reflecting the fact that the real error on the
    measurements and thus on the weighted mean were underestimated. The
    variance must be corrected to take into account this fact, and can be
    scaled by the reduced chi-squared,

    $$ \hat\sigma_{\bar x}^2 = \sigma_{\bar x}^2 \cdot \frac{1}{n-1} \sum_i
    $$ \\frac{\left(x_i - \bar x\right)^2}{\sigma_i^2} $$

    In the case of reliability weights (as opposed to frequencies of
    observation weights), the weights are not random. The unbiased estimator
    for the variance of the weighted mean is given by [1,2]

    $$ s^2 = \frac{ \sum_i w_i }{ \left(\sum_i w_i\right)^2 - \sum_i w_i^2 }
    $$ \sum_i w_i \left(x_i - \mu^*\right)^2 $$

    In the same example as above, the estimator gives a standard error of 2.1,
    much more consistent with the distribution of our data points.

    References:

    [1] [Weighted arithmetic mean - Wikipedia - Dealing with variance](https:/
    [/en.wikipedia.org/wiki/Weighted_arithmetic_mean#Dealing_with_variance)

    [2] [Weighted arithmetic mean - Wikipedia - Reliability_weights](https://e
    [n.wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights)

    [3] [unbiased estimator - Bias correction in weighted variance - Cross
    Validated](https://stats.stackexchange.com/questions/47325/bias-correction-in-
    weighted-variance?newreg=097b2d4d156e47ea845a8e9ac1c97f13)
    """
    
    w = 1./(sigma**2)
    w_norm = sum(w)
    w = w/w_norm
    n = len(x)
    avg = np.average(x, weights=w)
    correctionFactor = (sum(w)) / (sum(w)**2 - sum(w**2))
    theoreticalVar = 1. / (sum(1./sigma**2))
    if n > 1:
        stddev = np.sqrt( correctionFactor * sum(w * ((x - avg)**2)) )
    elif n == 1:
        stddev = sigma[0]
    if verbose >= 2:
        print("x =", x)
        print("sigma =", sigma)
        print("w = ", w)
        print("correctionFactor =", correctionFactor)
        print("((x - avg)**2)) =", ((x - avg)**2))
        print("theoreticalVar =", theoreticalVar)
    return avg, stddev


def jensen_shannon_div(P, Q):
    """Compute the Jensen-Shannon divergence between two probability distributions.
    

    Input
    -----
    P, Q : array-like
        Probability distributions of equal length that sum to 1
    """

    def _kldiv(A, B):
        return np.sum([v for v in A * np.log2(A/B) if not np.isnan(v)])

    P = np.array(P)
    Q = np.array(Q)
    
    # Normalize both distributions
    P = P/sum(P)
    Q = Q/sum(Q)

    M = 0.5 * (P + Q)

    return 0.5 * (_kldiv(P, M) + _kldiv(Q, M))


def multiple_category_test(df, category, value, family_wise_FDR=0.05, orderList=None, verbose=0):
    """
    Wilcoxon-Mann-Whitney statistic on multiple categories.

    First we define the distribution of data in category A and notA. We apply
    a Wilcoxon-Mann-Whitney (WMM) U test to test significant difference in
    distribution of data value in group A versus in group notA. We apply this
    test iteratively to all categories. Then, we apply multiple tests
    correction with Benjamini-Hochberg (non-negative) method.

    See also:
    https://stats.stackexchange.com/questions/79843/is-the-w-statistic-output-by-wilcox-test-in-r-the-same-as-the-u-statistic
    """

    dfg = df.groupby(category)
    dfg.size()

    if orderList is None:
        catList = sorted(list(dfg.groups.keys()))
    else:
        if set(dfg.groups.keys()) != set(orderList):
            raise ValueError('The order list does not contain the same set of categories as the dataframe.')
        else:
            catList = orderList

    mwResultList = []
    for cat in catList:
        dfCat = df[df[category] == cat]
        dfNotCat = df[df[category] != cat]
        for direction, alternative in [('>', 'greater'), ('<', 'less')]:
            u_stat, p_val = stats.mannwhitneyu(dfCat[value].values, dfNotCat[value].values, alternative=alternative)
            if verbose >= 2:
                print (cat, "MWW RankSum P =", p_val, u_stat)
            mwResultList.append({'category':cat, 'WMM_pvalue':p_val, 'WMM_ustat':u_stat, 'direction':direction,
                                 'n_samples_in_cat':len(dfCat), 'n_samples_not_in_cat':len(dfNotCat)})

    mwResultDf = pd.DataFrame(mwResultList)
    if verbose >= 2: print(mwResultDf)

    # Multiple test correction
    # False discovery rate for the multiple tests correction
    pvalueList = mwResultDf['WMM_pvalue'].tolist()

    reject, pvals_corrected, alphacSidak, alphacBonf = \
        multipletests(pvalueList, alpha=family_wise_FDR, method='fdr_bh',
                      is_sorted=False, returnsorted=False)

    # The list of reject_null_hypotheses booleans has the same order as the original pvalue list,
    # so we assign the same index to `reject` dataframe.
    resultDf = pd.concat([mwResultDf,
                          pd.DataFrame(reject, index=mwResultDf.index,
                                       # columns=['multiple_tests_correction_FDR_' + str(family_wise_FDR)])],
                                       columns=['multiple_tests_correction'])],
                         axis=1)
    return resultDf


def enrichment_analysis(df, idCol, groupCol, groupValuesList, catCol, catList,
                        alpha_family_wise_error_rate, enrichmentTestcolumnName, verbose=0):

    """
    We group protein sequences by group (COG, orthogroup, etc) and compute the enrichment
    of all the protein subsets within a binary category 'cat'.
    """
    print("v1.2")
    
    groupsMultiIndex = pd.Index(groupValuesList, name='group')

    colName_count_cat_in_group = 'count_cat_in_group'
    colName_count_NOT_cat_in_group = 'count_NOT_cat_in_group'
    colName_count_cat_NOT_in_group = 'count_cat_NOT_in_group'
    colName_count_NOT_cat_NOT_in_group = 'count_NOT_cat_NOT_in_group'

    enrichMultiIndex = pd.MultiIndex.from_product([catList,
                                                  [colName_count_cat_in_group,
                                                   colName_count_NOT_cat_in_group,
                                                   colName_count_cat_NOT_in_group,
                                                   colName_count_NOT_cat_NOT_in_group,
                                                   'odds_ratio',
                                                   'pvalue_fisher',
                                                   enrichmentTestcolumnName
                                                   ]],
                                                  names=['cat', 'observable'])
    groupEnrichDf = pd.DataFrame(index=groupsMultiIndex, columns=enrichMultiIndex)
    groupEnrichDf = groupEnrichDf.sort_index(axis=0).sort_index(axis=1)


    for cat in catList:
        if verbose >= 1:
            print("#######", cat)

        subsetBoolDf = df[catCol] == cat
        proteinSubsetDf = df[subsetBoolDf]
        proteinNotSubsetDf = df[~subsetBoolDf]

        for groupIndex in groupValuesList:
            # Select one group and compute enrichment

            # Nb of proteins in cat that is included in the group.
            # *IMPORTANT*: We include all proteins that **contains as a string the group index**
            #              within the list of group values.
            # We use == True in order to set NaN values to False. Here we want a bi-partition
            # of all records, such that all records that are not in the group index, including
            # NaN values, belong to the second group.
            sel = (proteinSubsetDf[groupCol].str.contains(groupIndex) == True)
            NjA = len(proteinSubsetDf[sel].drop_duplicates(subset=[idCol]))

            # Nb of proteins in cat that is not included in the group.
            NjNotA = len(proteinSubsetDf[~sel].drop_duplicates(subset=[idCol]))

            # Nb of proteins in other cat that is included in the group.
            # We use == True in order to set NaN values to False.
            sel = (proteinNotSubsetDf[groupCol].str.contains(groupIndex) == True)
            NrefA = len(proteinNotSubsetDf[sel].drop_duplicates(subset=[idCol]))

            # Nb of proteins with other C-terminal AA that is not included in the group.
            NrefNotA = len(proteinNotSubsetDf[~sel].drop_duplicates(subset=[idCol]))

            contingencyTable = [[NjA, NrefA], [NjNotA, NrefNotA]]
            oddsRatio, pvalue = stats.fisher_exact(contingencyTable, alternative='two-sided')

            groupEnrichDf.loc[groupIndex, (cat, colName_count_cat_in_group)] = NjA
            groupEnrichDf.loc[groupIndex, (cat, colName_count_cat_NOT_in_group)] = NjNotA
            groupEnrichDf.loc[groupIndex, (cat, colName_count_NOT_cat_in_group)] = NrefA
            groupEnrichDf.loc[groupIndex, (cat, colName_count_NOT_cat_NOT_in_group)] = NrefNotA

            groupEnrichDf.loc[groupIndex, (cat, 'odds_ratio')] = oddsRatio
            groupEnrichDf.loc[groupIndex, (cat, 'pvalue_fisher')] = pvalue

        # Multiple test correction

        pvalueList = groupEnrichDf.loc[:, (cat, 'pvalue_fisher')].dropna()
        pvalueList.name = pvalueList.name[1]

        reject, pvals_corrected, alphacSidak, alphacBonf = \
            multipletests(pvalueList, alpha=alpha_family_wise_error_rate, method='fdr_bh',
                          is_sorted=False, returnsorted=False)

        # The list of reject_null_hypotheses booleans has the same order as the original pvalue list,
        # so we assign the same index to `reject` dataframe.
        pvalueListCorrected = pd.concat([pd.DataFrame(pvalueList),
                                         pd.DataFrame(reject, index=pvalueList.index, columns=[enrichmentTestcolumnName])],
                                        axis=1)
        pvalueListCorrected.index = pvalueListCorrected.index
        pvalueListCorrected.columns = pd.MultiIndex.from_product([[cat], pvalueListCorrected.columns])
        groupEnrichDf.loc[:, (cat, enrichmentTestcolumnName)] = pvalueListCorrected.loc[:, (cat, enrichmentTestcolumnName)]

    return groupEnrichDf


def pvalueAnnotation(pvalueList, pvalueThresholds=None, pvalueMask=None, directionList=None):
    if pvalueThresholds is None:
        pvalueThresholds = [[1.1,""], [1e-2,"**"], [1e-3,"***"], [1e-4,"****"]]

    if pvalueMask is None:
        pvalueMask = [True for _ in range(len(pvalueList))]

    if len(pvalueList) != len(pvalueMask):
        raise ValueError("Length of pvalueMask is different from length of pvalueList.")

    if type(pvalueList) is not pd.Series:
        pvalueS = pd.Series(pvalueList)
        pvalueMaskS = pd.Series(pvalueMask)
    else:
        pvalueS = pvalueList
        pvalueMaskS = pvalueMask
    

    # Build a table of text annotations representing pvalue

    # We create a copy of the data frame with string type (cannot mix types inside data frame columns)
    annotSeries = pvalueS.copy()
    annotSeries[:] = ""

    for i in range(0, len(pvalueThresholds)):
        if (i < len(pvalueThresholds) - 1):
            condition = ((pvalueS < pvalueThresholds[i][0]) &
                         (pvalueThresholds[i+1][0] < pvalueS) &
                         pvalueMaskS)
            annotSeries[condition] = pvalueThresholds[i][1]
        else:
            condition = ((pvalueS < pvalueThresholds[i][0]) &
                         pvalueMaskS)
            annotSeries[condition] = pvalueThresholds[i][1]

    annotSeriesCopy = annotSeries.copy()
    if directionList is not None:
        for index, annot in annotSeries.iteritems():
            if annot != "":
                annotSeriesCopy.loc[index] = directionList[index] + annot
        annotSeries = annotSeriesCopy

    return annotSeries


def quantile_normalize(df_input):
    df = df_input.copy()
    # compute rank
    dic = {}
    for col in df:
        dic.update({col : sorted(df[col])})
    sorted_df = pd.DataFrame(dic)
    rank = sorted_df.mean(axis=1).tolist()
    #sort
    for col in df:
        t = np.searchsorted(np.sort(df[col]), df[col])
        df[col] = [rank[i] for i in t]
    return df
