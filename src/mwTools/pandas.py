# coding: utf-8 
# Author: Marc Weber

"""
=========================================================================================
Pandas
=========================================================================================
"""

import numpy as np
import pandas as pd
from numpy import random
from collections import OrderedDict



def split_and_stack_column(df, col, colFieldSep=' ', chunkSize=2000, copy_df=True):
    """
    Split the strings in a Dataframe column and stack the resulting list of strings into new rows.
    Uses a memory efficient method.

    We tested the method for a dataframe with 50'000 rows and an average of 31 values to be splitted per row
    (number of values per row is very variable, up to 350).
    The results in terms of performance and memory usage are approximately the following::

        chunkSize = 100
        1 loop, best of 1: 1min 37s per loop
        peak memory: 784.55 MiB, increment: 87.39 MiB

        chunkSize = 500
        1 loop, best of 1: 1min 1s per loop
        peak memory: 801.57 MiB, increment: 100.66 MiB

        chunkSize = 1000
        1 loop, best of 1: 49.3 s per loop
        peak memory: 969.35 MiB, increment: 268.44 MiB

        chunkSize = 2000
        1 loop, best of 1: 41.8 s per loop
        peak memory: 1289.91 MiB, increment: 624.57 MiB

        chunkSize = 5000
        1 loop, best of 1: 40.9 s per loop
        peak memory: 2216.74 MiB, increment: 1551.89 MiB

        chunkSize = 10000
        1 loop, best of 1: 49.2 s per loop
        peak memory: 3806.75 MiB, increment: 3141.65 MiB

        chunkSize = 25000
        1 loop, best of 1: 1min 14s per loop
        peak memory: 8580.21 MiB, increment: 7915.12 MiB

    In this case, there is an optimum in performance using a chunksize of approx. 5000 rows
    (5000*31 = 155'000 splitted elements in a chunk in average).
    
    See also: http://stackoverflow.com/questions/33622470/fast-way-to-split-column-into-multiple-rows-in-pandas
    See also: http://stackoverflow.com/questions/17116814/pandas-how-do-i-split-text-in-a-column-into-multiple-rows
    """

    chunkDfList = []

    # In order to split the dataframe, we can use groupby or the numpy array_split method.
    nChunks = len(df) // chunkSize
    # np.array_split method:
    for g, chunkDf in enumerate(np.array_split(df, nChunks)):
    # pandas groupby method:
    # for g, chunkDf in df.groupby(np.arange(nChunks):

        print("split_and_stack_column, chunk ", g, " / ", nChunks)
        chunkStackedDf = chunkDf[col].str.split(colFieldSep, expand=True).stack()
        chunkStackedDf = chunkStackedDf.str.strip()
        chunkStackedDf.index = chunkStackedDf.index.droplevel(-1)
        chunkDfList.append(chunkStackedDf)

    stackedDf = pd.DataFrame(pd.concat(chunkDfList), columns=[col])

    # Merge the mapping with the dataframe by using original indexes
    dfCopy = df.copy() if copy_df else df
    dfCopy.drop(col, axis=1, inplace=True)
    dfCopy = pd.merge(dfCopy, stackedDf, how='left', left_index=True, right_index=True)

    return dfCopy


"""
def test_split_and_stack_column():
    # Test the performance and memory usage of the split_and_stack_column method
    # we use the eggMember dataframe from EggNOG database

    # To run in a ipython session
    %%capture myout

    # Import a dataframe that has a column containing multiple ids in a single string, separate by commas.
    # Each string contains between 1-359 ids, in average 31.
    eggMemberDf = pd.read_csv('/users/lserrano/mweber/Databases/EggNOG4.5/eggnogdb.embl.de/download/eggnog_4.5/data/bactNOG/bactNOG.members.tsv',
                              sep='\t', header=None,
                              names='TaxonomicLevel|GroupName|ProteinCount|SpeciesCount|COGFunctionalCategory|ProteinIDs'.split('|'))
    testDf = eggMemberDf[:50000].copy()

    chunkSize = 100
    %timeit -n1 -r1 split_and_stack_column(testDf, col='ProteinIDs', colFieldSep=',', chunkSize=chunkSize, copy_df=True)
    %memit -r1 split_and_stack_column(testDf, col='ProteinIDs', colFieldSep=',', chunkSize=chunkSize, copy_df=True)
    print(chunkSize)
    gc.collect()

    chunkSize = 500
    %timeit -n1 -r1 split_and_stack_column(testDf, col='ProteinIDs', colFieldSep=',', chunkSize=chunkSize, copy_df=True)
    %memit -r1 split_and_stack_column(testDf, col='ProteinIDs', colFieldSep=',', chunkSize=chunkSize, copy_df=True)
    print(chunkSize)
    gc.collect()

    chunkSize = 1000
    %timeit -n1 -r1 split_and_stack_column(testDf, col='ProteinIDs', colFieldSep=',', chunkSize=chunkSize, copy_df=True)
    %memit -r1 split_and_stack_column(testDf, col='ProteinIDs', colFieldSep=',', chunkSize=chunkSize, copy_df=True)
    print(chunkSize)
    gc.collect()

    chunkSize = 2000
    %timeit -n1 -r1 split_and_stack_column(testDf, col='ProteinIDs', colFieldSep=',', chunkSize=chunkSize, copy_df=True)
    %memit -r1 split_and_stack_column(testDf, col='ProteinIDs', colFieldSep=',', chunkSize=chunkSize, copy_df=True)
    print(chunkSize)
    gc.collect()


    chunkSize = 5000
    %timeit -n1 -r1 split_and_stack_column(testDf, col='ProteinIDs', colFieldSep=',', chunkSize=chunkSize, copy_df=True)
    %memit -r1 split_and_stack_column(testDf, col='ProteinIDs', colFieldSep=',', chunkSize=chunkSize, copy_df=True)
    print(chunkSize)
    gc.collect()

    chunkSize = 10000
    %timeit -n1 -r1 split_and_stack_column(testDf, col='ProteinIDs', colFieldSep=',', chunkSize=chunkSize, copy_df=True)
    %memit -r1 split_and_stack_column(testDf, col='ProteinIDs', colFieldSep=',', chunkSize=chunkSize, copy_df=True)
    print(chunkSize)
    gc.collect()

    chunkSize = 25000
    %timeit -n1 -r1 split_and_stack_column(testDf, col='ProteinIDs', colFieldSep=',', chunkSize=chunkSize, copy_df=True)
    %memit -r1 split_and_stack_column(testDf, col='ProteinIDs', colFieldSep=',', chunkSize=chunkSize, copy_df=True)
    print(chunkSize)
    gc.collect()
"""

"""
Results:

chunkSize = 100
1 loop, best of 1: 1min 37s per loop
peak memory: 784.55 MiB, increment: 87.39 MiB

chunkSize = 500
1 loop, best of 1: 1min 1s per loop
peak memory: 801.57 MiB, increment: 100.66 MiB

chunkSize = 1000
1 loop, best of 1: 49.3 s per loop
peak memory: 969.35 MiB, increment: 268.44 MiB

chunkSize = 2000
1 loop, best of 1: 41.8 s per loop
peak memory: 1289.91 MiB, increment: 624.57 MiB

chunkSize = 5000
1 loop, best of 1: 40.9 s per loop
peak memory: 2216.74 MiB, increment: 1551.89 MiB

chunkSize = 10000
1 loop, best of 1: 49.2 s per loop
peak memory: 3806.75 MiB, increment: 3141.65 MiB

chunkSize = 25000
1 loop, best of 1: 1min 14s per loop
peak memory: 8580.21 MiB, increment: 7915.12 MiB

There is an optimum at around a chunksize of 5000.
"""


def split_and_stack_column_from_file(filename, col, colFieldSep=' ', chunkSize=100000, **read_csv_kwargs):
    """
    Split the strings in a Dataframe column and stack the resulting list of strings into new rows.
    Reads the csv file by chunks, reducing memory usage.
    
    TODO: THIS METHOD NEEDS TO BE FINISHED.

    See also: http://pandas.pydata.org/pandas-docs/stable/io.html#iterating-through-files-chunk-by-chunk
    """
    print(read_csv_kwargs)

    chunkStackedDfList = []
    for chunkDf in pd.read_csv(filename, chunksize=chunkSize, **read_csv_kwargs):

        splittedDf = pd.DataFrame(chunkDf[col].str.split(colFieldSep).tolist())
        print(splittedDf)
        splittedDf = splittedDf.stack()
        splittedDf.index = splittedDf.index.droplevel(-1)
        splittedDf = splittedDf.str.strip()
        print(splittedDf)
        chunkStackedDfList.append(splittedDf)
        # In this case the index is starting from 0 at each chunk!!!!!!!!!!!!!!!!!!!!!!
        # This is a problem if we are reading the DF in chunks...

    print(pd.concat(chunkStackedDfList))
    return

    # # Merge the mapping with the dataframe by using original indexes
    # ....
    # df.drop(col, axis=1, inplace=True)
    # df = pd.merge(df, splittedDf, how='left', left_index=True, right_index=True)
    
    # return df


def sort_df_by_value_list(df, col, valueList):
    sorterIndex = dict(zip(valueList, range(len(valueList))))
    ranInt = random.random_integers(1e6)
    rank = df[col].map(sorterIndex)
    rank.name = 'rank{:d}'.format(ranInt)
    return df.join(rank).sort_values(rank.name).drop(rank.name, axis=1)


def sort_df(df, col, key=None, reverse=False):
    """
    Take into account duplicated entries by using a rank column and applying the sort_df_by_value_list
    method.

    Example, sorting by alphabetical order grouping together capital and small case, with capital case first:
    sort_df(df, 'col', key=lambda x: (x.upper(), x[0].islower()))
    """

    def sorter(key=None, reverse=False):
        """
        Using python built-in sorted function.
        """
        def sorter_(series):
            series_list = list(series)
            return [series_list.index(i)
                    for i in sorted(series_list, key=key, reverse=reverse)]
        return sorter_

    df2 = df.reset_index(drop=True)
    sortedIndex = sorter(key=key, reverse=reverse)(df2[col])
    sortedValuesUnique = list(OrderedDict.fromkeys(np.array([df2[col].iloc[i] for i in sortedIndex])))
    return sort_df_by_value_list(df2, col=col, valueList=sortedValuesUnique)


def rename_unnamed(df):
    """Rename unamed columns name for Pandas DataFrame

    See https://stackoverflow.com/questions/41221079/rename-multiindex-columns-in-pandas
    """
    for i, columns in enumerate(df.columns.levels):
        columns_new = columns.tolist()
        print(columns_new)
        for j, col in enumerate(columns_new):
            if type(col) is str:
                if "Unnamed: " in col:
                    columns_new[j] = ""
        if pd.__version__ < "0.21.0":  # https://stackoverflow.com/a/48186976/716469
            df.columns.set_levels(columns_new, level=i, inplace=True)
        else:
            df = df.rename(columns=dict(zip(columns.tolist(), columns_new)),
                           level=i)
    return df
