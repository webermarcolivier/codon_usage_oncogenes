import argparse
from collections import Counter
import pandas as pd
from numpy import log, exp
from mwTools.bio import extract_codons_list
from mwTools.stats import jensen_shannon_div



def split_6_codons_synonymous_families(aaCodonDict):
    # Split codon family that are encoded by two codon groups. These should be considered
    # as two different codon families as tRNA adaptation can only occur within a codon group.
    # See Xia2007 paper.
    aaCodonDict2 = aaCodonDict
    aaCodonDict = aaCodonDict2.copy()
    if 'R' in aaCodonDict2.keys():
        if set(aaCodonDict2['R']) == set(['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG']):
            aaCodonDict.pop('R')
            aaCodonDict['R1'] = ['CGT', 'CGC', 'CGA', 'CGG']
            aaCodonDict['R2'] = ['AGA', 'AGG']

    if 'L' in aaCodonDict2.keys():
        if set(aaCodonDict2['L']) == set(['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG']):
            aaCodonDict.pop('L')
            aaCodonDict['L1'] = ['CTT', 'CTC', 'CTA', 'CTG']
            aaCodonDict['L2'] = ['TTA', 'TTG']

    if 'S' in aaCodonDict2.keys():
        if set(aaCodonDict2['S']) == set(['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC']):
            aaCodonDict.pop('S')
            aaCodonDict['S1'] = ['TCT', 'TCC', 'TCA', 'TCG']
            aaCodonDict['S2'] = ['AGT', 'AGC']

    return aaCodonDict



def compute_codon_usage(seq, aaCodonDict, correctWijZeroCounts=0.01, checkLengthMultipleOf3=True, verbose=0):

    if type(seq) is pd.Series:
        seqList = seq.tolist()
    elif type(seq) is list:
        seqList = seq
    else:
        seqList = [seq]

    # Concatenate list of codons for all sequences
    codonListFinal = []
    for seq in seqList:
        codonList = extract_codons_list(seq, checkLengthMultipleOf3=checkLengthMultipleOf3)
        if codonList is not None:
            for codon in codonList:
                codonListFinal.append(codon)
    codonList = codonListFinal
    codonCount = Counter(codonList)
    codonFreq = {codon:count/sum(codonCount.values()) for codon, count in codonCount.items()}
    
    # Compute f_ij, frequency of codon j in synonymous family i
    synCodonCountDict = dict()
    fijDict = dict()
    wijDict = dict()
    for aa, synCodonList in aaCodonDict.items():
        if verbose >= 2: print(aa, synCodonList)
        synCodonCount = {synCodon:codonCount[synCodon] for synCodon in synCodonList}
        synCodonCountDict.update({aa: synCodonCount})
        if verbose >= 2: print(synCodonCount)
        
        if sum(synCodonCount.values()) > 0:
            fij = {codon:count/sum(synCodonCount.values()) for codon, count in synCodonCount.items()}
        else:
            fij = None
        fijDict.update({aa:fij})
        if verbose >= 2: print(fij)

        if fij is not None:
            wij = {codon:(freq/max(fij.values())) for codon, freq in fij.items()}
            if correctWijZeroCounts > 0:
                wij = {codon:(w if w > 0 else correctWijZeroCounts) for codon, w in wij.items()}
        else:
            wij = None
        wijDict.update({aa:wij})
        if verbose >= 2: print(wij)
        
    i = 1
    codonDatabase = ''
    for codon, freq in codonFreq.items():
        codonDatabase += '{} {:f}({:d})'.format(codon, freq, codonCount[codon])
        if i % 4 == 0:
            codonDatabase += '\n'
        else:
            codonDatabase += ' '
        i += 1
    
    return {'codonFreq':codonFreq, 'synCodonCount':synCodonCountDict,
            'fij':fijDict, 'wij':wijDict, 'codonDatabase':codonDatabase, 'codonCount':codonCount}


def compute_codon_usage_dict(seq, aaCodonDict, checkLengthMultipleOf3, verbose):
    dic = compute_codon_usage(seq, aaCodonDict, checkLengthMultipleOf3, verbose)['fij']
    dic2 = pd.Series({(outerKey, innerKey): values
                      for outerKey, innerDict in dic.items() if innerDict is not None
                      for innerKey, values in innerDict.items()})
    return dic2


def compute_codon_usage_series(seqQueryS, aaCodonDict, correctWijZeroCounts=0.01, checkLengthMultipleOf3=True, verbose=0):
    
    if type(seqQueryS) is not pd.Series:
        if verbose >= 2: print("seqQueryS not series.")
        seqQueryS = pd.Series(seqQueryS)
        
    codonDf = seqQueryS.apply(lambda seq: compute_codon_usage_dict(seq, aaCodonDict, checkLengthMultipleOf3, verbose))
    return codonDf


def _compute_metric_seq(seq, fijRef, aaCodonDict, checkLengthMultipleOf3=True, verbose=1):
    """
    Compute a single metric value for one query sequence against one reference codon usage.
    """

    codonUsage = compute_codon_usage(seq, aaCodonDict=aaCodonDict,
                                     checkLengthMultipleOf3=checkLengthMultipleOf3, verbose=1)
    fij = codonUsage['fij']
    if verbose >= 2: print("### _compute_metric_seq")
    if verbose >= 2: print("fij", fij)
    if verbose >= 2: print("fijRef", fijRef)

    # We drop out the aa families for which have no codons either
    # in the query sequence or in the ref. sequence.
    # for these we add 0.5, which would be the expected distance for
    # random sequences (?)
    metric = 0.0
    for aa, f in fij.items():
        fRef = fijRef[aa]
        if f is None or fRef is None:
            metric += 0.5
        else:
            fVector = [f[key] for key in f.keys() & fRef.keys()]
            fRefVector = [fRef[key] for key in f.keys() & fRef.keys()]
            if verbose >= 2: print("fVector", fVector)
            if verbose >= 2: print("fRefVector", fRefVector)
            JSD = jensen_shannon_div(fVector, fRefVector)
            if verbose >= 2: print("JSD", JSD)
            metric += sum(fVector)*JSD
    if verbose >= 2: print("metric", metric)

    return metric


def _compute_CAI(seq, wijRef, aaList, aaCodonDict, checkLengthMultipleOf3=True, verbose=1):
    """
    Compute a single CAI value for one query sequence against one reference codon usage.
    """

    codonUsage = compute_codon_usage(seq, aaCodonDict=aaCodonDict,
                                     checkLengthMultipleOf3=checkLengthMultipleOf3, verbose=1)
    fij = codonUsage['fij']
    if verbose >= 2: print("### _compute_CAI")
    if verbose >= 2: print("fij:", fij)
    if verbose >= 2: print("wijRef:", wijRef)

    sum1 = 0
    for aa in aaList:
        # This should not be necessary, we already filtered above for aa families with 0 counts in the ref. set.
        # if wijRef[aa] is not None:
        if True:
            for codon in wijRef[aa].keys():
                # We exclude codons which have 0 counts in the ref. set.
                if fij[aa] is not None and wijRef[aa][codon] > 0:
                    if verbose >= 2: print(aa, codon, "fij:", fij[aa][codon], "wij:", wijRef[aa][codon], "ln(wij)", log(wijRef[aa][codon]))
                    sum1 += fij[aa][codon]*log(wijRef[aa][codon])
                else:
                    if verbose  >= 2: print("skipped codon", aa, codon)
    if verbose >= 2: print("sum1:", sum1)

    sum2 = 0
    for aa in aaList:
        # if wijRef[aa] is not None:
        if True:
            for codon in wijRef[aa].keys():
                if fij[aa] is not None and wijRef[aa][codon] > 0:
                    sum2 += fij[aa][codon]
    if verbose >= 2: print("sum2:", sum2)

    CAI = exp(sum1/sum2) if sum2 != 0 else 0.0
    if verbose >= 2: print("CAI:", CAI)

    return CAI


def compute_CAI_df(seqQueryS2, seqRef, seqRefIndex, aaList0, aaCodonDict, method, codonUsageRef=None,
                   checkLengthMultipleOf3=True, verbose=1):
    """
    Compute the CAI and metric for a series of query sequences against one reference codon usage.
    """
    if codonUsageRef is None:
        # Correct wij for zero counts to a value of 0.01, in order to avoid
        # a codon having a relative adaptiveness of ln(wij) = -inf.
        codonUsageRef1 = compute_codon_usage(seqRef, aaCodonDict=aaCodonDict, correctWijZeroCounts=0.01,
                                             checkLengthMultipleOf3=checkLengthMultipleOf3, verbose=1)
    else:
        codonUsageRef1 = codonUsageRef

    fijRef = codonUsageRef1['fij']
    wijRef = codonUsageRef1['wij']
    if verbose >= 2: print("fijRef", codonUsageRef1['fij'])
    if verbose >= 2: print("wijRef", wijRef)

    # Exclude families of synonymous codons which have 0 counts for all codons in the reference set
    aa0Count = [aa for aa in aaList0 if codonUsageRef1['fij'][aa] is None]
    aaList = [aa for aa in aaList0 if aa not in aa0Count]
    if len(aa0Count) > 0:
        if verbose >= 1:
            print("WARNING: we discarded synonymous family for aa {} because all of their codons have 0 counts in the reference set.".format(aa0Count))
            print("The corrected synonymous family list is:", aaList)
    if verbose >= 2: print("aaList: ", aaList)

    # Since we only execute the following to raise a warning, it is not necessary in the
    # case of all to all
    if method in ['all_query_vs_avg_ref', 'avg_query_vs_avg_ref']:
        # Check if we have any zero-count codon in the reference set
        zeroCountCodonList = [aa for aa, codonCountD in codonUsageRef1['synCodonCount'].items()
                              if 0 in codonCountD.values() and aa in aaList]
        if len(zeroCountCodonList) > 0:
            if verbose >= 1:
                print("WARNING: we have some codons in {} families with zero counts in the reference set, the CAI will be unreliable.".format(",".join(zeroCountCodonList)))

    if method in ['avg_query_vs_avg_ref']:
        caiDf = pd.DataFrame({'CAI':[_compute_CAI(seqQueryS2, wijRef, aaList=aaList, aaCodonDict=aaCodonDict,
                                                  checkLengthMultipleOf3=checkLengthMultipleOf3, verbose=verbose)],
                              'metric':[_compute_metric_seq(seqQueryS2, fijRef, aaCodonDict=aaCodonDict,
                                                            checkLengthMultipleOf3=checkLengthMultipleOf3, verbose=verbose)]})
    else:
        caiDf = seqQueryS2.apply(lambda x: pd.Series({'CAI':_compute_CAI(x, wijRef, aaList=aaList, aaCodonDict=aaCodonDict,
                                                                         checkLengthMultipleOf3=checkLengthMultipleOf3,
                                                                         verbose=verbose),
                                                      'metric':_compute_metric_seq(x, fijRef, aaCodonDict=aaCodonDict,
                                                                                   checkLengthMultipleOf3=checkLengthMultipleOf3,
                                                                                   verbose=verbose),
                                                      'ref_index':seqRefIndex
                                                      }))
    return caiDf


def compute_CAI(seqQueryS, seqRefS, aaCodonDict, codonUsageRef=None,
                method='all_query_vs_avg_ref',
                excludeSeqNotMultipleOf3=True,
                computeAvgQueryCAI=False, computeAllQueryToAllRef=False, computeAllQueryToAllQuery=False,
                split6codonsSynonymousFamilies=True, exclude1CodonFamilies=True,
                verbose=1):
    """
    Calculate the Codon Adaptation Index (CAI) of a genome compared to a reference set of highly expressed genes.
    The method here is that described by Xia, X. (2007). An improved implementation of codon adaptation index.
    Evolutionary Bioinformatics, 3(613), 53–58. http://doi.org/10.1177/117693430700300028
    
    Three rules:
        1. ignore all amino acids only encoded by one codon (M and W in the standard genetic code)
        2. ignore any CAI where a count is 0. Because we don't know what the value should be
        3. Split 6-codons families L, R, and S into two groups of codons based on their first two bases
    
    aaCodonDict should be of the form:
    {'A': ['GCT', 'GCC', 'GCA', 'GCG'],
     'C': ['TGT', 'TGC'], ...}
     
    Note that aaCodonDict will be automatically splitted into the correct codons groups.
    """
    

    methodList = ['all_query_vs_avg_ref', 'avg_query_vs_avg_ref', 'all_query_vs_all_ref', 'all_query_vs_all_query']
    if method not in methodList:
        raise ValueError("ERROR, method should be one of the following values: {}".format(', '.join(methodList)))
    
    # Define codon dictionary and check input sequences
    if verbose >= 2: print("###")
    if split6codonsSynonymousFamilies:
        aaCodonDict = split_6_codons_synonymous_families(aaCodonDict)
    
    if type(seqQueryS) is not pd.Series:
        if verbose >= 2: print("seqQueryS not series.")
        seqQueryS = pd.Series(seqQueryS)
    if type(seqRefS) is not pd.Series and seqRefS is not None:
        if verbose >= 2: print("seqRefS not series.")
        seqRefS = pd.Series(seqRefS)

    # Exclude the stop codon
    aaCodonDict = {aa:l for aa, l in aaCodonDict.items() if aa != '*'}

    # Exclude families of synonymous codons with only 1 member
    if exclude1CodonFamilies:
        aaList = [aa for aa, codonList in aaCodonDict.items() if len(codonList) > 1]
        if len(aaList) != len(aaCodonDict):
            discardedAA = ",".join([aa for aa, codonList in aaCodonDict.items() if len(codonList) == 1])
            if verbose>= 1:
                print("WARNING: we discarded synonymous family for aa {} because they contain only 1 codon and could bias the computation of CAI.".format(discardedAA))
                print("The corrected synonymous family list is:", aaList)
    else:
        aaList = [aa for aa, codonList in aaCodonDict.items()]
    
    seqQueryNotValidList = seqQueryS.map(lambda seq: len(seq) % 3 != 0)
    if any(seqQueryNotValidList):
        print("WARNING: following query sequences have a length NOT multiple of 3:", list(seqQueryS[seqQueryNotValidList].index))
    if seqRefS is not None:
        seqRefNotValidList = seqRefS.map(lambda seq: len(seq) % 3 != 0)
        if any(seqRefNotValidList):
            print("WARNING: following reference sequences have a length NOT multiple of 3:", list(seqRefS[seqRefNotValidList].index))

    if excludeSeqNotMultipleOf3:
        if seqRefS is not None:
            seqRefS2 = seqRefS[~seqRefNotValidList]
        else:
            seqRefS2 = seqRefS
        seqQueryS2 = seqQueryS[~seqQueryNotValidList]
    else:
        if seqRefS is not None:
            seqRefS2 = seqRefS
        else:
            seqRefS2 = seqRefS
        seqQueryS2 = seqQueryS
        
    if method in ['all_query_vs_avg_ref', 'avg_query_vs_avg_ref']:
        # Compute CAI of query sequences against the average codon usage of reference sequence set.
        seqRefSuperList = pd.Series([seqRefS2], index=['avg_ref_set'])
    else:
        # Iterate over each individual sequence in the reference set
        # and compute CAI for all query sequences against all reference sequences.
        seqRefSuperList = seqRefS2
        
    caiDfList = []
    if method in ['all_query_vs_all_query']:
        if seqRefS is not None:
            print("ERROR: With method all_query_vs_all_query reference set has to be None.")
        else:
            caiDfList = []
            n = len(seqQueryS2)
            for i, (name, seqS_i) in enumerate(seqQueryS2.iteritems()):
                if i > 0:
                    seqS_i2 = pd.Series([seqS_i], index=[name])
                    seqS_j = seqQueryS2.iloc[:i]
                    if verbose >= 1: print("query seq #", i, "/", n, "len(seq_j):", len(seqS_j))
                    if verbose >= 2:
                        print("seqS_i2:\n", seqS_i2, '\n')
                        print(seqS_i2.index.tolist(), '\n')
                        print("seqS_j:\n", seqS_j, '\n')
                    caiDf = compute_CAI_df(seqS_j, seqS_i2, seqS_i2.index[0], aaList0=aaList, aaCodonDict=aaCodonDict,
                                           method=method, checkLengthMultipleOf3=excludeSeqNotMultipleOf3, verbose=verbose)
                    caiDf = caiDf.drop('CAI', axis=1)
                    caiDfList.append(caiDf)
            caiDf2 = pd.concat(caiDfList)
            
    else:
        for seqRefIndex, seqRef in seqRefSuperList.iteritems():
            caiDf = compute_CAI_df(seqQueryS2, seqRef, seqRefIndex, aaList0=aaList, aaCodonDict=aaCodonDict,
                                   method=method, checkLengthMultipleOf3=excludeSeqNotMultipleOf3, verbose=verbose)
            caiDfList.append(caiDf)

        if method in ['all_query_vs_avg_ref', 'avg_query_vs_avg_ref']:
            caiDf2 = caiDfList[0]
        else:
            caiDf2 = pd.concat(caiDfList)

    caiDf2.index.name = 'query_index'
    return caiDf2




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, help="Path of the query sequences in fasta format.")
    parser.add_argument('--ref', type=str, help="Path of the reference sequences in fasta format.")
    parser.add_argument('--codonTableId', type=int, help="ID of the NCBI codon table.")
    parser.add_argument('--verbose', type=int, default=1)
    args = parser.parse_args()

    print("not implemented yet...")

    # read fasta files
    # ...

    # get the codon table based on table id from arg
    # ...
