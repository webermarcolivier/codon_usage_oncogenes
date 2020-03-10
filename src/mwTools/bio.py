# coding: utf-8
# Author: Marc Weber

"""
=========================================================================================
Biopython sequences tools
=========================================================================================
"""

import pandas as pd
from pandas import isnull
import numpy as np
from collections import Counter
import re
import os.path
from pathlib import Path
import gzip
from Bio.Data import CodonTable
from Bio.SeqFeature import SeqFeature, FeatureLocation, ExactPosition
from Bio.Data.CodonTable import TranslationError
from Bio.Seq import Seq
import numbers
import textwrap

from .pandas import sort_df



codonTableBioMPN = CodonTable.unambiguous_dna_by_name['Mycoplasma']


def convert_df_to_fasta(df, seqCol, idColList=None, filepath=None, wrap_sequence=True,
                        verbose=0):
    """
    Convert a dataframe to fasta format. Each row of the dataframe contains one sequence
    in column `seqCol`. The fasta identifier is build based on several other columns of the 
    dataframe. Example:
    '>idCol0|idCol1|idCol2'
    
    If not `idColList` is given, the dataframe index will be used for the fasta identifiers.

    Note that for building a local blast database, the required formatting for `makeblastdb`
    is `>lcl|integer_or_string`
    """

    if len(df) == 0:
        print("Warning: input dataframe or series has length zero.")
        return None

    if type(df) is pd.Series:
        df2 = df.to_frame().T
    else:
        df2 = df

    indexName = df2.index.name
    # df2 = df2.reset_index()
    
    def convert_record_to_fasta(row):

        if idColList is not None:
            idList = row[idColList].tolist()
            idList = [str(idj) if not pd.isnull(idj) else '' for idj in idList]
        else:
            # By default, use row index as id
            idList = [row.name]

        fastaString = '>' + "|".join(idList) + '\n'
        seqString = row[seqCol]
        if wrap_sequence:
            seqString = textwrap.fill(seqString, 70)
        fastaString += seqString
        return fastaString

    seqList = df2.apply(convert_record_to_fasta, axis=1).tolist()
    fastaString = "\n".join(seqList)

    if filepath is not None:
        with open(str(filepath), 'w') as f:
            f.write(fastaString)
    
    return fastaString


def pretty_print_mRNA(genomeBioSeq, TSS, TTS, CDS_start, CDS_stop, strand):
    """
    Pretty print a DNA sequence (corresponding to a mRNA transcript) by highlighting start and stop codons.
    
    genomeBioSeq: DNA sequence of type Bio.Seq object
    TSS, TTS, CDS_start, CDS_stop: all position are 0-based start-inclusive end-exclusive index
    
    More information on how to use HTML and CSS styles in jupyter output:
    http://stackoverflow.com/questions/18024769/adding-custom-styled-paragraphs-in-markdown-cells
    http://stackoverflow.com/questions/18225302/how-can-i-wrap-or-break-long-text-word-in-a-fixed-width-span
    
    Example:
    seq = Bio.Seq.Seq('GCGCGATGAAAGTTTTGAGTTATGAGAATGAGTTGAACAAAATCCTAAGAAAACTCAGATGAAAGTTTTGATTAATAAGAATGAGTTGAACAAAATCCTAAGAAAACTCAGATGACATTTTTGA',
                      alphabet=Bio.Alphabet.IUPAC.ambiguous_dna)
    pretty_print_mRNA(seq, 1, len(seq), 6, 48, "+")
    pretty_print_mRNA(seq, len(seq), 1, len(seq)+1-7, 20, "-")
    """
    
    genomeSeq = 0
    if strand == '+':
        genomeSeq = str(genomeBioSeq.seq)
    elif strand == '-':
        # Reverse complement the genome sequence and reverse the locations
        genomeSeq = str(genomeBioSeq.reverse_complement().seq)
        n = len(genomeBioSeq)
        TSS = (n + 1) - TSS
        TTS = (n + 1) - TTS
        CDS_start = (n + 1) - CDS_start
        CDS_stop = (n + 1) - CDS_stop
        #print('TSS:',TSS,'CDS_start',CDS_start,'CDS_stop',CDS_stop,'TTS',TTS)
    
    # Check inconsistencies in locations
    if TSS is None or TTS is None or CDS_start is None or CDS_stop is None or strand is None:
        print("Error: None value in arguments")
        return None
    if not TSS < CDS_start or not CDS_start < CDS_stop or not CDS_stop < TTS:
        print("Error: unordered locations, should have TSS < CDS_start < CDS_stop < TTS")
        return None
    
    #color5UTR = "#88EFFC"
    colorStartCodon = "#88EFFC"
    colorStopCodon = "#FFA8A8"
    #color3UTR = "#88EFFC"
    # On the first line we define the html style using internal CSS styling
    # See: http://www.w3schools.com/html/html_css.asp
    htmlString = '<style>div.prettySeq { width: 700px;  word-break: break-all; }</style>' + \
                 '<pre><div class="prettySeq">' + \
                 genomeSeq[TSS-1 : CDS_start-1] + \
                 '<span word-wrap="break-word" style="background-color: ' + colorStartCodon + '">' + \
                 genomeSeq[CDS_start-1 : CDS_start-1+3] + '</span>' + \
                 genomeSeq[CDS_start-1+3 : CDS_stop-1-2] + \
                 '<span style="background-color: ' + colorStopCodon + '">' + genomeSeq[CDS_stop-1-2 : CDS_stop-1+1] + '</span>' + \
                 genomeSeq[CDS_stop-1+1 : TTS] + \
                 '</div></pre>'
    return htmlString


def convert_strand_numeric_to_strand_string(strandNum):
    # Convert strand +1/-1 to +/-
    if strandNum == +1:
        strand = '+'
    elif strandNum == -1:
        strand = '-'
    elif strandNum == '+' or strandNum == '-':
        strand = strandNum
    else:
        strand = None
    return strand


def convert_strand_string_to_strand_numeric(strandStr):
    # Convert strand from +/- to +1/-1
    if strandStr == '+':
        strand = +1
    elif strandStr == '-':
        strand = -1
    elif strandStr == +1 or strandStr == -1:
        strand = strandStr
    else:
        strand = None
    return strand


def convert_index_1b_inc_inc_to_0b_inc_exc(row, strandCol='strand', startCol='start', endCol='end'):
    """
    From: 1-based index inclusive-inclusive oriented (start > end on minus strand) [wetlab index]
    To:   0-based index inclusive-exclusive ordered (start < end always) [Biopython index]
    """
    strand = row[strandCol]
    start = row[startCol]
    start = int(start) if not isnull(start) else None
    end = row[endCol]
    end = int(end) if not isnull(end) else None
    
    if start is not None and end is not None:
        if start <= end:
            left, right = start - 1, end
        else:
            left, right = end - 1, start
        return {startCol:left, endCol:right, strandCol:strand}
    else:
        return {startCol:None, endCol:None, strandCol:strand}


def convert_index_1b_inc_inc_to_0b_inc_inc(row):
    """
    From: 1-based index inclusive-inclusive oriented (start > end on minus strand) [wetlab index]
    To:   0-based index inclusive-inclusive ordered (start < end always)
    """
    strand = row['strand']
    start = row['start']
    start = int(start) if not isnull(start) else None
    end = row['end']
    end = int(end) if not isnull(end) else None
    
    if start is not None and end is not None:
        if start <= end:
            left, right = start - 1, end - 1
        else:
            left, right = end - 1, start - 1
        return {'start':left, 'end':right, 'strand':strand}
    else:
        return {'start':None, 'end':None, 'strand':strand}


def convert_codon_pos_to_genome_pos(codonPos, CDS, strand=None):
    if strand is None:
        strand = CDS['strand']

    if strand == '+':
        posInGenome = 3*codonPos + CDS['start']
    elif strand == '-':
        posInGenome = CDS['end'] - 3*codonPos

    return posInGenome


def get_context_window(codonPos, CDS, windowContextUp, windowContextDown):
    """
    Only implemented protein seq !

    The window context width is in codons for protein sequence context window.
    """
    proteinSeq = CDS['protein_seq']

    contextCodonUp = max(0, codonPos - windowContextUp)
    contextCodonDown = min(len(proteinSeq), codonPos + windowContextDown)
    posCodonInContext = min(codonPos, windowContextUp)
    contextSeq = proteinSeq[contextCodonUp:contextCodonDown]
    
    contextPosInGenomeUp = convert_codon_pos_to_genome_pos(contextCodonUp, CDS)
    contextPosInGenomeDown = convert_codon_pos_to_genome_pos(contextCodonDown, CDS)

    strand = CDS['strand']
    if strand == '+':
        contextPosInGenomeStart, contextPosInGenomeEnd = contextPosInGenomeUp, contextPosInGenomeDown
    elif strand == '-':
        contextPosInGenomeStart, contextPosInGenomeEnd = contextPosInGenomeDown, contextPosInGenomeUp

    return {'context_window_protein_seq':contextSeq,
            'codon_pos_in_context':posCodonInContext,
            'context_window_codon_pos_up':contextCodonUp,
            'context_window_codon_pos_down':contextCodonDown,
            'context_window_genome_pos_start':contextPosInGenomeStart,
            'context_window_genome_pos_end':contextPosInGenomeEnd
            }


def find_motif_context_in_sequence(seq, motif, nUp, nDown, onlyFirstMatch=False):
    """
    Returns the subsequence around a motif in a sequence. Pads left and right with
    '-' characters if motif is close to the start or end of the sequence.
    Useful to align known motifs in a list of sequences.
    
    Note that by default, a list of all matched motifs is returned.
        
    Example:
    >>> for subseq in find_motif_context_in_sequence('VEKATTTTPVVQIELPAPPPTVTVVDQTSPPPTAAVTVSTSQPVIEEQTTVFNQTTQLEQLSV', 'PPP', 6, 6):
    >>>     print(subseq)
    >>> for subseq in find_motif_context_in_sequence('KTLLQKTNNSIQQLEAEIQIPTTHIKSDEVMFGPPPDMNERLVLNDSETDAILRSIEAELES', 'PPP', 6, 6):
    >>>     print(subseq)
    QIELPAPPPTVTVVD
    VVDQTSPPPTAAVTV
    DEVMFGPPPDMNERL
    """
    
    searchObj = re.finditer(motif, seq)

    stringList = []

    if searchObj:
        for match in searchObj:
            # Find the start index of the keyword
            start = match.span()[0]
            # Find the end index of the keyword
            end = match.span()[1]

            alignedString = "".join(['-' for _ in range(max(nUp - start, 0))] +
                                    [seq[max(start - nUp, 0):min(end + nDown, len(seq))]] +
                                    ['-' for _ in range(max(nDown - len(seq) + end, 0))]
                                    )
            stringList.append(alignedString)
            if onlyFirstMatch:
                break
        
    return stringList


def convert_location_bio_to_dict(feature, verbose=1):
    """Extracts start, end and strand from Biopython feature, only if it has an exact position,
    in 0-based start-inclusive end-exclusive counting with start <= end.
    """

    # First I wanted to filter out all locations that were fuzzy but this is too restrictive.
    # if (type(feature.location.start) is Bio.SeqFeature.ExactPosition) and \
    #    (type(feature.location.end) is Bio.SeqFeature.ExactPosition):
    # We just use all positions as integers.
    if True:

        start = int(feature.location.start)
        end = int(feature.location.end)
        strand = convert_strand_numeric_to_strand_string(feature.location.strand)
        # Biopython uses 0-based start-inclusive end-exclusive counting with start <= end
        return {'start':start, 'end':end, 'strand':strand}
    else:
        if verbose >= 1: print("ERROR: feature", feature, "has not an exact position.")
        if verbose >= 1: print(feature.location)
        return None


def convert_genbank_to_annotation_df(genomeBio, extractDNASeq=False, extractProteinSeq=False, verbose=1):
    
    species_name = genomeBio.annotations.get('organism')
    species_name = species_name if species_name is not None else ''
    
    def get_proteinGI(feature):
        proteinGI = feature.qualifiers.get('db_xref')
        proteinGI = int(re.sub(r'GI:', '', proteinGI[0])) if proteinGI is not None else None
        return proteinGI
    
    def get_attribute(feature, attName):
        if attName == 'proteinGI':
            att = get_proteinGI(feature)
        else:
            att = feature.qualifiers.get(attName)
            att = att[0] if att is not None else None
        return att
      
#     def get_attribute_dict(feature):
#         # Extract the first value of each qualifier
#         return {q:v[0] for q, v in feature.qualifiers.items()}

    def get_attribute_dict(feature):
        return {att:get_attribute(feature, att) for att in feature.qualifiers.keys()}
    
    hasWellDefinedGenomeSeq = \
        not (
            (genomeBio.seq.count('N') / len(genomeBio.seq)) > 0.5 or
            genomeBio.seq == ''
        )

    geneIdentifierPriorityList = ['locus_tag', 'gene', 'label', 'protein_id']
    
    CDSList = []
    if verbose >= 1: print("len(genomeBio.features):", len(genomeBio.features))
    for feature in genomeBio.features:
        attDict = get_attribute_dict(feature)
        location = convert_location_bio_to_dict(feature)
        locationBio = feature.location

        # Use the first of the attributes of the priority list found in the feature as main identifier
        featId = None
        for identifier in geneIdentifierPriorityList:
            if attDict.get(identifier) is not None:
                featId = attDict[identifier]
                break

        CDSDict = {'chromosome':genomeBio.id,
                   'id':featId,
                   'feature':feature.type,
                   'strand':location['strand'],
                   'start':location['start'],
                   'end':location['end'],
                   'locus_tag':attDict.get('locus_tag'),
                   'gene':attDict.get('gene'),
                   'protein_id':attDict.get('protein_id')
                   }
        if extractDNASeq and hasWellDefinedGenomeSeq:
            dnaSeqBio = SeqFeature(location=locationBio).extract(genomeBio)
            dnaSeq = str(dnaSeqBio.seq)
            CDSDict['DNA_seq'] = dnaSeq
        else:
            dnaSeqBio = None
            dnaSeq = None
        
        if extractProteinSeq and feature.type == 'CDS' and dnaSeqBio is not None:
            codonTableId = attDict.get('transl_table')
            codonStartPos = attDict.get('codon_start')    # in 1-based index
            codonStartPos = int(codonStartPos) if codonStartPos is not None else 1
            if codonTableId is not None:
                try:
                    proteinSeq = str(dnaSeqBio.seq[3*(codonStartPos - 1):].translate(table=codonTableId, cds=True))
                except TranslationError:
                    try:
                        proteinSeq = str(dnaSeqBio.seq[3*(codonStartPos - 1):].translate(table=codonTableId, cds=False))
                    except TranslationError:
                        proteinSeq = None

            CDSDict['protein_seq'] = proteinSeq

        CDSList.append(CDSDict)
        if verbose >= 2: print("\n\n")

    CDSDf = pd.DataFrame(CDSList)
    CDSDf = CDSDf.sort_values(by=['start', 'end', 'strand'])
    
    return CDSDf


def convert_annotation_df_to_bed(annotDf, chromosomeCol='chromosome', startCol='start', endCol='end',
                                 strandCol='strand', idCol='id', featureCol='feature', combineFeatureAndId=False,
                                 sort=True, sortBy=None, sortAscending=None
                                 ):
    """We assume that annotation features dataframe uses 0-based start-inclusive end-exclusive
    counting with start <= end."""

    df = annotDf
    if type(df) is pd.Series:
        df = df.to_frame().T
    sortColList = [startCol, endCol, strandCol]
    sortAscendingList = [True, True, True]
    if sort:
        if sortBy is not None:
            if 'chromosome' in sortBy:
                raise ValueError("chromosome is included by default and has to be sorted.")
            sortColList = sortBy + sortColList
            sortAscendingList = sortAscending + sortAscendingList
        df = df.sort_values(by=sortColList, ascending=sortAscendingList)
        df = sort_df(df, chromosomeCol, key=lambda x: (x.upper(), x[0].islower()), reverse=False)
    print(df)
    df.index.name = idCol

    if idCol not in df.columns:
        df = df.reset_index()

    bed_string = ""
    for index, annot in df.iterrows():
        # BED format uses 0-based start-inclusive end-exclusive counting (as in Python)
        # BED format start < end
        chromosomeName = annot[chromosomeCol]
        if combineFeatureAndId:
            bed_id = "{};{}".format(annot[featureCol], annot[idCol])
        else:
            bed_id = annot[idCol]
        bed_start = int(annot[startCol])
        bed_end = int(annot[endCol])
        bed_strand = annot[strandCol]
        if bed_start > bed_end:
            bed_start, bed_end = bed_end, bed_start
        bed_string += "{}\t{}\t{}\t{}\t{:d}\t{}\n".format(chromosomeName, str(bed_start), str(bed_end),
                                                          bed_id, 0, bed_strand)
    return bed_string


def convert_Bio_feature_to_bed(featureList, referenceName):
    bed_string = ""
    for feature in featureList:
        # BED format uses 0-based start-inclusive end-exclusive counting (as in Python)
        # BED format start < end
        start = feature.location.start.position
        end = feature.location.end.position
        if 'gene' in feature.qualifiers.keys():
            name = feature.qualifiers['gene'][0]
        elif 'locus_tag' in feature.qualifiers.keys():
            # some features only have a locus tag
            name = feature.qualifiers['locus_tag'][0]
        else:
            name = ''
        strand = convert_strand_numeric_to_strand_string(feature.strand)

        # Start and end should be always ordered in GenBank file but we check anyway
        if start > end:
            start, end = end, start
            
        bed_string += "{}\t{:d}\t{:d}\t{}\t0\t{}\n".format(referenceName, start, end, name, strand)

    return bed_string


def convert_annotation_df_to_gtf(annotDf, chromosomeCol='chromosome', startCol='start', endCol='end',
                                 strandCol='strand', idCol='id', featureToIncludeList=None,
                                 transcriptIdCol='transcript_id_unique', usePseudoTranscriptOnly=False
                                 ):
    """We assume that annotation features dataframe uses 0-based start-inclusive end-exclusive
    counting with start <= end."""
    
    def format_attribute(attDict):
        attList = []
        for (key, value) in attDict.items():
            if isinstance(value, numbers.Number):
                attString = '{} "{}"'.format(str(key), value)
            else:
                attString = '{} "{}"'.format(str(key), value)
            attList.append(attString)
        return "; ".join(attList)
    
    gtf_string = ""
    for index, annot in annotDf.sort_values(by=[chromosomeCol, startCol, endCol, strandCol]).iterrows():
        
        if not (featureToIncludeList is not None and annot['feature'] not in featureToIncludeList):

            # See note on GTF format:
            # > chromosome names can be given with or without the 'chr' prefix. Important note:
            # > the seqname must be one used within Ensembl, i.e. a standard chromosome name or an Ensembl identifier
            # > such as a scaffold ID, without any additional content such as species or assembly.
            # chromosomeName = "chr1"
            chromosomeName = annot[chromosomeCol]
            gtf_feature = annot['feature']
            gtf_attributeDict = {}
            # GTF format uses 1-based start-inclusive end-inclusive.
            if annot['feature'] == 'transcript' and not usePseudoTranscriptOnly:
                gtf_feature = 'transcript'
                gtf_attributeDict['transcript_id'] = annot['id']
                gtf_start = int(annot[startCol]) + 1
                gtf_end = int(annot[endCol])
            elif annot['feature'] == 'CDS':
                gtf_feature = 'CDS'
                gtf_attributeDict['gene_id'] = annot['id']
                gtf_attributeDict['transcript_id'] = annot[transcriptIdCol]
                gtf_start = int(annot[startCol]) + 1
                # GTF format excludes stop codon from CDS interval
                # See http://mblab.wustl.edu/GTF22.html
                gtf_end = int(annot[endCol]) - 3
            elif annot['feature'] == 'ncRNA':
                gtf_feature = 'gene'
                gtf_attributeDict['gene_id'] = annot['id']
                gtf_attributeDict['transcript_id'] = annot[transcriptIdCol]
                gtf_start = int(annot[startCol]) + 1
                gtf_end = int(annot[endCol])
            else:
                gtf_feature = annot['feature']
                gtf_attributeDict['transcript_id'] = annot[transcriptIdCol]
                gtf_start = int(annot[startCol]) + 1
                gtf_end = int(annot[endCol])

            if annot['feature'] != 'transcript' and usePseudoTranscriptOnly:
                # Update the reference to the transcript id
                gtf_attributeDict['transcript_id'] = 't' + annot['id']

            gtf_strand = annot[strandCol]
            gtf_frame = '.'

            if annot['feature'] != 'transcript' or not usePseudoTranscriptOnly:
                if gtf_start > gtf_end:
                    gtf_start, gtf_end = gtf_end, gtf_start
                gtf_string += "{}\t.\t{}\t{}\t{}\t.\t{}\t{}\t{}\n"\
                              .format(chromosomeName,
                                      gtf_feature,
                                      str(gtf_start),
                                      str(gtf_end),
                                      gtf_strand,
                                      gtf_frame,
                                      format_attribute(gtf_attributeDict)
                                      )

            if annot['feature'] == 'CDS':
                gtf_feature = 'exon'
                gtf_string += "{}\t.\t{}\t{}\t{}\t.\t{}\t{}\t{}\n"\
                              .format(chromosomeName,
                                      gtf_feature,
                                      str(gtf_start),
                                      str(gtf_end),
                                      gtf_strand,
                                      gtf_frame,
                                      format_attribute(gtf_attributeDict)
                                      )

            if annot['feature'] != 'transcript' and usePseudoTranscriptOnly:
                # We add a pseudo-transcript with the same coordinates as the CDS
                # but including the stop codon
                gtf_feature = 'transcript'
                gtf_attributeDict = {}
                gtf_attributeDict['transcript_id'] = 't' + annot['id']
                gtf_start = int(annot[startCol]) + 1
                gtf_end = int(annot[endCol])
                gtf_string += "{}\t.\t{}\t{}\t{}\t.\t{}\t{}\t{}\n"\
                              .format(chromosomeName,
                                      gtf_feature,
                                      str(gtf_start),
                                      str(gtf_end),
                                      gtf_strand,
                                      gtf_frame,
                                      format_attribute(gtf_attributeDict)
                                      )

    return gtf_string


def extract_SeqFeature_Bio(annot, startCol='start', endCol='end', featureCol='feature',
                           strandCol='strand', idCol='id'):
    """Input positions: 0-based start-inclusive end-exclusive"""
    strandNum = convert_strand_string_to_strand_numeric(annot[strandCol])
    # Biopython uses 0-based start-inclusive end-exclusive counting with start <= end
    if featureCol in annot.index:
        featType = annot[featureCol]
    else:
        featType = None
    if idCol in annot.index:
        featId = annot[idCol]
    else:
        featId = annot.name
    return SeqFeature(location=FeatureLocation(int(annot[startCol]), int(annot[endCol]), strand=strandNum),
                      type=featType, id=featId)


def extract_dna_seq(annot, genomeBio, idCol='id'):
    seqFeatureBio = extract_SeqFeature_Bio(annot, idCol=idCol)
    return seqFeatureBio.extract(genomeBio).seq


def extract_protein_seq(annotDf, genomeBio, idCol='id'):
    if annotDf.feature == 'CDS':
        dnaSeq = extract_dna_seq(annotDf, genomeBio, idCol=idCol)
        # Warning: we have to use the codon table of Mycoplasma!
        # Note that by using the cds=True option, we will always use Methionine as first amino acid.
        return dnaSeq.translate(table=codonTableBioMPN, cds=True)
    else:
        return None


def extract_codons_list(seq, frame=0, checkLengthMultipleOf3=False, frameFromEnd=False):
    
    if len(seq) % 3 != 0 and checkLengthMultipleOf3:
        print("ERROR: seq length is not multiple of 3.")
        return None

    l = len(seq)
    if frameFromEnd:
        frame1 = ((l % 3) + frame) % 3
    else:
        frame1 = frame
    codonList = (seq[3*n + frame1: 3*n + frame1 + 3] for n in range(0, int((l - frame1) / 3)))
    return codonList


def find_ORFs(seqBio, codonTable, startCodons=['ATG', 'GTG', 'TTG'], verbose=0):
    """
    Find all putative open reading frames (ORFs) in the nucleotide sequence.
    
    We assume that an ORF start with one of the start codons as defined in the argument
    list and ends with a stop codon as defined from the codon table.
    
    Note that the expression levels from other non-canonical start codons different from
    ATG, GTG or TTG is extremely low, as shown in E. coli [1].
    
    [1] Hecht, A., Glasgow, J., Jaschke, P. R., Bawazer, L. A., Munson, M. S., Cochran, J. R., … Salit, M. (2017).
    Measurements of translation initiation from all 64 codons in E. coli. Nucleic Acids Research, 1–12.
    http://doi.org/10.1093/nar/gkx070
    """
    
    ORFList = []
    
    for strand, seq in [(+1, seqBio), (-1, seqBio.reverse_complement())]:
        # We define frame relative to the start of the seq
        for frame in range(3):
            translatedSeq = seq[frame:].translate(codonTable)
            if verbose >= 2: print("\n\n#### seq", seq, "\nframe", frame, "\ntranslatedSeq", translatedSeq)
            
            codonList = list(extract_codons_list(seq, frame=frame, checkLengthMultipleOf3=False))

            # Find positions of all stop codons
            iCodonStopList = [iCodon for iCodon, codon in enumerate(codonList)
                              if codon in codonTable.stop_codons]
            if verbose >= 2:
                print(iCodonStopList)
                print([codonList[i] for i in iCodonStopList])
            
            # In each subsequence in between stop codons, search for start codons
            iCodonFirstInChunk = 0
            for iCodonStop in iCodonStopList:
                if verbose >= 2: print("iCodonFirstInChunk", iCodonFirstInChunk, "iCodonStop", iCodonStop)
                codonListChunk = codonList[iCodonFirstInChunk:iCodonStop + 1]
                ORFStopCodon = codonList[iCodonStop]
                if verbose >= 2: print("codonListChunk", [str(c) for c in codonListChunk])
                
                # Find the start codon positions inside the chunk
                iCodonStartList = [i for i, c in enumerate(codonListChunk) if str(c) in startCodons]
                if verbose >= 2: print("iCodonStartList", iCodonStartList)
                
                for iStart in iCodonStartList:
                    if verbose >= 2: print("iStart", iStart)
                    iCodonInSeq = iCodonFirstInChunk + iStart
                    ORFStartCodon = codonList[iCodonInSeq]
                    if verbose >= 2: print("iCodonInSeq", iCodonInSeq, "codonList[iCodonInSeq]", codonList[iCodonInSeq])
                    
                    # Position of start and stop codons in the nucleotide sequence
                    iStartInNucleotideSeq = iCodonInSeq*3 + frame
                    iStopInNucleotideSeq = iCodonStop*3 + frame
                    if verbose >= 2: print("iStartInNucleotideSeq", iStartInNucleotideSeq, "iStopInNucleotideSeq", iStopInNucleotideSeq)

                    # Define ORF region on the nucleotide sequence
                    # STRAND ???????
                    if strand == +1:
                        ORFStart = iStartInNucleotideSeq
                        ORFEnd = iStopInNucleotideSeq + 2 + 1
                    elif strand == -1:
                        # reverse location
                        ORFStart = (len(seq) - 1) - (iStopInNucleotideSeq + 2)    # include stop codon
                        ORFEnd = (len(seq) - 1) - (iStartInNucleotideSeq) + 1
                    if verbose >= 2: print("ORFStart", ORFStart, "ORFEnd", ORFEnd, "strand", strand)
            
                    # Define ORF as Biopython SeqFeature
                    ORFFeat = SeqFeature(location=FeatureLocation(ORFStart, ORFEnd, strand=strand),
                                         type='putative ORF', id=None)
                    
                    # Compute translation of ORF
                    ORFFeat.qualifiers['translation'] = ORFFeat.extract(seqBio).translate(table=codonTable,
                                                                                          cds=True)
                    ORFFeat.qualifiers['start_codon'] = str(ORFStartCodon)
                    ORFFeat.qualifiers['stop_codon'] = str(ORFStopCodon)

                    if verbose >= 2: 
                        print("ORFFeat", ORFFeat)
                        print("ORFFeat.extract(seqBio)", ORFFeat.extract(seqBio))
                    ORFList.append(ORFFeat)
                
                iCodonFirstInChunk = iCodonStop + 1
                
    return ORFList



#====================================================================

# ## RefSeq bacterial genomes database

def read_assemblySummary_file(assemblySummaryFilePath):
    """
    Read assembly summary file and choose only complete genomes of the representative and reference RefSeq sets.
    """

    assemblySummaryDf = pd.read_csv(assemblySummaryFilePath, header=1, sep='\t')
    assemblySummaryDf.rename(columns={'# assembly_accession':'assembly_accession'}, inplace=True)
    assemblySummaryDf.set_index('assembly_accession', inplace=True, verify_integrity=True)
    assemblySummaryDf['compressedGenomeFilename'] = \
        assemblySummaryDf['ftp_path'].map(lambda path: re.sub(r'(ftp://ftp\.ncbi\.nlm\.nih\.gov.*)/(GCF_.+)',
                                                              r'\2_genomic.gbff.gz', path))
    
    # We choose only the genomes in the representative and reference sets.
    print('Nb of genomes in RefSeq database:',len(assemblySummaryDf))
    assemblySummaryRepDf = assemblySummaryDf[ (assemblySummaryDf.refseq_category == 'representative genome') |
                                              (assemblySummaryDf.refseq_category == 'reference genome')
                                             ].copy()
    print('Nb of genomes in RefSeq database, representative or reference genomes:',len(assemblySummaryRepDf))
    
    # We filter out genomes that are from shotgun genome sequencing, because they are still drafts
    assemblySummaryRepDf = assemblySummaryRepDf[assemblySummaryRepDf['assembly_level'] == 'Complete Genome']
    print('Nb of genomes in RefSeq database, after filtering out contigs:',len(assemblySummaryRepDf))
    
    return assemblySummaryRepDf


# ## Import genome file

def import_genome_gbff_file(assemblySummaryRepDf, species_name=None, genome_accession=None):
    
    if genome_accession is None and species_name is not None:
        index_match = assemblySummaryRepDf['organism_name'].str.contains(species_name)
        if sum(index_match) != 1:
            print('Error finding species name in genome list. Species name matches found:')
            print(assemblySummaryRepDf[index_match])
            return None
        genome_accession = assemblySummaryRepDf[index_match].index[0]

    compressedGenomeFilename = assemblySummaryRepDf.loc[genome_accession,'compressedGenomeFilename']
    organism_name = assemblySummaryRepDf.loc[genome_accession,'organism_name']
    return compressedGenomeFilename, organism_name, genome_accession


def extract_compressed_genome_file(compressedGenomeFilename, compressedFolder, extractedFolder):
    with gzip.GzipFile(os.path.join(compressedFolder,compressedGenomeFilename), mode='r') as genomeZipFile:
        extractedGenomeFilename = re.match(r'(.+)\.gz', compressedGenomeFilename).group(1)
        with open(os.path.join(extractedFolder, extractedGenomeFilename), 'wb') as file:
            file.write(genomeZipFile.read())
    return extractedGenomeFilename


def build_refCodonTable(codonTableBio):

    # Group amino acids by physicochemical properties
    # Note: we drop the Selenocysteine U (very rare amino acid)
    aaTable = ['R', 'K', 'H', 'D', 'E', 'S', 'T', 'N', 'Q', 'C', 'G', 'P', 'A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V']

    # Build the reference codon table (as dataframe) using the codon table from the genome file
    refCodonTableDf = pd.DataFrame(index=range(64), columns=['codon','aa','i','j','alphabetical_sorted_index','aa_groups_sorted_index'])
    letters = ["T","C","A","G"]
    i, j = 0, 0
    k = 0
    for k2, c2 in enumerate(letters):
        for k1, c1 in enumerate(letters):
            for k3, c3 in enumerate(letters):
                i = 4*k1 + k3 + 1
                j = 2*k2 + 1
                codon = c1 + c2 + c3
                refCodonTableDf.loc[k, 'alphabetical_sorted_index'] = k
                refCodonTableDf.loc[k, 'codon'] = codon
                refCodonTableDf.loc[k, 'i'] = i
                refCodonTableDf.loc[k, 'j'] = j

                # Add the amino acid
                # Here we follow the rules defined in the codon table from the Biopython genome object
                if codon in codonTableBio.stop_codons:
                    refCodonTableDf.loc[k, 'aa'] = "*"
                else:
                    try:
                        amino = codonTableBio.forward_table[codon]
                    except KeyError:
                        amino = "?"
                    except TranslationError:
                        amino = "?"
                    # if codon in codonTableBio.start_codons:
                    #     amino += "(s)"
                    refCodonTableDf.loc[k, 'aa'] = amino
                k += 1

    refCodonTableDf.set_index('codon', inplace=True)

    # Group the synonymous codons together in the same order as they appear in the codon table,
    # and sort them following the same order as the amino acid table (physico-chemical properties)
    grp = refCodonTableDf.groupby(['aa'], sort=False)

    # Sort the group list
    # Note: because we have several versions of the same aa in the codon table (ex: I and I(s)),
    # we have to search for matching string in the group keys
    aaTable2 = aaTable + [aa for aa in refCodonTableDf['aa'].unique() if aa not in aaTable]
    sortedGroups = \
        [
            [[key, value] for key, value in grp.groups.items() if key.startswith(aa)]
            for aa in aaTable2
        ]

    # Flatten the nested list
    sortedGroups = [element for sublist in sortedGroups for element in sublist]

    # Sort the codon table by reassembling groups in the correct order
    refCodonTableDf = pd.concat([grp.get_group(group[0]) for group in sortedGroups])

    refCodonTableDf['aa_groups_sorted_index'] = refCodonTableDf.index.map(lambda codon: refCodonTableDf.index.get_loc(codon))
    return refCodonTableDf


def sort_codon_index(df, refCodonTableDf, addAminoAcidLetterToIndex=True, addAminoAcidLetterAsColumn=False,
                     addEmptyRowBetweenGroups=False, addEmptyRowBetweenGroupsFillValue=np.nan, useULetter=True):
    """Sort the codon index of the dataframe and group them by amino acid."""
    
    aaTable = refCodonTableDf.sort_values('aa_groups_sorted_index')['aa'].unique()
    # Sort codons following order defined in the reference codon table, grouping synomymous codons
    sort_order = {codon:i for i, codon in
                  enumerate(refCodonTableDf.sort_values(['aa_groups_sorted_index', 'alphabetical_sorted_index']).index)}
    if len(df.index.unique().difference(refCodonTableDf.index)) > 0:
        raise ValueError("DataFrame index contains a codon not found in the reference codon table index.")
    
    plotDf = df
    plotDf['sort_index'] = plotDf.index.get_level_values(0).map(lambda x: sort_order.get(x))
    plotDf = plotDf.sort_values('sort_index').drop('sort_index', axis=1)
    plotDf.index.name = 'codon'
    
    aaDict = refCodonTableDf['aa'].to_dict()
    plotDf['aa'] = plotDf.index.map(lambda x: aaDict[x])
    if useULetter:
        plotDf
    if addAminoAcidLetterToIndex:
        plotDf.index = plotDf.index.map(lambda x: '{}/{}'.format(refCodonTableDf.loc[x, 'aa'], x))
    if useULetter:
        plotDf.index = plotDf.index.map(lambda x: x[:-3] + re.sub('T', 'U', x[-3:]))
    if addEmptyRowBetweenGroups:
        dfDict = {}
        for aa, group in plotDf.groupby('aa'):
            df = group.copy().append(pd.Series(addEmptyRowBetweenGroupsFillValue, index=group.columns, name=''))
            dfDict[aa] = df
        plotDf = pd.concat([dfDict[aa] for aa in aaTable if aa in dfDict.keys()])
    if not addAminoAcidLetterAsColumn:
        plotDf = plotDf.drop('aa', axis=1)

    return plotDf


def find_triplet_in_sequence_in_frames(seq, triplet):
    tripletMatch = []
    for frame in [0,1,2]:
        codonList = list(extract_codons_list(seq, frame=frame, frameFromEnd=True))
        codonList = [(i, codon) for i, codon in enumerate(codonList)]
        frameShift = len(seq) - sum([len(codon) for i, codon in codonList]) - ((3 - frame) % 3)
        tripletMatchList = []
        k = frameShift
        for i, codon in codonList:
            if codon == triplet:
                tripletMatchList.append({'nucleotide_pos':k, 'codon_pos':i})
            k += len(codon)
        tripletMatch.append(tripletMatchList)
    return tripletMatch


def convert_BED_coverage_dataframe(path, basename, verbose=1):

    cov_p = pd.read_csv(str(path / (basename + '.strandp_coverage.bed')), sep='\t', header=None)
    cov_p.columns = ['ref','position','coverage_p']
    if verbose >= 2:
        print(len(cov_p))
        print(cov_p.head())

    cov_m = pd.read_csv(str(path / (basename + '.strandm_coverage.bed')), sep='\t', header=None)
    cov_m.columns = ['ref','position','coverage_m']
    if verbose >= 2:
        print(len(cov_m))
        print(cov_m.head())

    covDf = pd.merge(cov_p, cov_m[['position','coverage_m']], on='position', how='outer')
    covDf = covDf.sort_values('position')
    covDf.drop('ref', axis=1, inplace=True)
    covDf.fillna(0, inplace=True)

    # Check if the genome coverage file is 0-based or 1-based
    if covDf['position'].min() == 1:
        if verbose >= 1: print("BED coverage file uses 1-based index. Converting to 0-based.")
        covDf['position'] = covDf['position'] - 1
    elif covDf['position'].min() == 0:
        if verbose >= 1: print("BED coverage file uses 0-based index.")

    if verbose >= 1: print(len(covDf))
    return covDf


def import_roesti_coverage_df(sampleDf, path=None,
                              covStrandPlusSuffix='.strandp_coverage.bed', covStrandMinusSuffix='.strandm_coverage.bed',
                              verbose=1):

    dfList = []
    for i, sample in sampleDf.iterrows():

        if path is None:
            path1 = sample['path']
        else:
            path1 = path

        rna_cov_p = pd.read_csv(path1 / (sample['sample_filename_prefix'] + '.strandp_coverage.bed'),
                                sep='\t', header=None, nrows=None)
        rna_cov_p.columns = ['ref','position','coverage_p']

        rna_cov_m = pd.read_csv(path1 / (sample['sample_filename_prefix'] + '.strandm_coverage.bed'),
                                sep='\t', header=None, nrows=None)
        rna_cov_m.columns = ['ref','position','coverage_m']

        rnaCovDf = pd.merge(rna_cov_p, rna_cov_m, on=['ref', 'position'], how='outer')
        rnaCovDf = rnaCovDf.sort_values(['ref', 'position'])
#         rnaCovDf.fillna(0, inplace=True)
        rnaCovDf.set_index(['ref', 'position'], inplace=True)
        rnaCovDf.columns = pd.MultiIndex.from_product([[sample['sample']], [sample['replicate']],
                                                       ['coverage_plus', 'coverage_minus']],
                                                      names=['sample', 'replicate', 'coverage'])

        # Check if the genome coverage file is 0-based or 1-based
        if rnaCovDf.index.get_level_values('position').min() == 1:
            print("BED coverage file uses 1-based index. Converting to 0-based.")
            rnaCovDf = rnaCovDf.reset_index()
            rnaCovDf['position'] = rnaCovDf['position'] - 1
            rnaCovDf = rnaCovDf.set_index(['ref', 'position'])
        elif rnaCovDf.index.get_level_values('position').min() == 0:
            print("BED coverage file uses 0-based index.")

        dfList.append(rnaCovDf)

    rnaCovDf = pd.concat(dfList, axis=1)
    return rnaCovDf


def import_roesti_expression_df(sampleDf, path=None, verbose=1):

    dfList = []
    for i, sample in sampleDf.iterrows():

        if path is None:
            path1 = sample['path']
        else:
            path1 = path

        df = pd.read_csv(path1 / (sample['sample_filename'] + '.CDS_values.csv'), index_col=0)
        df['sample'] = sample['sample']
        df['replicate'] = sample['replicate']
        dfList.append(df)

    rnaDf = pd.concat(dfList, sort=True)
    # reorder the columns to make it nicer ;)
    cols = rnaDf.columns
    startCol = [c for c in cols if re.search(r'start_', c)][0]
    endCol = [c for c in cols if re.search(r'end_', c)][0]
    firstcols = ['sample', 'replicate', 'ref', 'id', 'strand', startCol, endCol]
    cols = firstcols + [c for c in cols if not c in firstcols]
    rnaDf = rnaDf[cols]
    return rnaDf


def overlap_1D(start1, end1, start2, end2):
    return max(0, min(end1, end2) - max(start1, start2))


def annot_region_overlap(annot, region_start, region_end, region_strand=None,
                         startCol='start', endCol='end', strandCol='strand', strand_specific=True):

    regionLength = region_end - region_start
    if annot[startCol] is None or np.isnan(annot[startCol]) or annot[endCol] is None or np.isnan(annot[endCol]):
        return None, None, None
    else:
        annotLength = annot[endCol] - annot[startCol]
        
        annot_location = None
        overlap = overlap_1D(region_start, region_end, annot[startCol], annot[endCol])
        if overlap == 0:
            annot_location = 'no_overlap'
        elif overlap < annotLength:
            annot_location = 'partial'
        elif overlap == annotLength:
            annot_location = 'inside'
        elif overlap == regionLength:
            annot_location = 'full'
        # if overlap > 0:
        #     print(region_start, region_end, annot[startCol], annot[endCol])
        #     print("annotId", annotId, "overlap", overlap, "annotLength", annotLength, "regionLength", regionLength, "annot_location", annot_location)
        
        if strand_specific and region_strand != annot[strandCol]:
            annot_location = 'no_overlap'

        return annot_location, overlap, annot[strandCol]


def find_annotations_in_region(annotDf, region_start, region_end, region_strand=None,
                               startCol='start', endCol='end', strandCol='strand', strand_specific=True):
    """
    Note: we use a very naive implementation which is slow.
    For faster algorithm, we could use the implementation of BEDtools, but we would have
    to create BED files first. For small annotation dataframe this is fast enough though.
    
    start and end should be 0-based inclusive-exclusive (biopython)
    """
    if region_start == region_end:
        return []
    if strand_specific:
        if region_strand not in ['+', '-']:
            raise ValueError("region_strand should be '+' or '-'")

    annotDict = annotDf[[startCol, endCol, strandCol]].to_dict(orient='index')

    overlapList = []
    for annotId, annot in annotDict.items():
        annot_location, overlap, strand = \
            annot_region_overlap(annot, region_start=region_start, region_end=region_end, region_strand=region_strand,
                                 startCol=startCol, endCol=endCol, strandCol=strandCol, strand_specific=strand_specific)
        overlapList.append({'id':annotId, 'overlap_type':annot_location, 'overlap_length':overlap, 'strand':strand})
    return overlapList
