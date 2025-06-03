#!/usr/bin/env python3
import math, csv, sys
from collections import Counter
from Bio import AlignIO, Align

align_file = "ispG_full.aln-clustalw"
ref_id = "AAC75568.1"

#make sure to put them -1 of what you see in the alignment output file
motif_cols = [809, 812, 860, 867] #[814, 817, 862, 869] this is for the previous alignment 
window = 25  # expand ± this many columns around motifs
lowest_entropy = 50 # number of top‐entropy sites to penalize
nonzero = 1e-9

# 1) load the alignment
try:
    aln = AlignIO.read(align_file, "clustal")
except Exception as e:
    sys.exit(f"Error reading alignment: {e}")
L = aln.get_alignment_length()

# 2) find the E. coli sequence record
try:
    ecoli = next(r for r in aln if ref_id in r.id)
except StopIteration:
    sys.exit(f"No sequence with '{ref_id}' in its header found.")

# 3) turn your 1-based alignment columns into 0-based, check bounds
for c in motif_cols:
    if c < 0 or c >= L:
        sys.exit(f"Motif column {c+1} out of range 1–{L}")

# 4) define the binding‐site window around those motif columns
minc, maxc = min(motif_cols), max(motif_cols)
win_start = max(0, minc - window)
win_end   = min(L - 1, maxc + window)
binding_cols = list(range(win_start, win_end + 1))

print(f"Using binding window columns {win_start+1}–{win_end+1}(size {len(binding_cols)}) around motifs {motif_cols}")

# 5) compute per‐column Shannon entropy (ignoring gaps)
entropy = []
for col in zip(*aln):
    freq = Counter(a for a in col if a != "-")
    total = sum(freq.values())
    if total == 0:
        entropy.append(0.0)
    else:
        entropy.append(-sum((v/total) * math.log2(v/total) for v in freq.values()))

# 6) select the top-N most variable columns outside the binding window
nonbind = [i for i in range(L) if (i not in binding_cols and ecoli.seq[i] != '-')]
lowest_entropy_cols = sorted(nonbind, key=lambda i: entropy[i])[:lowest_entropy] #put reverse=True if I want the opposite

# 7) build & write the feature matrix
with open("ispG_MSA_features_full.csv", "w", newline="") as outf:
    w = csv.writer(outf)
    w.writerow([
        "id",
        "pid_total",
        "penalty_bind",
        f"penalty_top{lowest_entropy}",
        "gaps_bind"])

    for rec in aln:
        seq = rec.seq

        # a) overall % identity to E. coli
        aligner = Align.PairwiseAligner(match_score = 1.0)
        aligner.mismatch_score = 0
        alignments = aligner.align(ecoli.seq, seq)

        pid = alignments.score / alignments[0].length

        # c) penalties where the ortholog *differs* from E. coli
        pen_bind = sum(1/(entropy[i] + nonzero) for i in binding_cols if seq[i] != ecoli.seq[i])
        pen_topN = sum(1/(entropy[i] + nonzero) for i in lowest_entropy_cols if seq[i] != ecoli.seq[i])

        # d) gap count in the binding window
        gaps_bind = sum(1 for i in binding_cols if seq[i] == "-")

        w.writerow([
            rec.id,
            f"{pid:.4f}",
            f"{pen_bind:.4f}",
            f"{pen_topN:.4f}",
            gaps_bind
        ])

print(f"ispG_MSA_features_full.csv generated for {len(aln)} sequences.")
