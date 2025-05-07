#!/usr/bin/env python3
import math, csv, sys
from Bio import AlignIO
from collections import Counter

# ───── CONFIG ────────────────────────────────────────────────────────────────
ALIGN_FILE         = "ispG.aln-clustalw"     # your MUSCLE CLUSTAL output
ECOLI_ID_SUBSTR    = "AAC75568"              # substring to identify the E. coli row
# Your known binding‐site columns in the *alignment* (1-based!)
MOTIF_ALIGN_COLS   = [815, 818, 863, 870]
WINDOW_OFFSET      = 50                       # expand ± this many columns around motifs
TOP_N_ENTROPY      = 15                       # number of top‐entropy sites to penalize
# ──────────────────────────────────────────────────────────────────────────────

# 1) load the alignment
try:
    aln = AlignIO.read(ALIGN_FILE, "clustal")
except Exception as e:
    sys.exit(f"❌ Error reading alignment: {e}")
L = aln.get_alignment_length()

# 2) find the E. coli sequence record
try:
    ecoli = next(r for r in aln if ECOLI_ID_SUBSTR in r.id)
except StopIteration:
    sys.exit(f"❌ No sequence with '{ECOLI_ID_SUBSTR}' in its header found.")

# 3) turn your 1-based alignment columns into 0-based, check bounds
motif_cols = [c - 1 for c in MOTIF_ALIGN_COLS]
for c in motif_cols:
    if c < 0 or c >= L:
        sys.exit(f"❌ Motif column {c+1} out of range 1–{L}")

# 4) define the binding‐site window around those motif columns
minc, maxc = min(motif_cols), max(motif_cols)
win_start = max(0, minc - WINDOW_OFFSET)
win_end   = min(L - 1, maxc + WINDOW_OFFSET)
binding_cols = list(range(win_start, win_end + 1))

print(f"▶︎ Using binding window columns {win_start+1}–{win_end+1} "
      f"(size {len(binding_cols)}) around motifs {MOTIF_ALIGN_COLS}")

# 5) compute per‐column Shannon entropy (ignoring gaps)
entropy = []
for col in zip(*aln):
    freq  = Counter(a for a in col if a != "-")
    total = sum(freq.values())
    if total == 0:
        entropy.append(0.0)
    else:
        entropy.append(-sum((v/total) * math.log2(v/total) for v in freq.values()))

# 6) select the top-N most variable columns outside the binding window
nonbind = [i for i in range(L) if i not in binding_cols]
top_entropy_cols = sorted(nonbind, key=lambda i: entropy[i], reverse=True)[:TOP_N_ENTROPY]

# 7) build & write the feature matrix
with open("ispG_MSA_features.csv", "w", newline="") as outf:
    w = csv.writer(outf)
    w.writerow([
        "id",
        "pid_total",
        "mean_entropy_bind",
        "median_entropy_bind",
        "penalty_bind",
        f"penalty_top{TOP_N_ENTROPY}",
        "gaps_bind"
    ])

    for rec in aln:
        seq = rec.seq

        # a) overall % identity to E. coli
        matches   = sum(1 for a, b in zip(seq, ecoli.seq) if a == b and a != "-")
        pid_total = matches / L

        # b) mean & median entropy in the binding window
        bind_vals = [entropy[i] for i in binding_cols]
        mean_bind = sum(bind_vals) / len(bind_vals)
        med_bind  = sorted(bind_vals)[len(bind_vals)//2]

        # c) penalties where the ortholog *differs* from E. coli
        pen_bind = sum(entropy[i] for i in binding_cols if seq[i] != ecoli.seq[i])
        pen_topN = sum(entropy[i] for i in top_entropy_cols if seq[i] != ecoli.seq[i])

        # d) gap count in the binding window
        gaps_bind = sum(1 for i in binding_cols if seq[i] == "-")

        w.writerow([
            rec.id,
            f"{pid_total:.4f}",
            f"{mean_bind:.4f}",
            f"{med_bind:.4f}",
            f"{pen_bind:.4f}",
            f"{pen_topN:.4f}",
            gaps_bind
        ])

print(f"✅ ispG_MSA_features.csv generated for {len(aln)} sequences.")
