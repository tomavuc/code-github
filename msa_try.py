#!/usr/bin/env python3

import math, csv, sys
from Bio import AlignIO
from collections import Counter

# ───── CONFIG ────────────────────────────────────────────────────────────────
ALIGN_FILE       = "ispG.aln-clustalw"   # your MUSCLE output
ECOLI_ID_SUBSTR  = "AAC75568.1"            # substring to identify the E. coli record
BIND_WINDOW      = (856, 879)            # 1-based positions in the ungapped E. coli sequence
TOP_N_ENTROPY    = 15                    # how many high-entropy sites to use for penalization
# ──────────────────────────────────────────────────────────────────────────────

# 1) load alignment
try:
    aln = AlignIO.read(ALIGN_FILE, "clustal")
except Exception as e:
    sys.exit(f"Error reading alignment: {e}")

# 2) find the E. coli sequence record
try:
    ecoli = next(r for r in aln if ECOLI_ID_SUBSTR in r.id)
except StopIteration:
    sys.exit(f"Could not find any record containing '{ECOLI_ID_SUBSTR}' in its header.")

# 3) map the binding-window positions to alignment column indices
bind_start, bind_end = BIND_WINDOW
binding_cols = []
seq_pos = 0
for col_idx, aa in enumerate(ecoli.seq):
    if aa != "-":
        seq_pos += 1
    if bind_start <= seq_pos <= bind_end:
        binding_cols.append(col_idx)
if not binding_cols:
    sys.exit(
      f"No binding columns mapped for window {BIND_WINDOW}; "
      f"verify that your window matches the ungapped E. coli sequence."
    )

# 4) compute per-column Shannon entropy (ignore gaps)
entropy = []
for col in zip(*aln):
    freq = Counter(a for a in col if a != "-")
    total = sum(freq.values())
    if total == 0:
        entropy.append(0.0)
    else:
        entropy.append(-sum((v/total) * math.log2(v/total) for v in freq.values()))

# 5) select top-N most variable columns outside the binding window
nonbind = [i for i in range(len(entropy)) if i not in binding_cols]
top_entropy_cols = sorted(nonbind, key=lambda i: entropy[i], reverse=True)[:TOP_N_ENTROPY]

# 6) build feature matrix
with open("ispG_MSA_features.csv", "w", newline="") as outf:
    writer = csv.writer(outf)
    writer.writerow([
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

        # 6a) pairwise identity
        matches = sum(1 for a, b in zip(seq, ecoli.seq) if (a == b and a != "-"))
        pid_total = matches / len(seq)

        # 6b) mean & median entropy in binding window
        bind_ent = [entropy[i] for i in binding_cols]
        mean_bind = sum(bind_ent) / len(bind_ent)
        med_bind  = sorted(bind_ent)[len(bind_ent)//2]

        # 6c) penalties for deviations
        pen_bind = sum(entropy[i] for i in binding_cols if seq[i] != ecoli.seq[i])
        pen_topN = sum(entropy[i] for i in top_entropy_cols if seq[i] != ecoli.seq[i])

        # 6d) gap count in binding window
        gaps_bind = sum(1 for i in binding_cols if seq[i] == "-")

        writer.writerow([
            rec.id,
            f"{pid_total:.4f}",
            f"{mean_bind:.4f}",
            f"{med_bind:.4f}",
            f"{pen_bind:.4f}",
            f"{pen_topN:.4f}",
            gaps_bind
        ])

print("✅ Features written to ispG_MSA_features.csv")
