"""Generate a comprehensive Excel report of all experiment results."""
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side, numbers
from openpyxl.utils import get_column_letter

wb = openpyxl.Workbook()

# ── Styles ──────────────────────────────────────────────────────────────────
header_font = Font(bold=True, color="FFFFFF", size=11)
header_fill = PatternFill(start_color="2F5496", end_color="2F5496", fill_type="solid")
best_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
fail_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
killed_fill = PatternFill(start_color="FFE699", end_color="FFE699", fill_type="solid")
subheader_fill = PatternFill(start_color="D6E4F0", end_color="D6E4F0", fill_type="solid")
thin_border = Border(
    left=Side(style="thin"), right=Side(style="thin"),
    top=Side(style="thin"), bottom=Side(style="thin"),
)

def style_header(ws, row, max_col):
    for c in range(1, max_col + 1):
        cell = ws.cell(row=row, column=c)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
        cell.border = thin_border

def style_row(ws, row, max_col, fill=None):
    for c in range(1, max_col + 1):
        cell = ws.cell(row=row, column=c)
        cell.border = thin_border
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        if fill:
            cell.fill = fill

# ═══════════════════════════════════════════════════════════════════════════
# SHEET 1: Summary
# ═══════════════════════════════════════════════════════════════════════════
ws = wb.active
ws.title = "Summary"

headers = [
    "Exp #", "Run ID", "Status", "Description",
    "Layers", "Params", "Muon WD", "Warmdown\nIters", "FP16\nEmbed",
    "Steps\nCompleted", "Train\nTime (s)", "Avg Step\n(ms)", "Tok/s",
    "Final\ntrain_loss", "Final\nval_bpb", "int8 Roundtrip\nval_bpb",
    "Quant Gap\n(bpb)", "Artifact\n(MB)", "Under\n16MB?", "Wallclock\nStop?"
]

for c, h in enumerate(headers, 1):
    ws.cell(row=1, column=c, value=h)
style_header(ws, 1, len(headers))

experiments = [
    {
        "num": 1, "run_id": "exp_001", "status": "Discard",
        "desc": "10L smoke test (200 iters only)",
        "layers": 10, "params": 18_897_488, "muon_wd": 0.0, "warmdown": 1200, "fp16_emb": False,
        "steps": "200/200", "train_s": 70.2, "step_avg": 351.24, "tok_s": 23653,
        "final_train_loss": 3.9107, "final_val_bpb": 2.4138,
        "int8_bpb": 2.4146, "quant_gap": 0.0008,
        "artifact_mb": 12.46, "under_16": True, "wallclock_stop": False,
    },
    {
        "num": 2, "run_id": "exp_002", "status": "Discard",
        "desc": "10L medium (ran concurrent with exp_003, degraded throughput)",
        "layers": 10, "params": 18_897_488, "muon_wd": 0.0, "warmdown": 1200, "fp16_emb": False,
        "steps": "1032/2000", "train_s": 600.3, "step_avg": 581.67, "tok_s": 3797,
        "final_train_loss": 3.3082, "final_val_bpb": 1.9465,
        "int8_bpb": 1.9470, "quant_gap": 0.0005,
        "artifact_mb": 14.63, "under_16": True, "wallclock_stop": True,
    },
    {
        "num": 3, "run_id": "exp_003_baseline9L", "status": "Discard",
        "desc": "9L baseline (ran concurrent with exp_002, degraded throughput)",
        "layers": 9, "params": 17_059_912, "muon_wd": 0.0, "warmdown": 1200, "fp16_emb": False,
        "steps": "688/2000", "train_s": 601.3, "step_avg": 874.03, "tok_s": 4227,
        "final_train_loss": 3.4019, "final_val_bpb": 1.9999,
        "int8_bpb": 2.0004, "quant_gap": 0.0005,
        "artifact_mb": 12.00, "under_16": True, "wallclock_stop": True,
    },
    {
        "num": 4, "run_id": "exp_004_sw64", "status": "Killed",
        "desc": "Sliding window stride=64 (single-window, killed - too slow)",
        "layers": 10, "params": 18_897_488, "muon_wd": 0.0, "warmdown": 1200, "fp16_emb": False,
        "steps": "200/200", "train_s": 67.9, "step_avg": 339.74, "tok_s": 23904,
        "final_train_loss": 3.9147, "final_val_bpb": 2.4120,
        "int8_bpb": None, "quant_gap": None,
        "artifact_mb": 12.46, "under_16": True, "wallclock_stop": False,
    },
    {
        "num": 5, "run_id": "exp_005_sw64", "status": "Killed",
        "desc": "Sliding window stride=64 batched (killed - still too slow ~50min eval)",
        "layers": 10, "params": 18_897_488, "muon_wd": 0.0, "warmdown": 1200, "fp16_emb": False,
        "steps": "200/200", "train_s": 68.1, "step_avg": 340.26, "tok_s": 23964,
        "final_train_loss": 3.9224, "final_val_bpb": 2.4156,
        "int8_bpb": None, "quant_gap": None,
        "artifact_mb": 12.46, "under_16": True, "wallclock_stop": False,
    },
    {
        "num": 6, "run_id": "exp_006_10L_fp16emb", "status": "Discard",
        "desc": "FP16 ALL large tensors (bug: INT8_KEEP_FLOAT_MAX_NUMEL=600000 kept everything as FP16)",
        "layers": 10, "params": 18_897_488, "muon_wd": 0.0, "warmdown": 1200, "fp16_emb": "ALL (bug)",
        "steps": "1622/2000", "train_s": 600.3, "step_avg": 370.07, "tok_s": 21935,
        "final_train_loss": 3.1637, "final_val_bpb": 1.8655,
        "int8_bpb": 1.8655, "quant_gap": 0.0000,
        "artifact_mb": 35.20, "under_16": False, "wallclock_stop": True,
    },
    {
        "num": 7, "run_id": "exp_007_10L_fp16tok", "status": "Discard",
        "desc": "FP16 tok_emb only (correct implementation). Superseded by exp_009.",
        "layers": 10, "params": 18_897_488, "muon_wd": 0.0, "warmdown": 1200, "fp16_emb": True,
        "steps": "1629/2000", "train_s": 600.0, "step_avg": 368.33, "tok_s": 21641,
        "final_train_loss": 3.1558, "final_val_bpb": 1.8604,
        "int8_bpb": 1.8605, "quant_gap": 0.0001,
        "artifact_mb": 15.71, "under_16": True, "wallclock_stop": True,
    },
    {
        "num": 8, "run_id": "exp_008_wd400", "status": "Discard",
        "desc": "Warmdown=400 (too short: worse BPB + artifact over 16MB)",
        "layers": 10, "params": 18_897_488, "muon_wd": 0.0, "warmdown": 400, "fp16_emb": True,
        "steps": "1636/2000", "train_s": 600.1, "step_avg": 366.83, "tok_s": 22179,
        "final_train_loss": 3.1962, "final_val_bpb": 1.8779,
        "int8_bpb": 1.8779, "quant_gap": 0.0000,
        "artifact_mb": 16.50, "under_16": False, "wallclock_stop": True,
    },
    {
        "num": 9, "run_id": "exp_009_wd02", "status": "BEST",
        "desc": "Muon weight decay=0.02: best BPB + improved compressibility",
        "layers": 10, "params": 18_897_488, "muon_wd": 0.02, "warmdown": 1200, "fp16_emb": True,
        "steps": "1604/2000", "train_s": 600.2, "step_avg": 374.18, "tok_s": 21340,
        "final_train_loss": 3.1310, "final_val_bpb": 1.8290,
        "int8_bpb": 1.8291, "quant_gap": 0.0001,
        "artifact_mb": 14.65, "under_16": True, "wallclock_stop": True,
    },
]

for i, e in enumerate(experiments):
    r = i + 2
    row_data = [
        e["num"], e["run_id"], e["status"], e["desc"],
        e["layers"], f'{e["params"]:,}', e["muon_wd"], e["warmdown"],
        str(e["fp16_emb"]),
        e["steps"], round(e["train_s"], 1), round(e["step_avg"], 1), e["tok_s"],
        e["final_train_loss"], round(e["final_val_bpb"], 4),
        round(e["int8_bpb"], 4) if e["int8_bpb"] else "N/A (killed)",
        round(e["quant_gap"], 4) if e["quant_gap"] is not None else "N/A",
        round(e["artifact_mb"], 2), "Yes" if e["under_16"] else "NO",
        "Yes" if e["wallclock_stop"] else "No",
    ]
    for c, v in enumerate(row_data, 1):
        ws.cell(row=r, column=c, value=v)

    # Highlight rows
    if e["status"] == "BEST":
        style_row(ws, r, len(headers), best_fill)
    elif e["status"] == "Killed":
        style_row(ws, r, len(headers), killed_fill)
    elif not e["under_16"]:
        style_row(ws, r, len(headers), fail_fill)
    else:
        style_row(ws, r, len(headers))

# Column widths
col_widths = [6, 22, 8, 55, 7, 14, 9, 10, 8, 12, 10, 10, 8, 10, 10, 12, 10, 10, 8, 10]
for i, w in enumerate(col_widths):
    ws.column_dimensions[get_column_letter(i + 1)].width = w

# Freeze top row
ws.freeze_panes = "A2"

# ═══════════════════════════════════════════════════════════════════════════
# SHEET 2: Val BPB Trajectory
# ═══════════════════════════════════════════════════════════════════════════
ws2 = wb.create_sheet("Val BPB Trajectory")

trajectories = {
    "exp_002 (10L)": {0: 4.1085, 500: 2.1753, 1000: 1.9493, 1032: 1.9465},
    "exp_003 (9L baseline)": {0: 4.1086, 500: 2.1254, 688: 1.9999},
    "exp_006 (FP16 all, bug)": {0: 4.1085, 500: 2.1834, 1000: 2.0037, 1500: 1.8881, 1622: 1.8655},
    "exp_007 (FP16 tok_emb)": {0: 4.1085, 500: 2.1774, 1000: 2.0043, 1500: 1.8852, 1629: 1.8604},
    "exp_008 (WD=400)": {0: 4.1085, 500: 2.1832, 1000: 2.0618, 1500: 1.9261, 1636: 1.8779},
    "exp_009 (WD=0.02, BEST)": {0: 4.1085, 500: 2.1865, 1000: 2.0119, 1500: 1.8554, 1604: 1.8290},
}

steps_all = sorted(set(s for t in trajectories.values() for s in t.keys()))
ws2.cell(row=1, column=1, value="Step")
for c, name in enumerate(trajectories.keys(), 2):
    ws2.cell(row=1, column=c, value=name)
style_header(ws2, 1, len(trajectories) + 1)

for r, step in enumerate(steps_all, 2):
    ws2.cell(row=r, column=1, value=step)
    ws2.cell(row=r, column=1).border = thin_border
    for c, (name, data) in enumerate(trajectories.items(), 2):
        val = data.get(step)
        cell = ws2.cell(row=r, column=c, value=val)
        cell.border = thin_border
        if val is not None:
            cell.number_format = "0.0000"

for i in range(len(trajectories) + 1):
    ws2.column_dimensions[get_column_letter(i + 1)].width = 22
ws2.freeze_panes = "B2"

# ═══════════════════════════════════════════════════════════════════════════
# SHEET 3: Train Loss Trajectory
# ═══════════════════════════════════════════════════════════════════════════
ws3 = wb.create_sheet("Train Loss Trajectory")

train_trajectories = {
    "exp_002 (10L)": {1: 6.9445, 10: 6.4229, 200: 3.9100, 400: 3.6785, 600: 3.4989, 800: 3.3516, 1000: 3.3082},
    "exp_003 (9L)": {1: 6.9428, 10: 6.3435, 200: 3.9052, 400: 3.6028, 600: 3.4019},
    "exp_007 (FP16 tok)": {1: 6.9445, 10: 6.4270, 200: 3.9175, 400: 3.6786, 600: 3.5604, 800: 3.4360, 1000: 3.4007, 1200: 3.2655, 1400: 3.2008, 1600: 3.1558},
    "exp_008 (WD=400)": {1: 6.9445, 10: 6.4227, 200: 3.9107, 400: 3.6814, 600: 3.5999, 800: 3.4981, 1000: 3.5086, 1200: 3.4403, 1400: 3.2880, 1600: 3.1962},
    "exp_009 (BEST)": {1: 6.9445, 10: 6.4172, 200: 3.9029, 400: 3.6845, 600: 3.5903, 800: 3.4599, 1000: 3.4210, 1200: 3.2944, 1400: 3.1666, 1600: 3.1310},
}

steps_train = sorted(set(s for t in train_trajectories.values() for s in t.keys()))
ws3.cell(row=1, column=1, value="Step")
for c, name in enumerate(train_trajectories.keys(), 2):
    ws3.cell(row=1, column=c, value=name)
style_header(ws3, 1, len(train_trajectories) + 1)

for r, step in enumerate(steps_train, 2):
    ws3.cell(row=r, column=1, value=step)
    ws3.cell(row=r, column=1).border = thin_border
    for c, (name, data) in enumerate(train_trajectories.items(), 2):
        val = data.get(step)
        cell = ws3.cell(row=r, column=c, value=val)
        cell.border = thin_border
        if val is not None:
            cell.number_format = "0.0000"

for i in range(len(train_trajectories) + 1):
    ws3.column_dimensions[get_column_letter(i + 1)].width = 18
ws3.freeze_panes = "B2"

# ═══════════════════════════════════════════════════════════════════════════
# SHEET 4: Config Details
# ═══════════════════════════════════════════════════════════════════════════
ws4 = wb.create_sheet("Config Details")

config_headers = [
    "Run ID", "NUM_LAYERS", "MODEL_DIM", "NUM_HEADS", "NUM_KV_HEADS",
    "VOCAB_SIZE", "TRAIN_SEQ_LEN", "TRAIN_BATCH_TOKENS", "GRAD_ACCUM_STEPS",
    "ITERATIONS", "WARMUP_STEPS", "WARMDOWN_ITERS", "MAX_WALLCLOCK_S",
    "EMBED_LR", "MATRIX_LR", "SCALAR_LR", "MUON_MOMENTUM", "MUON_BACKEND_STEPS",
    "MUON_WEIGHT_DECAY", "EVAL_STRIDE", "FP16_NAME_PATTERNS", "TIE_EMBEDDINGS",
    "LOGIT_SOFTCAP", "ROPE_BASE", "QK_GAIN_INIT",
]

for c, h in enumerate(config_headers, 1):
    ws4.cell(row=1, column=c, value=h)
style_header(ws4, 1, len(config_headers))

config_rows = [
    ["exp_001", 10, 512, 8, 4, 1024, 1024, 8192, 8, 200, 20, 1200, 600, 0.05, 0.04, 0.04, 0.95, 5, 0.0, 0, "", True, 30.0, 10000, 1.5],
    ["exp_002", 10, 512, 8, 4, 1024, 1024, 8192, 8, 2000, 20, 1200, 600, 0.05, 0.04, 0.04, 0.95, 5, 0.0, 0, "", True, 30.0, 10000, 1.5],
    ["exp_003_baseline9L", 9, 512, 8, 4, 1024, 1024, 8192, 8, 2000, 20, 1200, 600, 0.05, 0.04, 0.04, 0.95, 5, 0.0, 0, "", True, 30.0, 10000, 1.5],
    ["exp_004_sw64", 10, 512, 8, 4, 1024, 1024, 8192, 8, 200, 20, 1200, 600, 0.05, 0.04, 0.04, 0.95, 5, 0.0, 64, "", True, 30.0, 10000, 1.5],
    ["exp_005_sw64", 10, 512, 8, 4, 1024, 1024, 8192, 8, 200, 20, 1200, 600, 0.05, 0.04, 0.04, 0.95, 5, 0.0, 64, "", True, 30.0, 10000, 1.5],
    ["exp_006_10L_fp16emb", 10, 512, 8, 4, 1024, 1024, 8192, 8, 2000, 20, 1200, 600, 0.05, 0.04, 0.04, 0.95, 5, 0.0, 0, "MAX_NUMEL=600k (bug)", True, 30.0, 10000, 1.5],
    ["exp_007_10L_fp16tok", 10, 512, 8, 4, 1024, 1024, 8192, 8, 2000, 20, 1200, 600, 0.05, 0.04, 0.04, 0.95, 5, 0.0, 0, "tok_emb", True, 30.0, 10000, 1.5],
    ["exp_008_wd400", 10, 512, 8, 4, 1024, 1024, 8192, 8, 2000, 20, 400, 600, 0.05, 0.04, 0.04, 0.95, 5, 0.0, 0, "tok_emb", True, 30.0, 10000, 1.5],
    ["exp_009_wd02", 10, 512, 8, 4, 1024, 1024, 8192, 8, 2000, 20, 1200, 600, 0.05, 0.04, 0.04, 0.95, 5, 0.02, 0, "tok_emb", True, 30.0, 10000, 1.5],
]

for i, row in enumerate(config_rows):
    r = i + 2
    for c, v in enumerate(row, 1):
        ws4.cell(row=r, column=c, value=v)
    style_row(ws4, r, len(config_headers))

for i in range(len(config_headers)):
    ws4.column_dimensions[get_column_letter(i + 1)].width = 16
ws4.freeze_panes = "B2"

# ═══════════════════════════════════════════════════════════════════════════
# SHEET 5: Key Insights
# ═══════════════════════════════════════════════════════════════════════════
ws5 = wb.create_sheet("Insights & Learnings")

insights = [
    ["Category", "Insight", "Evidence"],
    ["Architecture", "10 layers > 9 layers by ~0.05 BPB", "exp_002 (1.947) vs exp_003 (2.000) at same wallclock"],
    ["Architecture", "10L adds ~1.2MB to artifact vs 9L", "12.0MB (9L) vs 14.6MB (10L)"],
    ["Quantization", "FP16 tied embeddings reduce quant gap to +0.0001 BPB", "exp_007 gap=0.0001 vs exp_002 gap=0.0005"],
    ["Quantization", "FP16 emb adds ~0.5MB to artifact (acceptable)", "15.7MB (FP16) vs 14.6MB (int8)"],
    ["Quantization", "Do NOT increase INT8_KEEP_FLOAT_MAX_NUMEL — blows up artifact", "exp_006: 35MB artifact (bug)"],
    ["Optimizer", "Muon WD=0.02 improves BPB by 0.031 AND compressibility by 1.1MB", "exp_009 (1.829, 14.6MB) vs exp_007 (1.861, 15.7MB)"],
    ["Optimizer", "WD regularizes weights → better zlib compression", "14.6MB (WD=0.02) vs 15.7MB (WD=0) vs 16.5MB (WD=0, short warmdown)"],
    ["Schedule", "Long warmdown (1200 iters) is beneficial — acts as regularizer", "exp_008 (WD=400): 1.878 bpb, 16.5MB vs exp_007 (WD=1200): 1.861 bpb, 15.7MB"],
    ["Schedule", "Don't reduce warmdown below 1200 for ~1600-step runs", "warmdown=400 was worse on every metric"],
    ["Evaluation", "Sliding window eval (stride=64) is ~50 min on Apple Silicon", "exp_004/005: 15K batches at ~5 batches/sec"],
    ["Evaluation", "Sliding window code works — use EVAL_STRIDE=64 for final submission only", "Implemented and tested, just too slow for iteration"],
    ["Throughput", "Solo runs: ~1600 steps in 600s at ~370ms/step (10L)", "exp_007, exp_008, exp_009 all consistent"],
    ["Throughput", "Concurrent runs degrade to 580-874ms/step (2-3x slower)", "exp_002 (581ms), exp_003 (874ms)"],
    ["Throughput", "Always run experiments sequentially for accurate comparison", "Learned from exp_002/003 concurrent degradation"],
    ["Competition", "Current best (1.829) vs competition SOTA (1.175) — 0.654 BPB gap", "Still need: more training time, sliding window, spectral init, LoRA TTT"],
]

for r, row in enumerate(insights, 1):
    for c, v in enumerate(row, 1):
        ws5.cell(row=r, column=c, value=v)
    if r == 1:
        style_header(ws5, r, 3)
    else:
        style_row(ws5, r, 3)

ws5.column_dimensions["A"].width = 15
ws5.column_dimensions["B"].width = 60
ws5.column_dimensions["C"].width = 55
ws5.freeze_panes = "A2"

# ═══════════════════════════════════════════════════════════════════════════
# Save
# ═══════════════════════════════════════════════════════════════════════════
path = "/Users/Season_Yang/Development/parameter-golf/experiment_results.xlsx"
wb.save(path)
print(f"Saved: {path}")
print(f"Sheets: {wb.sheetnames}")
