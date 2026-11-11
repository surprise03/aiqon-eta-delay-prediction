"""Build the 2-3 page technical report (PDF) from the pipeline outputs.

Usage: python report/build_report.py [--author "Your Name"]
All numbers are read from outputs/, so the report always matches the code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
OUT, FIG = ROOT / "outputs", ROOT / "outputs" / "figures"

ss = getSampleStyleSheet()
BODY = ParagraphStyle("b", parent=ss["BodyText"], fontSize=8.6, leading=11, alignment=TA_JUSTIFY, spaceAfter=3)
H1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=15, leading=18, spaceAfter=2)
SUB = ParagraphStyle("sub", parent=BODY, alignment=1, textColor=colors.HexColor("#555555"))
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=10.5, leading=13, spaceBefore=6, spaceAfter=2,
                    textColor=colors.HexColor("#1b4965"))
CAP = ParagraphStyle("cap", parent=BODY, fontSize=7.4, leading=9, textColor=colors.HexColor("#444444"))
MONO = ParagraphStyle("mono", fontName="Courier", fontSize=6.3, leading=7.6, backColor=colors.HexColor("#f4f6f8"),
                      borderPadding=4, leftIndent=4, rightIndent=4)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=10, bulletIndent=2, spaceAfter=1)


def P(t, s=BODY):
    return Paragraph(t, s)


def bullets(items):
    return [Paragraph(i, BUL, bulletText="•") for i in items]


def table(data, widths, header_bg="#1b4965", font=7.4, wrap=False):
    if wrap:  # wrap long text cells so they never overflow their column
        cell = ParagraphStyle("cell", fontName="Helvetica", fontSize=font, leading=font + 1.5)
        head = ParagraphStyle("head", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
        data = [[Paragraph(str(c), head if i == 0 else cell) for c in row] for i, row in enumerate(data)]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", font), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", font),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f5")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c8d0d8")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
    ]))
    return t


def img(name, width_cm):
    from reportlab.lib.utils import ImageReader
    path = FIG / name
    w, h = ImageReader(str(path)).getSize()
    return Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w)


def build(author: str):
    s = json.loads((OUT / "summary.json").read_text())
    m = pd.read_csv(OUT / "metrics_by_checkpoint.csv")
    ho = pd.read_csv(OUT / "holdout_metrics.csv")
    prop = pd.read_csv(OUT / "propagation_factors.csv")
    rob = pd.read_csv(OUT / "missing_data_robustness.csv")
    att = s["attribution_check"]
    # explanation of the first example shipment at the port-arrival checkpoint
    ex = (OUT / "example_explanations.txt").read_text(encoding="utf-8").split("=" * 100)[0]
    blocks = [b for b in ex.split("Shipment ") if "C2_port_arrival" in b]
    ex_text = ("Shipment " + blocks[0]).strip() if blocks else ex.strip()
    actual_line = [ln for ln in ex.splitlines() if "ACTUAL" in ln]
    if actual_line:
        ex_text += "\n\n" + actual_line[0]

    mi = m.set_index("checkpoint")
    c0, c2, c3 = mi.loc["C0_booking"], mi.loc["C2_port_arrival"], mi.loc["C3_customs_release"]
    imp_c2 = 1 - c2.mae_gbm / c2.mae_carry_forward
    imp_c3 = 1 - c3.mae_gbm / c3.mae_carry_forward

    def pr(cong, arr, col="median_downstream_added_h"):
        r = prop[(prop.congestion.str.startswith(cong)) & (prop.arrival == arr)]
        return float(r[col].iloc[0]) if len(r) else float("nan")

    story = [
        P("Predicting Shipment ETA and Delay Propagation", H1),
        P(f"Aiqon – Take-Home Assignment &nbsp;|&nbsp; {author}", SUB),
        Spacer(1, 4),
        P("1. Problem framing and definitions", H2),
        P("A shipment passes through six stages: <b>origin departure → ocean transit → berth → discharge "
          "→ customs release → inland delivery</b>. Every stage has a scheduled and an actual timestamp. "
          "<b>Delay</b> at a milestone is <i>actual − scheduled</i> (hours); the target is the delay of final "
          "delivery, and a shipment is <i>late</i> if it exceeds 24h. Because ETA is not static, the system re-estimates "
          "it at four <b>checkpoints</b> (booking, departure, port arrival, customs release), using only information "
          "available at that moment. ETA is defined as"),
        P("<para alignment='center'><b>ETA = scheduled delivery + delay already incurred + predicted remaining delay</b></para>"),
        P("Predicting the <i>remaining</i> delay (rather than the total) is the key design choice: it forces the model to "
          "learn whether delay already on the books will be <b>absorbed</b> by downstream buffers or <b>amplified</b> "
          "by missed windows – i.e. to model propagation explicitly, and it makes explanations cleanly separable "
          "into “what has happened” (exact) and “what will happen” (model)."),

        P("2. Data: a causal shipment simulator", H2),
        P(f"Public datasets rarely link per-stage timestamps to their causes, so I built a mechanism-based simulator "
          f"(<font face='Courier'>src/simulate.py</font>) over 6 real lanes (e.g. Shanghai→Los Angeles, "
          f"Shenzhen→Rotterdam via Suez), 4 carriers with different schedule reliability, and 4 cargo types. "
          f"Delays are never added to the final date directly; they arise from the mechanisms below and propagate "
          f"through the timeline. {s['n']} shipments are used for modelling plus 60 unseen “live” shipments "
          f"for a hold-out test and the demo. Calibration: {s['delayed_share']:.0%} of shipments are late (>24h), "
          f"median delay {s['median_delay_h']:.0f}h, heavy right tail – consistent with published container "
          f"schedule-reliability levels of roughly 50–65%."),
        table([["Stage", "Mechanism (ground truth)", "Propagation effect"],
               ["Origin", "Gamma delay ~ carrier reliability, origin congestion; rollover to next sailing (+7d), more likely in peak season", "Source"],
               ["Ocean", "Schedule padding (~4%) eaten by weather; Suez/Panama disruption (+2–12d); carriers speed up to recover", "Absorbs / shocks"],
               ["Berth", "Wait = f(congestion³); weekly AR(1) congestion per port; >24h late → berth window lost, back of queue", "Amplifies (lateness × congestion)"],
               ["Discharge", "Yard productivity collapses above 60% congestion; volume effect", "Amplifies"],
               ["Customs", "Docs incomplete +24h; inspection p = port × cargo × docs (+2–5d); no weekend processing", "Calendar amplification"],
               ["Inland", "Fixed truck appointment: early release waits; late release must rebook (scarcer in peak)", "Absorbs / amplifies"]],
              [1.6 * cm, 12.0 * cm, 4.0 * cm], wrap=True),
        Spacer(1, 2),
        P("Ground truth is the actual delivery time; the simulator also stores the exact per-stage delay increments "
          "(they sum to the final delay), which lets me <i>validate explanations</i> – impossible on real data. "
          "To mimic EDI/AIS gaps, 5–15% of intermediate events (arrival, berth, discharge, customs release) are "
          "removed in the observed dataset."),

        P("3. Features", H2),
        *bullets([
            "<b>Static (booking):</b> carrier reliability, origin congestion, forecast destination congestion and weather "
            "(noisy forecasts, not truth), chokepoint exposure, voyage/inland length, volume, peak season, "
            "<i>inspection risk</i> = port strictness × cargo risk × documentation, scheduled release on a weekend.",
            "<b>Propagation / interaction features:</b> departure and arrival delay, delay accrued at sea, "
            "<i>missed berth window</i> (>24h late), <i>lateness × congestion</i>, early-arrival (absorbed at anchor), "
            "<i>projected clearance hitting a weekend</i> (hours of working time before Saturday), "
            "<i>slack to the truck appointment</i> and <i>missed appointment</i>.",
            "<b>Missing data:</b> a missing arrival event is reconstructed from the berth time minus the expected "
            "congestion-dependent wait (flagged); remaining gaps are left as NaN, handled natively by the trees.",
            "Routes are described by their attributes (length, chokepoint, inland leg) rather than one-hot IDs, so the "
            "model can generalise to a new lane.",
        ]),

        P("4. Models and evaluation protocol", H2),
        P("Per checkpoint: a gradient-boosted tree regressor (scikit-learn HistGradientBoosting, absolute-error loss, "
          "depth 3, chosen for small data, non-linearity and native NaN support) predicts remaining delay; a classifier "
          "gives P(late&nbsp;>24h); quantile models with <b>conformal calibration</b> (CQR) give an 80% ETA band. "
          "Baselines: schedule (assume on time), carry-forward of the current delay, carry-forward + route median, and "
          "a ridge regression on the same features. Evaluation: 5-fold CV repeated 3× (MAE in hours, ROC-AUC, "
          "interval coverage) plus the untouched 60-shipment hold-out."),
    ]

    rows = [["Checkpoint", "Schedule", "Carry-fwd", "+Route med.", "Ridge", "GBM (ours)", "Hold-out GBM", "AUC", "80% cover."]]
    names = {"C0_booking": "Booking", "C1_departed": "Departed", "C2_port_arrival": "Port arrival",
             "C3_customs_release": "Customs released"}
    hoi = ho.set_index("checkpoint")
    for _, r in m.iterrows():
        rows.append([names[r.checkpoint], f"{r.mae_schedule:.1f}", f"{r.mae_carry_forward:.1f}",
                     f"{r.mae_carry_plus_route_mean:.1f}", f"{r.mae_ridge:.1f}", f"<b>{r.mae_gbm:.1f}</b>",
                     f"{hoi.loc[r.checkpoint].mae_gbm:.1f}", f"{r.auc:.2f}", f"{r.p10_p90_coverage:.0%}"])
    rows = [[Paragraph(c, ParagraphStyle("t", fontSize=7.4, leading=8.5)) if isinstance(c, str) and "<b>" in c else c
             for c in row] for row in rows]
    story += [
        P("5. Results", H2),
        KeepTogether([table(rows, [2.6 * cm, 1.5 * cm, 1.6 * cm, 1.9 * cm, 1.4 * cm, 1.9 * cm, 2.2 * cm, 1.2 * cm, 1.7 * cm]),
                      P("Table 1 – MAE of final delivery delay (hours), CV mean; hold-out on 60 unseen shipments.", CAP)]),
        P(f"The model beats every baseline at every checkpoint. At booking little is knowable "
          f"(MAE {c0.mae_gbm:.0f}h vs {c0.mae_schedule:.0f}h for the schedule) – random shocks such as rollovers and "
          f"canal disruptions dominate, and the honest output is a wide band. Once the vessel reaches port, the model is "
          f"<b>{imp_c2:.0%} better than carrying the current delay forward</b> ({c2.mae_gbm:.0f}h vs "
          f"{c2.mae_carry_forward:.0f}h), because it knows when that delay will be amplified at the berth or in customs. "
          f"After customs release it is {imp_c3:.0%} better ({c3.mae_gbm:.1f}h). The ridge model, given the same "
          f"interaction features, is a respectable second – evidence that most of the value lies in the feature design. "
          f"Conformal bands achieve close to the nominal 80% coverage in CV."),
        Table([[img("fig1_mae_by_checkpoint.png", 8.6), img("fig5_eta_evolution.png", 8.6)]],
              colWidths=[8.9 * cm, 8.9 * cm]),
        P("Figure 1 – Left: error vs checkpoint. Right: ETA updates for an unseen Singapore→Hamburg shipment "
          "that suffered a Suez diversion; the band narrows and converges on the actual delivery.", CAP),

        P("6. How delays propagate", H2),
        P(f"Table 2 measures the delay added <i>after</i> the vessel reaches port (final − arrival delay). At low "
          f"congestion a vessel arriving up to 24h late is largely <b>absorbed</b> (median {pr('low', 'late 0-24h'):+.0f}h; "
          f"{pr('low', 'late 0-24h', 'share_absorbed'):.0%} of shipments recover). At high congestion the same small "
          f"lateness adds <b>{pr('high', 'late 0-24h'):+.0f}h</b>, and a >24h-late vessel (missed berth window) adds "
          f"<b>{pr('high', 'late >24h'):+.0f}h</b> vs {pr('low', 'late >24h'):+.0f}h at low congestion – a "
          f"~{pr('high', 'late >24h') / pr('low', 'late >24h'):.0f}× amplification. Figure 2 (right) shows the "
          f"model learned exactly this non-linear interaction: a step at the 24h berth-window threshold whose height "
          f"grows with congestion. Figure 3 (right) shows where delay originates per lane: customs and origin "
          f"departure dominate on the trans-Pacific lanes, berth congestion is largest into Hamburg, and Suez "
          f"diversions add a large ocean-transit share on Nhava Sheva→Felixstowe."),
    ]
    prows = [["Congestion", "Arrival", "n", "Med. arrival delay", "Med. final delay", "Med. added downstream", "Absorbed"]]
    for _, r in prop.iterrows():
        prows.append([r.congestion, r.arrival, int(r.n), f"{r.median_arrival_delay_h:.0f}h", f"{r.median_final_delay_h:.0f}h",
                      f"{r.median_downstream_added_h:+.0f}h", f"{r.share_absorbed:.0%}"])
    story += [
        KeepTogether([table(prows, [2.8 * cm, 2.4 * cm, 0.8 * cm, 2.8 * cm, 2.6 * cm, 3.4 * cm, 1.6 * cm], font=7),
                      P("Table 2 – Downstream delay added after port arrival, by arrival lateness × congestion.", CAP)]),
        img("fig2_propagation.png", 16.5),
        P("Figure 2 – Left: observed arrival vs final delay coloured by congestion. Right: model partial dependence; "
          "the gap above the dashed 1:1 line is delay added downstream.", CAP),

        P("7. Explanations", H2),
        P("Each prediction is explained in two layers: (i) <b>delay already incurred</b>, decomposed exactly into "
          "milestone-to-milestone increments (missing events merge adjacent stages), and (ii) <b>expected further "
          "delay</b>, explained with TreeSHAP values aggregated into six plain-language cause groups. Example "
          "(unseen shipment, at port arrival, from <font face='Courier'>python predict.py</font>):"),
        KeepTogether(Preformatted(ex_text, MONO)),
        Spacer(1, 3),
        P(f"<b>Are the explanations right?</b> Because the simulator records the true cause of every hour of delay, I "
          f"checked, for the {att['n']} shipments with >24h still to come at port arrival, whether the top SHAP group "
          f"(port congestion / customs / inland) matches the stage that truly generated the most excess delay. "
          f"Agreement is <b>{att['top1_agreement']:.0%}</b> (top-2: {att['top2_agreement']:.0%}) vs {att['chance_top1']:.0%} "
          f"by chance. Figure 3 (left) shows how the dominant drivers shift over the lifecycle: customs and port congestion "
          f"matter most early on; after release only truck-appointment slack and the propagated delay remain."),
        Table([[img("fig4_drivers_by_checkpoint.png", 8.7), img("fig3_stage_contribution.png", 8.7)]],
              colWidths=[8.9 * cm, 8.9 * cm]),
        P("Figure 3 – Left: mean |SHAP| (hours) per cause group and checkpoint. Right: ground-truth origin of "
          "delay per lane (delayed shipments).", CAP),
    ]
    r0 = rob[(rob.missing_scale == 0) & (rob.checkpoint == "C3_customs_release")].mae_gbm.iloc[0]
    r3 = rob[(rob.missing_scale == 3) & (rob.checkpoint == "C3_customs_release")].mae_gbm.iloc[0]
    r3c2 = rob[(rob.missing_scale == 3) & (rob.checkpoint == "C2_port_arrival")].mae_gbm.iloc[0]
    r0c2 = rob[(rob.missing_scale == 0) & (rob.checkpoint == "C2_port_arrival")].mae_gbm.iloc[0]
    story += [
        P("8. Robustness, assumptions and limitations", H2),
        *bullets([
            f"<b>Missing events:</b> tripling the missing-event rate (up to 45% of berth/discharge events) moves MAE at "
            f"port arrival only from {r0c2:.0f}h to {r3c2:.0f}h and after customs from {r0:.1f}h to {r3:.1f}h, staying "
            f"well below the carry-forward baseline.",
            "<b>Assumptions:</b> schedules are fixed at booking; congestion is a weekly port index available from a "
            "port/AIS feed; customs works weekdays only; one truck appointment per shipment; forecasts at booking are "
            "noisy versions of the truth. All coefficients are stylised and documented in "
            "<font face='Courier'>src/config.py</font>.",
            "<b>Limitations:</b> the data is synthetic, so absolute errors are not real-world claims; the value shown is the "
            "method and its validation. With 300 records the uncertainty of each metric is visible in the CV spread "
            "(Fig. 1); the 60-shipment hold-out coverage after customs (" + f"{hoi.loc['C3_customs_release'].coverage:.0%}"
            + ") shows that bands are less reliable when very narrow.",
            "<b>Next steps with real data:</b> AIS vessel positions for live transit delay, port call/berth line-ups for "
            "congestion, customs-broker events; a sequence model across many vessels to capture network effects "
            "(one late vessel delaying the next ones at the same berth); and causal “what-if” estimates "
            "(e.g. value of expediting documents) on top of the explanations.",
        ]),
    ]

    doc = SimpleDocTemplate(str(ROOT / "report" / "Technical_Report.pdf"), pagesize=A4, leftMargin=1.6 * cm,
                            rightMargin=1.6 * cm, topMargin=1.3 * cm, bottomMargin=1.3 * cm,
                            title="Shipment ETA and Delay Propagation", author=author)
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    print("written", ROOT / "report" / "Technical_Report.pdf")


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.grey)
    canvas.drawRightString(A4[0] - 1.6 * cm, 0.8 * cm, f"{doc.page}")
    canvas.restoreState()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", default="Candidate")
    build(ap.parse_args().author)
