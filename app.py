from flask import Flask, render_template, url_for
import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

app = Flask(__name__)

DATA_PATH = os.path.join("data", "AXA_Case1_Analysis_FINAL.xlsx")
AXA_BLUE = "#00008F"
AXA_RED = "#FF1721"
AXA_NAVY = "#07143F"
TEXT_DARK = "#111827"
MUTED = "#667085"
COLOR_SEQUENCE = ["#00008F", "#FF1721", "#2563EB", "#0F766E", "#7C3AED", "#F59E0B", "#475569", "#0EA5E9", "#DC2626", "#64748B"]


def idr_billion(value):
    try:
        return f"IDR {float(value) / 1e9:,.1f}B"
    except Exception:
        return "IDR 0.0B"


def pct(value):
    try:
        return f"{float(value):.1f}%"
    except Exception:
        return "0.0%"


def number(value):
    try:
        return f"{float(value):,.0f}"
    except Exception:
        return "0"


def clean_label(value):
    return str(value).replace("⭐", "").replace("💎", "").replace("⚠️", "").strip()


def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().upper().replace(" ", "_") for c in df.columns]
    return df


def ensure_required_columns(df, key_col="COB"):
    if df is None or df.empty:
        return pd.DataFrame()

    df = normalize_columns(df)
    rename_map = {
        "BRANCH": "BRANCH_",
        "CHANNEL": "CHANNEL_",
        "GROSS_CLAIM": "GROSS_CLAIMS",
        "TOTAL_GWP": "GWP",
        "TOTAL_GROSS_CLAIM": "GROSS_CLAIMS",
        "UW_MARGIN": "UNDERWRITING_MARGIN",
        "UW_MARGIN_PCT": "UNDERWRITING_MARGIN_PCT",
        "LOSS_RATIO": "LOSS_RATIO_PCT",
        "COMBINED_RATIO": "COMBINED_RATIO_PCT",
        "SCORE": "COMPOSITE_SCORE",
        "LABEL": "SEGMENT_LABEL",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if key_col not in df.columns and len(df.columns) > 0:
        df = df.rename(columns={df.columns[0]: key_col})

    numeric_cols = [
        "GWP", "RWP", "NWP", "GWC", "NWC", "SUM_INSURED", "POLICY_COUNT",
        "GROSS_CLAIMS", "NET_CLAIMS", "GROSS_PAID", "GROSS_OS", "CLAIM_COUNT",
        "CLAIM_COUNT_NONZERO", "LOSS_RATIO_PCT", "NET_LOSS_RATIO_PCT", "COMMISSION_RATIO_PCT",
        "UNDERWRITING_MARGIN", "UNDERWRITING_MARGIN_PCT", "COMBINED_RATIO_PCT",
        "GWP_CONTRIBUTION_PCT", "RI_CESSION_RATE_PCT", "RETENTION_RATE_PCT",
        "CLAIM_FREQUENCY", "AVG_CLAIM_SEVERITY", "COMPOSITE_SCORE"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "GWP" not in df.columns:
        df["GWP"] = 0
    if "GROSS_CLAIMS" not in df.columns:
        df["GROSS_CLAIMS"] = 0
    if "LOSS_RATIO_PCT" not in df.columns:
        df["LOSS_RATIO_PCT"] = np.where(df["GWP"] > 0, df["GROSS_CLAIMS"] / df["GWP"] * 100, 0)
    if "UNDERWRITING_MARGIN" not in df.columns:
        df["UNDERWRITING_MARGIN"] = df["GWP"] - df["GROSS_CLAIMS"]
    if "UNDERWRITING_MARGIN_PCT" not in df.columns:
        df["UNDERWRITING_MARGIN_PCT"] = np.where(df["GWP"] > 0, df["UNDERWRITING_MARGIN"] / df["GWP"] * 100, 0)
    if "COMBINED_RATIO_PCT" not in df.columns:
        df["COMBINED_RATIO_PCT"] = 100 - df["UNDERWRITING_MARGIN_PCT"]
    if "GWP_CONTRIBUTION_PCT" not in df.columns:
        total = df["GWP"].sum()
        df["GWP_CONTRIBUTION_PCT"] = np.where(total > 0, df["GWP"] / total * 100, 0)
    if "RETENTION_RATE_PCT" not in df.columns:
        df["RETENTION_RATE_PCT"] = 0
    if "COMPOSITE_SCORE" not in df.columns:
        lr_score = 100 - df["LOSS_RATIO_PCT"].clip(0, 100)
        margin_score = df["UNDERWRITING_MARGIN_PCT"].clip(0, 100)
        gwp_score = df["GWP_CONTRIBUTION_PCT"].rank(pct=True) * 100
        df["COMPOSITE_SCORE"] = 0.4 * lr_score + 0.4 * margin_score + 0.2 * gwp_score
    if "SEGMENT_LABEL" not in df.columns:
        df["SEGMENT_LABEL"] = np.select(
            [
                (df["GWP_CONTRIBUTION_PCT"] >= df["GWP_CONTRIBUTION_PCT"].median()) & (df["UNDERWRITING_MARGIN_PCT"] >= df["UNDERWRITING_MARGIN_PCT"].median()),
                (df["GWP_CONTRIBUTION_PCT"] < df["GWP_CONTRIBUTION_PCT"].median()) & (df["UNDERWRITING_MARGIN_PCT"] >= df["UNDERWRITING_MARGIN_PCT"].median()),
                (df["GWP_CONTRIBUTION_PCT"] >= df["GWP_CONTRIBUTION_PCT"].median()) & (df["UNDERWRITING_MARGIN_PCT"] < df["UNDERWRITING_MARGIN_PCT"].median()),
            ],
            ["STAR", "NICHE", "VOLUME TRAP"],
            default="REVIEW",
        )
    if "POLICY_COUNT" not in df.columns:
        df["POLICY_COUNT"] = 1
    if "CLAIM_COUNT_NONZERO" not in df.columns:
        df["CLAIM_COUNT_NONZERO"] = 0
    if "GROSS_PAID" not in df.columns:
        df["GROSS_PAID"] = df["GROSS_CLAIMS"] * 0.65
    if "GROSS_OS" not in df.columns:
        df["GROSS_OS"] = df["GROSS_CLAIMS"] * 0.35

    if "SEGMENT_LABEL" in df.columns:
        df["SEGMENT_LABEL_CLEAN"] = df["SEGMENT_LABEL"].apply(clean_label)

    return df


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("File data/AXA_Case1_Analysis_FINAL.xlsx belum ditemukan.")

    sheets = pd.read_excel(DATA_PATH, sheet_name=None, engine="openpyxl")
    sheets = {k.strip(): v for k, v in sheets.items()}

    seg_cob = ensure_required_columns(sheets.get("Segment_COB"), "COB")
    seg_branch = ensure_required_columns(sheets.get("Segment_Branch"), "BRANCH_")
    seg_channel = ensure_required_columns(sheets.get("Segment_Channel"), "CHANNEL_")
    seg_cob_branch = ensure_required_columns(sheets.get("COB_x_Branch"), "COB")
    seg_cob_channel = ensure_required_columns(sheets.get("COB_x_Channel"), "COB")

    return {
        "cob": seg_cob,
        "branch": seg_branch,
        "channel": seg_channel,
        "cob_branch": seg_cob_branch,
        "cob_channel": seg_cob_channel,
        "status": f""
    }


def plot_to_json(fig, height=430):
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial, sans-serif", color=TEXT_DARK),
        margin=dict(l=40, r=24, t=56, b=40),
        hoverlabel=dict(bgcolor="white", font_size=13, font_family="Inter, Arial, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="left", x=0),
    )
    return json.dumps(fig, cls=PlotlyJSONEncoder)


def make_bar(df, x, y, title, x_title, y_title, color_col=None, horizontal=True, text_suffix="", height=430):
    d = df.copy()
    orientation = "h" if horizontal else "v"
    kwargs = dict(title=title, color_discrete_sequence=COLOR_SEQUENCE)
    if color_col and color_col in d.columns:
        kwargs["color"] = color_col
    fig = px.bar(d, x=x, y=y, orientation=orientation, text=d[x].round(1).astype(str) + text_suffix if horizontal else d[y].round(1).astype(str) + text_suffix, **kwargs)
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title)
    return plot_to_json(fig, height)


def chart_priority(df, key="COB", title="Business priority ranking"):
    d = df.sort_values("COMPOSITE_SCORE", ascending=True).tail(10)
    return make_bar(d, "COMPOSITE_SCORE", key, title, "Priority score", key, "SEGMENT_LABEL_CLEAN", True, "", 460)


def chart_profit_leak(df):
    d = df.sort_values("UNDERWRITING_MARGIN", ascending=True)
    colors = [AXA_RED if v < 0 else AXA_BLUE for v in d["UNDERWRITING_MARGIN"]]
    fig = go.Figure(go.Bar(
        x=d["UNDERWRITING_MARGIN"] / 1e9,
        y=d["COB"],
        orientation="h",
        marker_color=colors,
        text=[f"{v/1e9:,.1f}B" for v in d["UNDERWRITING_MARGIN"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Margin: IDR %{x:.1f}B<extra></extra>"
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="#94A3B8", annotation_text="Break-even")
    fig.update_layout(title="Where profit is created or leaked", xaxis_title="Underwriting margin (IDR Billion)", yaxis_title="COB")
    return plot_to_json(fig, 470)


def chart_risk_level(df):
    d = df.sort_values("LOSS_RATIO_PCT", ascending=True)
    fig = px.bar(
        d,
        x="LOSS_RATIO_PCT",
        y="COB",
        orientation="h",
        color="LOSS_RATIO_PCT",
        color_continuous_scale=["#0F766E", "#F59E0B", "#DC2626"],
        text=d["LOSS_RATIO_PCT"].round(1).astype(str) + "%",
        title="Risk level by segment"
    )
    fig.add_vline(x=60, line_dash="dash", line_color="#0F766E", annotation_text="Healthy")
    fig.add_vline(x=80, line_dash="dash", line_color="#DC2626", annotation_text="Needs attention")
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(xaxis_title="Claims as % of premium", yaxis_title="COB", coloraxis_showscale=False)
    return plot_to_json(fig, 470)


def chart_claim_burden(df):
    d = df.sort_values("GROSS_CLAIMS", ascending=False).head(10)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["COB"], y=d["GROSS_PAID"] / 1e9, name="Already paid", marker_color=AXA_RED))
    fig.add_trace(go.Bar(x=d["COB"], y=d["GROSS_OS"] / 1e9, name="Still reserved", marker_color="#F59E0B"))
    fig.update_layout(title="Claim burden by segment", barmode="stack", xaxis_title="COB", yaxis_title="Claims (IDR Billion)", legend_title_text="")
    return plot_to_json(fig, 450)


def chart_claim_frequency(df):
    d = df.sort_values("CLAIM_FREQUENCY", ascending=True).tail(10)
    fig = px.bar(
        d, x="CLAIM_FREQUENCY", y="COB", orientation="h", color="CLAIM_FREQUENCY",
        color_continuous_scale=["#DBEAFE", "#2563EB", "#00008F"],
        text=(d["CLAIM_FREQUENCY"] * 100).round(1).astype(str) + "%",
        title="Segments with higher claim frequency"
    )
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(xaxis_title="Claim frequency", yaxis_title="COB", coloraxis_showscale=False)
    return plot_to_json(fig, 450)


def chart_mix(df, category_col, title):
    d = df.copy()
    if d.empty or category_col not in d.columns:
        return None
    p = d.pivot_table(index="COB", columns=category_col, values="GWP", aggfunc="sum", fill_value=0)
    p = p.div(p.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
    p = p.reset_index()
    fig = go.Figure()
    for i, col in enumerate([c for c in p.columns if c != "COB"]):
        fig.add_trace(go.Bar(y=p["COB"], x=p[col], name=str(col), orientation="h", marker_color=COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)]))
    fig.update_layout(title=title, barmode="stack", xaxis_title="Share of GWP (%)", yaxis_title="COB", legend_title_text="")
    return plot_to_json(fig, 500)


def chart_top_combinations(df, key_col, title, metric="COMPOSITE_SCORE"):
    if df.empty or key_col not in df.columns:
        return None
    d = df.copy()
    d["COMBINATION"] = d["COB"].astype(str) + " · " + d[key_col].astype(str)
    d = d.sort_values(metric, ascending=True).tail(12)
    label = "Priority score" if metric == "COMPOSITE_SCORE" else "GWP (IDR Billion)"
    values = d[metric] if metric == "COMPOSITE_SCORE" else d[metric] / 1e9
    fig = go.Figure(go.Bar(
        x=values,
        y=d["COMBINATION"],
        orientation="h",
        marker_color=AXA_BLUE,
        text=[f"{v:.1f}" for v in values],
        textposition="outside"
    ))
    fig.update_layout(title=title, xaxis_title=label, yaxis_title="Combination")
    return plot_to_json(fig, 520)


def chart_branch_margin(df):
    if df.empty:
        return None
    d = df.sort_values("UNDERWRITING_MARGIN_PCT", ascending=True)
    colors = [AXA_RED if v < 0 else AXA_BLUE for v in d["UNDERWRITING_MARGIN_PCT"]]
    fig = go.Figure(go.Bar(x=d["UNDERWRITING_MARGIN_PCT"], y=d["BRANCH_"], orientation="h", marker_color=colors, text=d["UNDERWRITING_MARGIN_PCT"].round(1).astype(str)+"%", textposition="outside"))
    fig.add_vline(x=0, line_dash="dash", line_color="#94A3B8")
    fig.update_layout(title="Branch margin health", xaxis_title="Underwriting margin (%)", yaxis_title="Branch")
    return plot_to_json(fig, 460)


def get_common_context(active="overview"):
    data = load_data()
    cob = data["cob"]
    branch = data["branch"]
    channel = data["channel"]
    cb = data["cob_branch"]
    cc = data["cob_channel"]

    total_gwp = cob["GWP"].sum()
    total_claims = cob["GROSS_CLAIMS"].sum()
    total_margin = cob["UNDERWRITING_MARGIN"].sum()
    portfolio_lr = total_claims / total_gwp * 100 if total_gwp else 0
    portfolio_margin_pct = total_margin / total_gwp * 100 if total_gwp else 0

    top_cob = cob.sort_values("COMPOSITE_SCORE", ascending=False).iloc[0]
    top_branch = branch.sort_values("COMPOSITE_SCORE", ascending=False).iloc[0] if not branch.empty else None
    top_channel = channel.sort_values("COMPOSITE_SCORE", ascending=False).iloc[0] if not channel.empty else None

    problem = cob[cob["UNDERWRITING_MARGIN"] < 0].sort_values("UNDERWRITING_MARGIN").head(1)
    problem_cob = problem.iloc[0] if not problem.empty else None

    profitable_count = int((cob["UNDERWRITING_MARGIN"] > 0).sum())
    attention_count = int((cob["LOSS_RATIO_PCT"] >= 80).sum())

    kpis = {
        "total_gwp": idr_billion(total_gwp),
        "total_claims": idr_billion(total_claims),
        "total_margin": idr_billion(total_margin),
        "portfolio_lr": pct(portfolio_lr),
        "portfolio_margin_pct": pct(portfolio_margin_pct),
        "total_policy": number(cob["POLICY_COUNT"].sum()),
        "top_segment": top_cob["COB"],
        "top_score": f"{top_cob['COMPOSITE_SCORE']:.1f}",
        "profitable_count": profitable_count,
        "attention_count": attention_count,
    }

    snapshot = {
        "top_cob": {
            "name": top_cob["COB"],
            "label": clean_label(top_cob["SEGMENT_LABEL"]),
            "gwp_share": pct(top_cob["GWP_CONTRIBUTION_PCT"]),
            "margin_pct": pct(top_cob["UNDERWRITING_MARGIN_PCT"]),
            "loss_ratio": pct(top_cob["LOSS_RATIO_PCT"]),
            "score": f"{top_cob['COMPOSITE_SCORE']:.1f}",
        },
        "top_branch": None if top_branch is None else {
            "name": top_branch["BRANCH_"],
            "label": clean_label(top_branch["SEGMENT_LABEL"]),
            "margin_pct": pct(top_branch["UNDERWRITING_MARGIN_PCT"]),
            "loss_ratio": pct(top_branch["LOSS_RATIO_PCT"]),
            "score": f"{top_branch['COMPOSITE_SCORE']:.1f}",
        },
        "top_channel": None if top_channel is None else {
            "name": top_channel["CHANNEL_"],
            "label": clean_label(top_channel["SEGMENT_LABEL"]),
            "margin_pct": pct(top_channel["UNDERWRITING_MARGIN_PCT"]),
            "loss_ratio": pct(top_channel["LOSS_RATIO_PCT"]),
            "score": f"{top_channel['COMPOSITE_SCORE']:.1f}",
        },
        "problem_cob": None if problem_cob is None else {
            "name": problem_cob["COB"],
            "margin": idr_billion(problem_cob["UNDERWRITING_MARGIN"]),
            "loss_ratio": pct(problem_cob["LOSS_RATIO_PCT"]),
            "label": clean_label(problem_cob["SEGMENT_LABEL"]),
        }
    }

    top_segments = cob.sort_values("COMPOSITE_SCORE", ascending=False).head(3).to_dict("records")
    for row in top_segments:
        row["SEGMENT_LABEL_CLEAN"] = clean_label(row.get("SEGMENT_LABEL", ""))

    return {
        "active": active,
        "data_status": data["status"],
        "kpis": kpis,
        "snapshot": snapshot,
        "top_segments": top_segments,
        "data": data,
    }


def interpretation(title, body, action=None):
    return {"title": title, "body": body, "action": action}


@app.route("/")
def overview():
    ctx = get_common_context("overview")
    s = ctx["snapshot"]
    cob = ctx["data"]["cob"]
    
    # 1. DEFINISIKAN NILAI DEFAULT
    pie_chart_json = None
    bar_chart_json = None
    
    # 2. LOGIKA PEMBUATAN CHART
    if not cob.empty:
        # Pie Chart (sekarang menjadi Bar Chart sesuai permintaanmu)
        fig_pie = px.bar(
            cob.sort_values('GWP_CONTRIBUTION_PCT', ascending=True), 
            x='GWP_CONTRIBUTION_PCT', y='COB', orientation='h',
            color_discrete_sequence=['#000080']
        )
        pie_chart_json = plot_to_json(fig_pie, height=380)
        
        # Bar Chart (Clustered)
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=cob['COB'], y=cob['GWP']/1e9, name='Gross Premium', marker_color='#000080'))
        fig_bar.add_trace(go.Bar(x=cob['COB'], y=cob['GROSS_CLAIMS']/1e9, name='Gross Claims', marker_color='#FF1721'))
        bar_chart_json = plot_to_json(fig_bar, height=380)

    # 3. UPDATE CONTEXT (Sekarang variabel sudah pasti ada)
    ctx.update({
        "page_title": "Executive Overview",
        "pie_chart": pie_chart_json,
        "bar_chart": bar_chart_json,
        # ... sisa kodinganmu
    })
    return render_template("overview.html", **ctx)    
    # --- 2. BUAT CHART: GWP vs Gross Claims (Bar Chart) ---
    # --- 2. REVISI MENJADI CLUSTERED BAR CHART ---
    fig_bar = go.Figure()

    # Menambahkan trace untuk GWP
    fig_bar.add_trace(go.Bar(
        x=cob['COB'], y=cob['GWP']/1e9, 
        name='Gross Premium', marker_color='#000080' # Navy AXA
    ))

    # Menambahkan trace untuk Gross Claims
    fig_bar.add_trace(go.Bar(
        x=cob['COB'], y=cob['GROSS_CLAIMS']/1e9, 
        name='Gross Claims', marker_color='#FF1721' # Merah AXA
    ))

    fig_bar.update_layout(
        barmode='group', # Ini yang membuat menjadi Clustered
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#e0e0e0'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    ctx.update({
        "page_title": "Executive Overview",
        "page_subtitle": "Ringkasan performa portofolio secara keseluruhan untuk menentukan kesehatan bisnis AXA sebelum membedah prioritas segmen.",
        "pie_chart": pie_chart_json,
        "bar_chart": bar_chart_json,
        "insights": [
            interpretation(
                "Distribusi Skala Bisnis (GWP)", 
                f"{s['top_cob']['name']} mendominasi portofolio dengan kontribusi {s['top_cob']['gwp_share']}. Portofolio kita sangat terpusat pada beberapa COB saja.", 
                "Pastikan kontribusi volume yang besar ini benar-benar sejalan dengan margin untung yang sehat."
            ),
            interpretation(
                "Disparitas Premi vs Beban Klaim", 
                f"Secara agregat margin kita positif {ctx['kpis']['total_margin']}. Namun di grafik terlihat batang merah (Klaim) nyaris menyamai biru (Premi) di beberapa segmen.", 
                "Volume besar bukan jaminan untung. Segmen dengan gap sempit antara premi dan klaim perlu diawasi ketat."
            ),
            interpretation(
                "Transisi: Area Perhatian (The Hook)", 
                f"Dari total portofolio, terdapat {ctx['kpis']['attention_count']} segmen yang Loss Ratio-nya masuk zona kritis (≥ 80%).", 
                "Buka menu 'Segment Portfolio' untuk membedah mana yang Bintang (STAR) dan mana Jebakan (VOLUME TRAP)."
            ),
        ]
    })
    return render_template("overview.html", **ctx)

@app.route("/segments")
def segments():
    ctx = get_common_context("segments")
    cob = ctx["data"]["cob"]
    top = ctx["snapshot"]["top_cob"]
    ctx.update({
        "page_title": "Segment Portfolio",
        "page_subtitle": "Menu ini menjawab segmen mana yang paling layak menjadi prioritas bisnis.",
        "priority_chart": chart_priority(cob, "COB", "Which segments should become business priorities?"),
        "interpretation": interpretation("Cara baca untuk client", f"Segmen dengan skor tertinggi bukan sekadar yang preminya besar, tetapi yang kombinasi premi, margin, dan risiko klaimnya paling sehat. Saat ini {top['name']} menjadi prioritas utama.", "Gunakan segmen STAR sebagai kandidat scale-up dan segmen REVIEW/VOLUME TRAP sebagai kandidat perbaikan."),
        "table_rows": make_segment_table(cob),
    })
    return render_template("segments.html", **ctx)


def make_segment_table(df):
    cols = ["COB", "GWP", "GROSS_CLAIMS", "LOSS_RATIO_PCT", "UNDERWRITING_MARGIN", "UNDERWRITING_MARGIN_PCT", "GWP_CONTRIBUTION_PCT", "COMPOSITE_SCORE", "SEGMENT_LABEL"]
    t = df[cols].copy().sort_values("COMPOSITE_SCORE", ascending=False)
    t["GWP"] = t["GWP"].apply(idr_billion)
    t["GROSS_CLAIMS"] = t["GROSS_CLAIMS"].apply(idr_billion)
    t["UNDERWRITING_MARGIN"] = t["UNDERWRITING_MARGIN"].apply(idr_billion)
    for c in ["LOSS_RATIO_PCT", "UNDERWRITING_MARGIN_PCT", "GWP_CONTRIBUTION_PCT"]:
        t[c] = t[c].apply(pct)
    t["COMPOSITE_SCORE"] = t["COMPOSITE_SCORE"].apply(lambda x: f"{x:.1f}")
    t["SEGMENT_LABEL"] = t["SEGMENT_LABEL"].apply(clean_label)
    return t.to_dict("records")


@app.route("/performance")
def performance():
    ctx = get_common_context("performance")
    cob = ctx["data"]["cob"]
    worst = cob.sort_values("UNDERWRITING_MARGIN").iloc[0]
    best = cob.sort_values("UNDERWRITING_MARGIN", ascending=False).iloc[0]
    ctx.update({
        "page_title": "Profitability & Risk",
        "page_subtitle": "Melihat sumber profit dan sumber kebocoran biaya klaim dalam bahasa bisnis.",
        "profit_chart": chart_profit_leak(cob),
        "risk_chart": chart_risk_level(cob),
        "profit_interpretation": interpretation("Inti grafik profit", f"{best['COB']} adalah kontributor margin terbesar, sedangkan {worst['COB']} menjadi sumber leakage terbesar dengan margin {idr_billion(worst['UNDERWRITING_MARGIN'])}.", "Jaga segmen yang menghasilkan margin dan lakukan pricing/underwriting review pada segmen yang bocor."),
        "risk_interpretation": interpretation("Inti grafik risiko", "Loss ratio menunjukkan seberapa besar premi yang habis untuk klaim. Semakin tinggi, semakin besar tekanan terhadap profitabilitas.", "Segmen di atas 80% perlu masuk daftar review karena ruang margin menjadi sempit."),
    })
    return render_template("performance.html", **ctx)


@app.route("/claims")
def claims():
    ctx = get_common_context("claims")
    cob = ctx["data"]["cob"]
    top_claim = cob.sort_values("GROSS_CLAIMS", ascending=False).iloc[0]
    freq = cob.sort_values("CLAIM_FREQUENCY", ascending=False).iloc[0]
    ctx.update({
        "page_title": "Claim Experience",
        "page_subtitle": "Membaca pola klaim untuk melihat segmen mana yang paling membebani portfolio.",
        "claim_chart": chart_claim_burden(cob),
        "frequency_chart": chart_claim_frequency(cob),
        "claim_interpretation": interpretation("Beban klaim terbesar", f"{top_claim['COB']} memiliki total klaim terbesar. Ini penting karena segmen dengan klaim besar bisa tetap berbahaya walaupun volume preminya juga besar.", "Pisahkan apakah klaim tinggi disebabkan frekuensi tinggi, severity tinggi, atau reserve outstanding yang besar."),
        "freq_interpretation": interpretation("Pola klaim berulang", f"{freq['COB']} memiliki frekuensi klaim paling tinggi, sehingga perlu dicek apakah banyak polis kecil yang sering claim atau ada masalah risk selection.", "Gunakan insight ini untuk memperbaiki underwriting rule dan claim control."),
    })
    return render_template("claims.html", **ctx)


@app.route("/branch-channel")
def branch_channel():
    ctx = get_common_context("branch-channel")
    branch = ctx["data"]["branch"]
    channel = ctx["data"]["channel"]
    topb = ctx["snapshot"]["top_branch"]
    topc = ctx["snapshot"]["top_channel"]
    ctx.update({
        "page_title": "Branch & Channel",
        "page_subtitle": "Melihat jalur distribusi dan cabang mana yang paling sehat secara bisnis.",
        "branch_chart": chart_priority(branch, "BRANCH_", "Which branches are strongest?"),
        "channel_chart": chart_priority(channel, "CHANNEL_", "Which channels are strongest?"),
        "branch_margin_chart": chart_branch_margin(branch),
        "branch_interpretation": interpretation("Branch terbaik", f"{topb['name']} menjadi branch paling kuat dengan margin {topb['margin_pct']} dan loss ratio {topb['loss_ratio']}." if topb else "Data branch belum tersedia.", "Branch dengan performa baik dapat dijadikan benchmark proses underwriting dan distribusi."),
        "channel_interpretation": interpretation("Channel terbaik", f"{topc['name']} menjadi channel paling kuat dengan margin {topc['margin_pct']} dan loss ratio {topc['loss_ratio']}." if topc else "Data channel belum tersedia.", "Channel terbaik dapat diprioritaskan untuk akuisisi bisnis baru selama kualitas risiko tetap dijaga."),
    })
    return render_template("branch_channel.html", **ctx)


@app.route("/cross-analysis")
def cross_analysis():
    ctx = get_common_context("cross-analysis")
    cb = ctx["data"]["cob_branch"]
    cc = ctx["data"]["cob_channel"]
    best_cb = cb.sort_values("COMPOSITE_SCORE", ascending=False).iloc[0] if not cb.empty else None
    best_cc = cc.sort_values("COMPOSITE_SCORE", ascending=False).iloc[0] if not cc.empty else None
    ctx.update({
        "page_title": "Opportunity Map",
        "page_subtitle": "Bukan hanya segmen mana yang bagus, tapi kombinasi segmen-cabang/channel mana yang paling layak diprioritaskan.",
        "cob_branch_chart": chart_top_combinations(cb, "BRANCH_", "Top COB x Branch opportunities"),
        "cob_channel_chart": chart_top_combinations(cc, "CHANNEL_", "Top COB x Channel opportunities"),
        "mix_chart": chart_mix(cc, "CHANNEL_", "How each COB is distributed by channel"),
        "cross_interpretation": interpretation("Kombinasi paling menarik", f"Kombinasi terbaik dari sisi branch adalah {best_cb['COB']} · {best_cb['BRANCH_']}, sedangkan dari sisi channel adalah {best_cc['COB']} · {best_cc['CHANNEL_']}." if best_cb is not None and best_cc is not None else "Data kombinasi belum tersedia.", "Gunakan kombinasi ini sebagai shortlist area ekspansi, bukan langsung sebagai keputusan final."),
    })
    return render_template("cross_analysis.html", **ctx)


@app.route("/strategy")
def strategy():
    ctx = get_common_context("strategy")
    cob = ctx["data"]["cob"]
    ctx.update({
        "page_title": "Business Recommendation",
        "page_subtitle": "Rekomendasi tindakan agar hasil analisis bisa langsung diterjemahkan menjadi keputusan bisnis.",
        "strategy_cards": build_strategy_cards(cob),
    })
    return render_template("strategy.html", **ctx)


def build_strategy_cards(df):
    rows = []
    for _, row in df.sort_values("COMPOSITE_SCORE", ascending=False).iterrows():
        label = clean_label(row.get("SEGMENT_LABEL", "REVIEW")).upper()
        if "STAR" in label:
            strategy = "Scale & Protect"
            priority = "HIGH"
            business_read = "Segmen ini sudah sehat dan layak menjadi engine pertumbuhan."
            action = "Perbesar volume secara selektif, pertahankan pricing discipline, dan monitor klaim agar margin tetap kuat."
        elif "NICHE" in label:
            strategy = "Selective Growth"
            priority = "MEDIUM"
            business_read = "Segmen ini profitable tetapi kontribusi bisnisnya masih terbatas."
            action = "Cari sub-segmen serupa, tambah distribusi secara bertahap, dan validasi apakah profit tetap stabil saat volume naik."
        elif "VOLUME" in label:
            strategy = "Fix Before Scale"
            priority = "HIGH"
            business_read = "Volume besar belum tentu sehat; ada indikasi profit bocor."
            action = "Review pricing, underwriting rule, klaim besar, dan channel asal bisnis sebelum ekspansi."
        else:
            strategy = "Portfolio Review"
            priority = "MEDIUM"
            business_read = "Segmen ini perlu dilihat lebih detail sebelum menjadi prioritas."
            action = "Evaluasi risiko, klaim, komisi, dan potensi pertumbuhan. Pertahankan hanya bagian yang masih ekonomis."
        rows.append({
            "cob": row["COB"],
            "label": clean_label(row["SEGMENT_LABEL"]),
            "strategy": strategy,
            "priority": priority,
            "business_read": business_read,
            "action": action,
            "score": f"{row['COMPOSITE_SCORE']:.1f}",
            "margin": pct(row["UNDERWRITING_MARGIN_PCT"]),
            "loss_ratio": pct(row["LOSS_RATIO_PCT"]),
        })
    return rows


if __name__ == "__main__":
    app.run(debug=True)
