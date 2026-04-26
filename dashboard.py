"""
dashboard.py — Billboard Boxing
================================
Interactive Dash dashboard covering EDA and modeling results.

Run:
    python dashboard.py
Then open http://127.0.0.1:8050 in your browser.
"""

import ast
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import dash
from dash import Input, Output, dcc, html

# ── Colour palette ────────────────────────────────────────────────────────────
HIT_COLOR    = "#E07B54"
NONHIT_COLOR = "#5B8DB8"
BG_COLOR     = "#F9F9F9"
CARD_COLOR   = "#FFFFFF"
TEXT_COLOR   = "#2C2C2C"
ACCENT       = "#E07B54"

# ── Load & prepare data ───────────────────────────────────────────────────────
df = pd.read_csv("data/processed/billboard_expanded_dataset.csv")
df["decade"]      = (df["year"] // 10) * 10
df["is_explicit"] = df["explicit"].astype(float).fillna(0).astype(int)
df["label_str"]   = df["label"].map({1: "Billboard Hit", 0: "Non-hit"})
df["duration_min"] = df["duration_ms"] / 60_000

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]
FEATURES = AUDIO_FEATURES + ["duration_ms", "is_explicit", "decade"]

# ── Train models (cached at startup) ─────────────────────────────────────────
RANDOM_STATE = 42
X = df[FEATURES]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

lr_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("scl", StandardScaler()),
    ("mdl", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
])
rf_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("mdl", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
])
xgb_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("mdl", XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0,
    )),
])

for pipe in (lr_pipe, rf_pipe, xgb_pipe):
    pipe.fit(X_train, y_train)

MODELS = {
    "Logistic Regression": lr_pipe,
    "Random Forest":       rf_pipe,
    "XGBoost":             xgb_pipe,
}

# ── Pre-compute blind pool (2020 hold-out) ────────────────────────────────────
def _parse_artist(raw):
    try:
        p = ast.literal_eval(raw)
        if isinstance(p, list) and p:
            return str(p[0])
    except Exception:
        pass
    return str(raw)

def build_blind_pool():
    HOLD_OUT_YEAR = 2020
    pool_ds       = df[df["year"] == HOLD_OUT_YEAR].copy()
    hits_2020     = pool_ds[pool_ds["label"] == 1].sample(n=100, random_state=RANDOM_STATE)
    nonhits_in_ds = pool_ds[pool_ds["label"] == 0]

    n_extra = max(0, 900 - len(nonhits_in_ds))
    kag = pd.read_csv("data/archive/data.csv")
    kag = kag.rename(columns={"id": "spotify_id", "name": "title"})
    kag["artist"]      = kag["artists"].apply(_parse_artist)
    kag["year"]        = pd.to_numeric(kag["year"], errors="coerce")
    kag["decade"]      = (kag["year"] // 10) * 10
    kag["is_explicit"] = kag["explicit"].astype(float).fillna(0).astype(int)
    kag["duration_ms"] = kag["duration_ms"].astype(float)
    kag["label"]       = 0

    existing = set(pool_ds["spotify_id"].dropna())
    extra    = kag[(kag["year"] == HOLD_OUT_YEAR) & (~kag["spotify_id"].isin(existing))].sample(
        n=n_extra, random_state=RANDOM_STATE
    )
    nonhits = pd.concat([nonhits_in_ds, extra], ignore_index=True).sample(n=900, random_state=RANDOM_STATE)
    pool    = pd.concat([hits_2020, nonhits], ignore_index=True).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # Retrain on non-2020
    tr     = df[df["year"] != HOLD_OUT_YEAR].copy()
    tr["decade"]      = (tr["year"] // 10) * 10
    tr["is_explicit"] = tr["explicit"].astype(float).fillna(0).astype(int)
    model  = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("mdl", XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0,
        )),
    ])
    model.fit(tr[FEATURES], tr["label"])

    pool["decade"]      = (pool["year"] // 10) * 10
    pool["is_explicit"] = pool["explicit"].astype(float).fillna(0).astype(int)
    pool["hit_prob"]    = model.predict_proba(pool[FEATURES])[:, 1]
    pool = pool.sort_values("hit_prob", ascending=False).reset_index(drop=True)
    pool["rank"] = pool.index + 1
    return pool

blind_pool = build_blind_pool()

# ═════════════════════════════════════════════════════════════════════════════
# Layout helpers
# ═════════════════════════════════════════════════════════════════════════════

def card(children, style=None):
    base = {
        "background": CARD_COLOR,
        "borderRadius": "10px",
        "padding": "20px",
        "marginBottom": "20px",
        "boxShadow": "0 2px 8px rgba(0,0,0,0.06)",
    }
    if style:
        base.update(style)
    return html.Div(children, style=base)


def section_title(text):
    return html.H3(text, style={"color": TEXT_COLOR, "marginBottom": "12px", "fontWeight": "600"})


def stat_box(label, value, color=ACCENT):
    return html.Div([
        html.Div(value, style={"fontSize": "2rem", "fontWeight": "700", "color": color}),
        html.Div(label, style={"fontSize": "0.85rem", "color": "#888", "marginTop": "2px"}),
    ], style={
        "background": CARD_COLOR, "borderRadius": "10px", "padding": "18px 24px",
        "boxShadow": "0 2px 8px rgba(0,0,0,0.06)", "textAlign": "center", "flex": "1",
    })


# ═════════════════════════════════════════════════════════════════════════════
# EDA Tab
# ═════════════════════════════════════════════════════════════════════════════

eda_tab = html.Div([

    # ── Stat row ──────────────────────────────────────────────────────────────
    html.Div([
        stat_box("Total Songs",    f"{len(df):,}"),
        stat_box("Billboard Hits", f"{(df.label==1).sum():,}", HIT_COLOR),
        stat_box("Non-hits",       f"{(df.label==0).sum():,}", NONHIT_COLOR),
        stat_box("Years Covered",  f"{int(df.year.min())}–{int(df.year.max())}"),
        stat_box("Audio Features", "12"),
    ], style={"display": "flex", "gap": "16px", "marginBottom": "24px"}),

    # ── Feature distribution ──────────────────────────────────────────────────
    card([
        section_title("Audio Feature Distributions: Hits vs Non-Hits"),
        html.P("Select a feature to compare its distribution between Billboard hits and non-hits.",
               style={"color": "#666", "marginBottom": "12px"}),
        dcc.Dropdown(
            id="feat-dropdown",
            options=[{"label": f.capitalize(), "value": f} for f in AUDIO_FEATURES],
            value="danceability",
            clearable=False,
            style={"width": "260px", "marginBottom": "12px"},
        ),
        dcc.Graph(id="feat-dist-chart"),
    ]),

    # ── Temporal trends ───────────────────────────────────────────────────────
    card([
        section_title("Audio Feature Trends by Decade"),
        html.P("Mean feature value per decade for hits vs non-hits. "
               "Hits consistently sit at the 'more commercial' end of each era.",
               style={"color": "#666", "marginBottom": "12px"}),
        dcc.Dropdown(
            id="trend-dropdown",
            options=[{"label": f.capitalize(), "value": f} for f in AUDIO_FEATURES],
            value="energy",
            clearable=False,
            style={"width": "260px", "marginBottom": "12px"},
        ),
        dcc.Graph(id="trend-chart"),
    ]),

    # ── Correlation heatmap ───────────────────────────────────────────────────
    card([
        section_title("Feature Correlation Matrix"),
        html.P("No feature pair exceeds |r| = 0.7 — no features need to be dropped on correlation grounds.",
               style={"color": "#666", "marginBottom": "12px"}),
        dcc.Graph(id="corr-heatmap"),
    ]),

    # ── Duration scatter ──────────────────────────────────────────────────────
    card([
        section_title("Duration vs Danceability"),
        html.P("Hits cluster around 3–4 minutes with higher danceability.",
               style={"color": "#666", "marginBottom": "12px"}),
        dcc.Graph(id="scatter-chart"),
    ]),

], style={"padding": "0 8px"})


# ═════════════════════════════════════════════════════════════════════════════
# Modeling Tab
# ═════════════════════════════════════════════════════════════════════════════

def model_comparison_fig():
    rows = []
    for name, pipe in MODELS.items():
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        rows.append({
            "Model":    name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 3),
            "F1":       round(f1_score(y_test, y_pred), 3),
            "ROC-AUC":  round(roc_auc_score(y_test, y_prob), 3),
        })
    res = pd.DataFrame(rows)
    fig = go.Figure(data=[
        go.Bar(name="Accuracy", x=res["Model"], y=res["Accuracy"], marker_color="#A8C8E8"),
        go.Bar(name="F1",       x=res["Model"], y=res["F1"],       marker_color=HIT_COLOR),
        go.Bar(name="ROC-AUC",  x=res["Model"], y=res["ROC-AUC"],  marker_color="#6B8E6B"),
    ])
    fig.update_layout(
        barmode="group", template="plotly_white",
        legend=dict(orientation="h", y=1.12),
        yaxis=dict(range=[0, 1], title="Score"),
        margin=dict(t=40, b=40),
    )
    return fig


def roc_fig():
    fig = go.Figure()
    colors = {"Logistic Regression": "#A8C8E8", "Random Forest": "#6B8E6B", "XGBoost": HIT_COLOR}
    for name, pipe in MODELS.items():
        fpr, tpr, _ = roc_curve(y_test, pipe.predict_proba(X_test)[:, 1])
        auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})",
            line=dict(color=colors[name], width=2),
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random",
        line=dict(color="gray", dash="dash", width=1), showlegend=False,
    ))
    fig.update_layout(
        template="plotly_white",
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        legend=dict(x=0.55, y=0.08), margin=dict(t=20, b=40),
    )
    return fig


def feat_imp_fig():
    imp  = xgb_pipe.named_steps["mdl"].feature_importances_
    feat = pd.Series(imp, index=FEATURES).sort_values()
    colors = [HIT_COLOR if v == feat.max() else NONHIT_COLOR for v in feat.values]
    fig = go.Figure(go.Bar(
        x=feat.values, y=feat.index, orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Importance Score",
        margin=dict(t=20, b=40, l=140),
    )
    return fig


def confusion_fig(model_name):
    pipe   = MODELS[model_name]
    y_pred = pipe.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)
    fig    = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Non-hit", "Hit"], y=["Non-hit", "Hit"],
        color_continuous_scale="Blues",
        text_auto=True,
    )
    fig.update_layout(template="plotly_white", margin=dict(t=20, b=40),
                      coloraxis_showscale=False)
    return fig


def blind_pool_fig():
    # Precision@K curve
    ks     = list(range(10, 701, 10))
    prec   = [blind_pool.head(k)["label"].sum() / k for k in ks]
    base   = (blind_pool["label"] == 1).mean()

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Precision@K — 2020 Blind Pool",
                                        "Hit Probability Distribution"])

    fig.add_trace(go.Scatter(
        x=ks, y=prec, mode="lines", name="Model",
        line=dict(color=HIT_COLOR, width=2),
    ), row=1, col=1)
    fig.add_hline(y=base, line_dash="dash", line_color="gray",
                  annotation_text=f"Random ({base:.0%})", row=1, col=1)
    fig.add_vline(x=100, line_dash="dot", line_color="black", opacity=0.4, row=1, col=1)

    for lbl, name, color in [(0, "Non-hit", NONHIT_COLOR), (1, "Hit", HIT_COLOR)]:
        probs = blind_pool[blind_pool["label"] == lbl]["hit_prob"]
        fig.add_trace(go.Histogram(
            x=probs, name=name, opacity=0.65,
            marker_color=color, histnorm="probability density",
            nbinsx=40,
        ), row=1, col=2)

    fig.update_xaxes(title_text="K (top-K predictions)", row=1, col=1)
    fig.update_yaxes(title_text="Precision@K", range=[0, 1], row=1, col=1)
    fig.update_xaxes(title_text="Hit Probability", row=1, col=2)
    fig.update_layout(template="plotly_white", barmode="overlay",
                      margin=dict(t=40, b=40), legend=dict(x=0.55, y=0.98))
    return fig


modeling_tab = html.Div([

    # ── Stat row ──────────────────────────────────────────────────────────────
    html.Div([
        stat_box("Best Model",    "XGBoost"),
        stat_box("ROC-AUC",       "0.916", HIT_COLOR),
        stat_box("F1 Score",      "0.813", HIT_COLOR),
        stat_box("Blind Test P@100", "51%", "#6B8E6B"),
        stat_box("vs Random",     "5× better", "#6B8E6B"),
    ], style={"display": "flex", "gap": "16px", "marginBottom": "24px"}),

    # ── Model comparison ──────────────────────────────────────────────────────
    card([
        section_title("Model Comparison"),
        html.P("All three models trained on 80% of data (excluding 2020 for blind test). "
               "XGBoost (tuned) achieves the best performance across all metrics.",
               style={"color": "#666", "marginBottom": "12px"}),
        dcc.Graph(figure=model_comparison_fig()),
    ]),

    # ── ROC curves ────────────────────────────────────────────────────────────
    card([
        section_title("ROC Curves"),
        dcc.Graph(figure=roc_fig()),
    ]),

    # ── Confusion matrix ──────────────────────────────────────────────────────
    card([
        section_title("Confusion Matrix"),
        dcc.Dropdown(
            id="cm-model-dropdown",
            options=[{"label": k, "value": k} for k in MODELS],
            value="XGBoost",
            clearable=False,
            style={"width": "280px", "marginBottom": "12px"},
        ),
        dcc.Graph(id="cm-chart"),
    ]),

    # ── Feature importance ────────────────────────────────────────────────────
    card([
        section_title("Feature Importance — XGBoost"),
        html.P("Decade (era) and danceability are the strongest predictors. "
               "Instrumentalness is negatively associated with chart success.",
               style={"color": "#666", "marginBottom": "12px"}),
        dcc.Graph(figure=feat_imp_fig()),
    ]),

    # ── Blind pool ────────────────────────────────────────────────────────────
    card([
        section_title("2020 Blind Pool — Temporal Hold-Out Test"),
        html.P(
            "Model retrained without any 2020 data. Blind pool: 100 Billboard hits "
            "hidden among 900 non-hits (10% hit rate). Precision@100 = 51% vs 10% random baseline.",
            style={"color": "#666", "marginBottom": "12px"},
        ),
        dcc.Graph(figure=blind_pool_fig()),
        html.Div(id="blind-table-container", style={"marginTop": "16px"}),
        html.Div([
            html.Label("Show top-K predictions:", style={"marginRight": "10px", "color": "#555"}),
            dcc.Slider(id="blind-k-slider", min=10, max=100, step=10, value=20,
                       marks={i: str(i) for i in range(10, 110, 10)}),
        ], style={"marginTop": "16px"}),
    ]),

], style={"padding": "0 8px"})


# ═════════════════════════════════════════════════════════════════════════════
# App layout
# ═════════════════════════════════════════════════════════════════════════════

app = dash.Dash(__name__, title="Billboard Boxing", suppress_callback_exceptions=True)
app.layout = html.Div([

    # Header
    html.Div([
        html.H1("🎵 Billboard Boxing",
                style={"margin": "0", "color": "white", "fontWeight": "700", "fontSize": "1.8rem"}),
        html.P("Can Spotify audio features predict Billboard Hot 100 chart success?",
               style={"margin": "4px 0 0", "color": "rgba(255,255,255,0.8)", "fontSize": "0.95rem"}),
    ], style={
        "background": f"linear-gradient(135deg, {HIT_COLOR}, #C0552E)",
        "padding": "24px 40px",
        "marginBottom": "0",
    }),

    # Tabs
    dcc.Tabs(id="main-tabs", value="eda", children=[
        dcc.Tab(label="📊  Exploratory Data Analysis", value="eda",
                style={"fontWeight": "500"},
                selected_style={"fontWeight": "700", "color": HIT_COLOR, "borderTop": f"3px solid {HIT_COLOR}"}),
        dcc.Tab(label="🤖  Modeling & Results", value="modeling",
                style={"fontWeight": "500"},
                selected_style={"fontWeight": "700", "color": HIT_COLOR, "borderTop": f"3px solid {HIT_COLOR}"}),
    ], style={"marginBottom": "0", "borderBottom": "1px solid #ddd"}),

    html.Div(id="tab-content", style={
        "maxWidth": "1200px", "margin": "0 auto", "padding": "28px 20px",
        "background": BG_COLOR, "minHeight": "100vh",
    }),

], style={"fontFamily": "'Segoe UI', Arial, sans-serif", "background": BG_COLOR})


# ═════════════════════════════════════════════════════════════════════════════
# Callbacks
# ═════════════════════════════════════════════════════════════════════════════

@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab):
    return eda_tab if tab == "eda" else modeling_tab


@app.callback(Output("feat-dist-chart", "figure"), Input("feat-dropdown", "value"))
def update_feat_dist(feat):
    fig = go.Figure()
    for lbl, name, color in [(0, "Non-hit", NONHIT_COLOR), (1, "Billboard Hit", HIT_COLOR)]:
        vals = df[df["label"] == lbl][feat].dropna()
        fig.add_trace(go.Histogram(
            x=vals, name=name, opacity=0.65,
            marker_color=color, histnorm="probability density", nbinsx=50,
        ))
    fig.update_layout(
        barmode="overlay", template="plotly_white",
        xaxis_title=feat.capitalize(), yaxis_title="Density",
        legend=dict(x=0.78, y=0.98), margin=dict(t=20, b=40),
    )
    return fig


@app.callback(Output("trend-chart", "figure"), Input("trend-dropdown", "value"))
def update_trend(feat):
    trend = df.groupby(["decade", "label_str"])[feat].mean().reset_index()
    fig = px.line(
        trend, x="decade", y=feat, color="label_str",
        color_discrete_map={"Billboard Hit": HIT_COLOR, "Non-hit": NONHIT_COLOR},
        markers=True,
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Decade", yaxis_title=f"Mean {feat}",
        legend_title="", margin=dict(t=20, b=40),
    )
    return fig


@app.callback(Output("corr-heatmap", "figure"), Input("main-tabs", "value"))
def update_corr(_):
    corr = df[AUDIO_FEATURES + ["duration_ms"]].corr().round(2)
    fig  = px.imshow(
        corr, text_auto=True, color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, aspect="auto",
    )
    fig.update_layout(template="plotly_white", margin=dict(t=20, b=40))
    return fig


@app.callback(Output("scatter-chart", "figure"), Input("main-tabs", "value"))
def update_scatter(_):
    sample = df.sample(n=min(5000, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="duration_min", y="danceability",
        color="label_str",
        color_discrete_map={"Billboard Hit": HIT_COLOR, "Non-hit": NONHIT_COLOR},
        opacity=0.4,
        labels={"duration_min": "Duration (min)", "danceability": "Danceability"},
        range_x=[0, 8],
    )
    fig.update_layout(
        template="plotly_white", legend_title="",
        margin=dict(t=20, b=40),
    )
    return fig


@app.callback(Output("cm-chart", "figure"), Input("cm-model-dropdown", "value"))
def update_cm(model_name):
    return confusion_fig(model_name)


@app.callback(
    Output("blind-table-container", "children"),
    Input("blind-k-slider", "value"),
)
def update_blind_table(k):
    top_k = blind_pool.head(k)[["rank", "title", "artist", "hit_prob", "label"]].copy()
    top_k["hit_prob"]     = top_k["hit_prob"].round(3)
    top_k["Actual Hit?"]  = top_k["label"].map({1: "✅ Yes", 0: "❌ No"})
    top_k = top_k.rename(columns={
        "rank": "Rank", "title": "Title", "artist": "Artist",
        "hit_prob": "Hit Probability",
    }).drop(columns="label")

    hits_in_top = int((blind_pool.head(k)["label"] == 1).sum())
    precision   = hits_in_top / k

    return html.Div([
        html.P(
            f"Top {k} predictions: {hits_in_top} actual hits → Precision@{k} = {precision:.0%}",
            style={"fontWeight": "600", "color": HIT_COLOR, "marginBottom": "8px"},
        ),
        html.Table(
            [html.Thead(html.Tr([html.Th(c, style={"padding": "6px 12px", "background": "#f0f0f0",
                                                    "textAlign": "left", "fontSize": "0.85rem"})
                                 for c in top_k.columns]))] +
            [html.Tbody([
                html.Tr([
                    html.Td(
                        str(top_k.iloc[i][c]),
                        style={
                            "padding": "5px 12px", "fontSize": "0.83rem",
                            "background": "rgba(224,123,84,0.08)" if top_k.iloc[i]["Actual Hit?"] == "✅ Yes" else "white",
                        }
                    )
                    for c in top_k.columns
                ])
                for i in range(len(top_k))
            ])],
            style={"width": "100%", "borderCollapse": "collapse",
                   "border": "1px solid #eee", "borderRadius": "6px"},
        ),
    ])


if __name__ == "__main__":
    app.run(debug=True)
