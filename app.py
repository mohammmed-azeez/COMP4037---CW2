import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template

app = Flask(__name__)

df = pd.read_csv('static/nhs_merged.csv')

df['diagnosis_desc'] = df['diagnosis_desc'].str.strip()

df['short_desc'] = df['diagnosis_desc'].apply(
    lambda x: x[:35] + '…' if len(str(x)) > 35 else x
)

def assign_super_group(code):
    code = str(code).strip()
    if not code:
        return 'Other Conditions'
    c = code[0].upper()
    if c in ('A', 'B'):
        return 'Infectious & Parasitic'
    if c in ('C', 'D'):
        return 'Cancer & Blood'
    if c in ('E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N'):
        return 'Organ System Diseases'
    if c in ('S', 'T', 'V', 'W', 'X', 'Y'):
        return 'Injuries & External Causes'
    return 'Other Conditions'

df['super_group'] = df['diagnosis_code'].apply(assign_super_group)

df_2023 = df[df['year'] == 2023].copy()

df_new = df[df['year'].between(2012, 2023)].copy()
cats_2018 = set(df[df['year'] == 2018]['diagnosis_desc'])
cats_2023 = set(df[df['year'] == 2023]['diagnosis_desc'])
cats_both  = cats_2018 & cats_2023
df_trend = df_new[df_new['diagnosis_desc'].isin(cats_both)].copy()
df_trend = df_trend[df_trend['pct_change_admissions'].notna()].copy()

df_baseline = df[df['year'] == 2018].copy()

DARK = dict(
    plot_bgcolor='#0d1117',
    paper_bgcolor='#0d1117',
    font=dict(color='white', size=11),
)

_years_range = list(range(2018, 2024))

_top40 = df_2023.nlargest(40, 'admissions')['diagnosis_desc'].tolist()
_chart_cats = [c for c in _top40 if c in cats_both]

_categories_data = []
for _cat in _chart_cats:
    _cat_df = (df[(df['diagnosis_desc'] == _cat) &
                  (df['year'].isin(_years_range))]
               .set_index('year'))

    _row23 = df[(df['diagnosis_desc'] == _cat) & (df['year'] == 2023)]
    if _row23.empty:
        continue
    _row23 = _row23.iloc[0]

    _years_dict = {}
    for _yr in _years_range:
        if _yr in _cat_df.index:
            _r = _cat_df.loc[_yr]
            _years_dict[str(_yr)] = {
                'pct_change':   round(float(_r['pct_change_admissions']), 1)
                                if pd.notna(_r['pct_change_admissions']) else None,
                'pct_waiting':  round(float(_r['pct_change_waiting']), 1)
                                if pd.notna(_r['pct_change_waiting']) else 0,
                'admissions':   int(_r['admissions'])
                                if pd.notna(_r['admissions']) else 0,
                'waiting_list': int(_r['waiting_list'])
                                if pd.notna(_r['waiting_list']) else 0,
                'backlog_trap': bool(_r['backlog_trap']),
            }
        else:
            _years_dict[str(_yr)] = None

    _fs = (df[df['diagnosis_desc'] == _cat]
           [['year', 'admissions', 'pct_change_admissions']]
           .sort_values('year'))
    _full_series = []
    for _, _sr in _fs.iterrows():
        if pd.notna(_sr['admissions']) and _sr['admissions'] > 0:
            _full_series.append({
                'year':       int(_sr['year']),
                'admissions': int(_sr['admissions']),
                'pct_change': round(float(_sr['pct_change_admissions']), 1)
                              if pd.notna(_sr['pct_change_admissions']) else None,
            })

    _recent = [
        _years_dict[str(_yr)]['pct_change']
        for _yr in [2022, 2023]
        if _years_dict.get(str(_yr)) and
           _years_dict[str(_yr)].get('pct_change') is not None
    ]
    _recovery_score = round(sum(_recent) / len(_recent), 1) if _recent else 0.0

    _categories_data.append({
        'name':            _cat,
        'short_name':      (_cat[:32] + '\u2026') if len(_cat) > 32 else _cat,
        'super_group':     str(_row23['super_group']),
        'backlog_trap':    bool(_row23['backlog_trap']),
        'admissions_2023': int(_row23['admissions'])
                           if pd.notna(_row23['admissions']) else 0,
        'mean_age':        float(_row23['mean_age'])
                           if pd.notna(_row23['mean_age']) else 0.0,
        'years':           _years_dict,
        'full_series':     _full_series,
        'recovery_score':  _recovery_score,
    })

_categories_data.sort(key=lambda x: x['recovery_score'])

_all_waiting = [
    abs(_yd['pct_waiting'])
    for _c in _categories_data
    for _yd in _c['years'].values()
    if _yd and _yd.get('pct_waiting')
]
_global_max_waiting = float(np.percentile(_all_waiting, 90)) if _all_waiting else 100.0

chart1_data = {
    'categories':         _categories_data,
    'years':              [str(_y) for _y in _years_range],
    'year_labels':        [f"{_y}-{str(_y + 1)[2:]}" for _y in _years_range],
    'global_max_waiting': round(_global_max_waiting, 1),
    'super_groups':       sorted({_c['super_group'] for _c in _categories_data}),
}

chart1_data_json = json.dumps(chart1_data, ensure_ascii=False)

pc_data = (
    df_2023[df_2023['pct_change_admissions'].notna()]
    .nlargest(50, 'admissions')
    .copy()
)

metrics = [
    'pct_change_admissions',
    'pct_change_waiting',
    'emergency',
    'mean_los_days',
    'mean_age',
]
labels = [
    'Admissions % vs Baseline',
    'Waiting List % vs Baseline',
    'Emergency Admissions',
    'Mean Length of Stay (days)',
    'Mean Age of Patients',
]

scaler = MinMaxScaler()
pc_scaled = scaler.fit_transform(pc_data[metrics].fillna(0))

dimensions = [
    dict(label=lbl, values=pc_scaled[:, i], range=[0, 1])
    for i, lbl in enumerate(labels)
]

fig2 = go.Figure(go.Parcoords(
    line=dict(
        color=pc_data['pct_change_admissions'].values,
        colorscale='RdYlGn',
        cmin=-80,
        cmax=80,
        showscale=True,
        colorbar=dict(
            title=dict(text='Admissions % Change', side='right'),
        ),
    ),
    dimensions=dimensions,
))

fig2.update_layout(
    **DARK,
    title=dict(
        text=(
            'Parallel Coordinates: Multi-Metric Profile of NHS Diagnosis Categories (2023-24)'
            '<br><sub>Each line = one diagnosis category | '
            'Color = admissions recovery (red=not recovered, green=recovered)</sub>'
        ),
        font=dict(size=14),
    ),
    height=650,
)

parcoords_div = fig2.to_html(full_html=False, include_plotlyjs=False)

tm_data = df_2023[
    df_2023['admissions'].notna() &
    (df_2023['admissions'] > 0)
].copy()

fig3 = px.treemap(
    data_frame=tm_data,
    path=[px.Constant('NHS Admissions 2023-24'), 'super_group', 'short_desc'],
    values='admissions',
    color='pct_change_admissions',
    color_continuous_scale='RdYlGn',
    range_color=[-80, 80],
    color_continuous_midpoint=0,
    hover_data={
        'admissions': True,
        'pct_change_admissions': ':.1f',
        'diagnosis_desc': True,
    },
)

fig3.update_layout(
    **DARK,
    title=dict(
        text=(
            'Treemap: NHS Hospital Admissions Volume and Post-Pandemic Recovery by Category (2023-24)'
            '<br><sub>Rectangle size = admissions volume | '
            'Color = % change from 2018-19 baseline (red=below, green=recovered)</sub>'
        ),
        font=dict(size=14),
    ),
    height=750,
    coloraxis_colorbar=dict(
        title='% Change vs 2018-19',
        tickformat='.0f',
        ticksuffix='%',
    ),
)

treemap_div = fig3.to_html(full_html=False, include_plotlyjs=False)

top8 = df_2023.nlargest(8, 'admissions').copy()
top8['emergency_rate'] = np.where(
    top8['admissions'] > 0,
    top8['emergency'] / top8['admissions'] * 100,
    0,
)

radar_metrics = ['admissions', 'emergency_rate', 'waiting_list',
                 'mean_los_days', 'mean_age']
radar_labels  = ['Admissions Volume', 'Emergency Rate %',
                 'Waiting List Size', 'Avg Length of Stay',
                 'Patient Age (Mean)']

radar_scaled = MinMaxScaler().fit_transform(
    top8[radar_metrics].fillna(0)
) * 100

colors = px.colors.qualitative.Plotly

fig4 = go.Figure()
for i, (_, row) in enumerate(top8.iterrows()):
    name = str(row['short_desc'])[:25]
    r_vals = radar_scaled[i].tolist()
    r_vals_closed   = r_vals + [r_vals[0]]
    theta_closed    = radar_labels + [radar_labels[0]]
    fig4.add_trace(go.Scatterpolar(
        r=r_vals_closed,
        theta=theta_closed,
        fill='toself',
        name=name,
        opacity=0.7,
        line=dict(color=colors[i % len(colors)]),
    ))

fig4.update_layout(
    **DARK,
    title=dict(
        text=(
            'Radar Chart: Multi-Metric Profile of Top 8 NHS Diagnosis Categories (2023-24)'
            '<br><sub>All metrics normalized 0-100 | '
            'Larger area = higher relative burden across all dimensions</sub>'
        ),
        font=dict(size=14),
    ),
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 100],
            gridcolor='#444',
            color='white',
        ),
        angularaxis=dict(gridcolor='#444', color='white'),
        bgcolor='#0d1117',
    ),
    showlegend=True,
    legend=dict(x=1.05, y=1.0, font=dict(color='white')),
    height=650,
)

radar_div = fig4.to_html(full_html=False, include_plotlyjs=False)

sm_cols = ['admissions', 'emergency', 'waiting_list',
           'mean_los_days', 'mean_age']
sm_data = (
    df_2023
    .nlargest(60, 'admissions')
    .dropna(subset=sm_cols)
    .copy()
)

fig5 = px.scatter_matrix(
    sm_data,
    dimensions=sm_cols,
    labels={
        'admissions':   'Total Admissions',
        'emergency':    'Emergency Admissions',
        'waiting_list': 'Waiting List',
        'mean_los_days':'Avg Stay (Days)',
        'mean_age':     'Mean Age',
    },
    color='super_group',
    color_discrete_sequence=px.colors.qualitative.Safe,
    hover_name='short_desc',
)

fig5.update_traces(
    diagonal_visible=False,
    marker=dict(size=7, opacity=0.75,
                line=dict(width=0.5, color='#333')),
)

fig5.update_layout(
    **DARK,
    title=dict(
        text=(
            'Scatterplot Matrix: Pairwise Relationships Between NHS Admissions Metrics (2023-24)'
            '<br><sub>Each cell shows one metric vs another | '
            'Color = disease category group</sub>'
        ),
        font=dict(size=14),
    ),
    height=800,
)

scatter_div = fig5.to_html(full_html=False, include_plotlyjs=False)

_recovery_lookup = (
    df[df['year'].isin([2022, 2023])]
    .groupby('diagnosis_desc')['pct_change_admissions']
    .mean().round(1).to_dict()
)

_ac = df_2023[df_2023['admissions'] > 0].copy()
_ac = _ac.sort_values('admissions', ascending=False)
_ac['emergency_rate'] = np.where(
    _ac['admissions'] > 0,
    (_ac['emergency'] / _ac['admissions'] * 100).round(1), 0.0
)

_SG_COLORS = {
    'Organ System Diseases':      '#3b82f6',
    'Cancer & Blood':             '#a855f7',
    'Injuries & External Causes': '#f59e0b',
    'Infectious & Parasitic':     '#10b981',
    'Other Conditions':           '#94a3b8',
}

_ac_list = []
for _, _r in _ac.iterrows():
    _d = str(_r['diagnosis_desc'])
    _adm    = int(_r['admissions'])    if pd.notna(_r['admissions'])    else 0
    _base   = int(_r['baseline_admissions']) if pd.notna(_r['baseline_admissions']) else 0
    _ac_list.append({
        'name':           _d,
        'short_name':     (_d[:32] + '\u2026') if len(_d) > 32 else _d,
        'super_group':    str(_r['super_group']),
        'backlog_trap':   bool(_r['backlog_trap']),
        'admissions':     _adm,
        'emergency':      int(_r['emergency'])   if pd.notna(_r['emergency'])   else 0,
        'waiting_list':   int(_r['waiting_list'])if pd.notna(_r['waiting_list'])else 0,
        'mean_los_days':  round(float(_r['mean_los_days']), 1) if pd.notna(_r['mean_los_days']) else 0.0,
        'mean_age':       round(float(_r['mean_age']),       1) if pd.notna(_r['mean_age'])       else 0.0,
        'pct_change_admissions': round(float(_r['pct_change_admissions']), 1)
                                 if pd.notna(_r['pct_change_admissions']) else None,
        'pct_change_waiting':    round(float(_r['pct_change_waiting']),    1)
                                 if pd.notna(_r['pct_change_waiting'])    else None,
        'emergency_rate':  round(float(_r['emergency_rate']), 1),
        'recovery_score':  round(float(_recovery_lookup.get(_d, 0.0)), 1),
        'baseline_admissions': _base,
        'treatment_gap':  _adm - _base,
    })

all_cats_json = json.dumps({
    'cats':         _ac_list,
    'super_groups': sorted({c['super_group'] for c in _ac_list}),
    'sg_colors':    _SG_COLORS,
}, ensure_ascii=False)


@app.route('/')
def index():
    return render_template(
        'index.html',
        chart1_data_json=chart1_data_json,
        all_cats_json=all_cats_json,
        parcoords_div=parcoords_div,
        treemap_div=treemap_div,
        radar_div=radar_div,
        scatter_div=scatter_div,
        stats=dict(
            years=26,
            categories=len(df['diagnosis_desc'].unique()),
            max_year_admissions=int(
                df.groupby('year')['admissions'].sum().max()
            ),
            peak_admissions_label=(
                f"{df.groupby('year')['admissions'].sum().max()/1e6:.1f}M+"
            ),
            peak_year=int(
                df.groupby('year')['admissions'].sum().idxmax()
            ),
            backlog_count=int(
                df[df['backlog_trap'] == True]['diagnosis_desc'].nunique()
            ),
        ),
    )


if __name__ == '__main__':
    app.run(debug=True)

server = app
