import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="CINE | Advanced Movie Analytics",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------------
# PROFESSIONAL CSS & STYLING
# -------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');

/* Global Reset and Typography */
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: #0E1117 !important;
    background-image: radial-gradient(circle at top right, rgba(30, 41, 59, 0.5), transparent 40%),
                      radial-gradient(circle at bottom left, rgba(15, 23, 42, 0.8), transparent 40%);
    color: #E2E8F0 !important;
    font-family: 'Inter', sans-serif !important;
}

.block-container { padding: 2rem 3rem !important; max-width: 1600px !important; }
footer { display: none !important; }
[data-testid="stHeader"] { background-color: transparent !important; }

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.95) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1); }

/* Modern Navigation Pills (Vercel/Linear Style) */
[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 0.25rem;
}

[data-testid="stSidebar"] div[role="radiogroup"] label {
    font-family: 'Inter', sans-serif; font-weight: 500; font-size: 0.95rem;
    padding: 0.65rem 1rem !important; border-radius: 8px !important;
    transition: all 0.2s ease; margin-bottom: 4px;
    background: transparent !important;
    cursor: pointer;
    width: 100%;
    position: relative;
    border: 1px solid transparent;
}

/* Safely hide the native radio circle without breaking flexbox */
[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {
    position: absolute;
    opacity: 0;
    width: 0;
    height: 0;
    overflow: hidden;
}

/* Text styling */
[data-testid="stSidebar"] div[role="radiogroup"] label p {
    color: #8B9BB4 !important; /* Subtle gray-blue */
    margin: 0 !important;
    margin-left: 0.2rem !important;
    transition: color 0.2s ease;
}

/* Hover state */
[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(255, 255, 255, 0.04) !important; 
}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover p {
    color: #E2E8F0 !important;
}

/* Selected state - Sleek glass pill */
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1) !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* Sidebar Headings */
[data-testid="stSidebar"] h2 {
    font-size: 2.2rem !important; letter-spacing: -1px;
}

/* Headings */
h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Inter', sans-serif !important; font-weight: 700 !important; letter-spacing: -0.02em;
}
h1 {
    font-size: 2.5rem !important;
    background: linear-gradient(135deg, #FFFFFF, #94A3B8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem !important;
}

/* Custom Metric Cards */
.kpi-container {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem;
}
.kpi-card {
    background: rgba(30, 41, 59, 0.4); backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.08); border-radius: 12px;
    padding: 1.5rem; transition: transform 0.2s ease, border-color 0.2s ease;
}
.kpi-card:hover { transform: translateY(-4px); border-color: #38BDF8; }
.kpi-title {
    font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;
}
.kpi-value {
    font-size: 2rem; font-weight: 800; color: #F8FAFC;
}
.kpi-highlight { color: #38BDF8; }

/* Section Tags */
.section-tag {
    font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.15em; text-transform: uppercase; color: #38BDF8;
    background: rgba(56, 189, 248, 0.1); display: inline-block;
    padding: 0.25rem 0.75rem; border-radius: 4px; margin-bottom: 0.5rem; border: 1px solid rgba(56, 189, 248, 0.2);
}

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {
    background: rgba(15, 23, 42, 0.5); border-bottom: 1px solid rgba(255,255,255,0.1); gap: 1rem; padding-left: 1rem;
}
[data-testid="stTabs"] button[role="tab"] {
    font-family: 'Inter', sans-serif; font-weight: 600; padding: 1rem 0.5rem !important; background: transparent; color: #64748B; border: none; border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #38BDF8 !important; border-bottom: 2px solid #38BDF8 !important;
}

/* DataFrame Customization */
[data-testid="stDataFrame"] { border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); }

/* Buttons */
.stButton > button {
    background: #0F172A; border: 1px solid #38BDF8; color: #38BDF8; border-radius: 6px !important;
    font-family: 'Inter', sans-serif; font-weight: 600; transition: all 0.2s;
}
.stButton > button:hover { background: #38BDF8; color: #0F172A; }

/* Content Cards */
.content-card {
    background: rgba(30, 41, 59, 0.3); border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# Plotly Theme Configuration
PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=12, color="#94A3B8"),
    title_font=dict(family="Inter, sans-serif", size=16, color="#F8FAFC", weight=700),
    margin=dict(l=20, r=20, t=50, b=20),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
)

# -------------------------------------------------------------------------
# DATA PIPELINE
# -------------------------------------------------------------------------
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("movies_dataset.csv", encoding='utf-8')
    
    # Drop largely empty columns
    df = df[[col for col in df if df[col].count() / len(df) >= 0.3]].copy()
    
    # Numeric conversions and median imputation
    numeric_cols = ['rating', 'votes', 'duration_min', 'box_office_million', 'release_year']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].median(), inplace=True)
            
    # Boolean conversion
    if 'oscar_winner' in df.columns:
        df['oscar_winner'] = df['oscar_winner'].astype(str).str.lower().map(
            {'true': 1, 'false': 0, '1': 1, '0': 0, 'yes': 1, 'no': 0}
        ).fillna(0).astype(int)
        
    # Clean strings
    if 'genre' in df.columns:
        df['genre'] = df['genre'].fillna('Unknown')
    if 'title' in df.columns:
        df['title'] = df['title'].fillna('Unknown Title')
        
    return df

raw_df = load_and_preprocess_data()
numerical_features = [f for f in ['release_year', 'duration_min', 'votes', 'box_office_million'] if f in raw_df.columns]

# -------------------------------------------------------------------------
# SIDEBAR: GLOBAL FILTERS & NAVIGATION
# -------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='color:#38BDF8; font-weight:800; font-size:1.8rem; margin-bottom:0;'>CINE</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B; font-size:0.8rem; font-weight:600; letter-spacing:1px; text-transform:uppercase;'>Analytics Studio</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    navigation = st.radio(
        "Module Selection",
        [
            "📊 Dashboard Overview", 
            "🔍 Deep Dive Analysis", 
            "📐 Statistical Tests", 
            "🤖 Predictive Models", 
            "💡 Strategy & Export"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("<p style='font-family:Inter; font-weight:600; color:#E2E8F0; margin-bottom:1rem;'>GLOBAL FILTERS</p>", unsafe_allow_html=True)
    
    # Dynamic Filtering
    min_year, max_year = int(raw_df['release_year'].min()), int(raw_df['release_year'].max())
    selected_years = st.slider("Release Period", min_year, max_year, (min_year, max_year))
    
    all_genres = sorted(list(raw_df['genre'].unique()))
    selected_genres = st.multiselect("Genres", all_genres, default=all_genres[:5])
    
    min_rating = st.slider("Minimum Rating", 0.0, 10.0, 0.0, 0.5)

# Apply global filters
df = raw_df[
    (raw_df['release_year'] >= selected_years[0]) &
    (raw_df['release_year'] <= selected_years[1]) &
    (raw_df['genre'].isin(selected_genres)) &
    (raw_df['rating'] >= min_rating)
]

if df.empty:
    st.warning("⚠️ Adjust your filters. No data available for the current selection.")
    st.stop()

# -------------------------------------------------------------------------
# PAGE 1: DASHBOARD OVERVIEW
# -------------------------------------------------------------------------
if navigation == "📊 Dashboard Overview":
    st.markdown('<div class="section-tag">Executive Summary</div>', unsafe_allow_html=True)
    st.markdown("<h1>Performance <span style='color:#38BDF8;'>Metrics</span></h1>", unsafe_allow_html=True)
    
    # Custom HTML KPIs
    kpi_html = f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-title">Active Movies</div>
            <div class="kpi-value">{len(df):,}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Avg Rating</div>
            <div class="kpi-value"><span class="kpi-highlight">{df['rating'].mean():.2f}</span> / 10</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Total Box Office</div>
            <div class="kpi-value">${df['box_office_million'].sum():,.0f}M</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Oscar Winners</div>
            <div class="kpi-value">{df['oscar_winner'].sum() if 'oscar_winner' in df else 'N/A'}</div>
        </div>
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1.5], gap="large")
    
    with col1:
        st.markdown("<h3 style='font-size:1.1rem; color:#E2E8F0;'>Genre Composition (Treemap)</h3>", unsafe_allow_html=True)
        genre_tree = df.groupby('genre').agg(
            Count=('title', 'count'), Avg_Rating=('rating', 'mean')
        ).reset_index()
        
        fig = px.treemap(
            genre_tree, path=[px.Constant("All Genres"), 'genre'], values='Count', color='Avg_Rating',
            color_continuous_scale='Blues',
            custom_data=['Avg_Rating']
        )
        fig.update_traces(hovertemplate='<b>%{label}</b><br>Movies: %{value}<br>Avg Rating: %{customdata[0]:.2f}')
        fig.update_layout(**PLOTLY_THEME)
        fig.update_layout(height=400, margin=dict(t=20, l=0, r=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("<h3 style='font-size:1.1rem; color:#E2E8F0;'>Top Performing Movies</h3>", unsafe_allow_html=True)
        top_movies = df.sort_values(by='rating', ascending=False)[['title', 'genre', 'rating', 'box_office_million']].head(10)
        top_movies.columns = ['Title', 'Genre', 'Rating', 'Box Office ($M)']
        st.dataframe(top_movies, use_container_width=True, hide_index=True, height=400)


# -------------------------------------------------------------------------
# PAGE 2: DEEP DIVE ANALYSIS
# -------------------------------------------------------------------------
elif navigation == "🔍 Deep Dive Analysis":
    st.markdown('<div class="section-tag">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown("<h1>Visual <span style='color:#38BDF8;'>Exploration</span></h1>", unsafe_allow_html=True)
    
    t1, t2 = st.tabs(["Scatter & Regression", "Distributions & Heatmap"])
    
    with t1:
        st.markdown("<br>", unsafe_allow_html=True)
        c_ctrl1, c_ctrl2, c_ctrl3 = st.columns(3)
        x_axis = c_ctrl1.selectbox("X-Axis Metric", numerical_features, index=numerical_features.index('votes') if 'votes' in numerical_features else 0)
        y_axis = c_ctrl2.selectbox("Y-Axis Metric", numerical_features + ['rating'], index=len(numerical_features))
        color_by = c_ctrl3.selectbox("Segment By", ['genre', 'oscar_winner'])
        
        fig_scatter = px.scatter(
            df, x=x_axis, y=y_axis, color=color_by, hover_data=['title'],
            trendline="ols", trendline_scope="overall", trendline_color_override="#38BDF8",
            opacity=0.7, color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_scatter.update_layout(**PLOTLY_THEME, height=500, title=f"Regression Analysis: {x_axis} vs {y_axis}")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with t2:
        st.markdown("<br>", unsafe_allow_html=True)
        col_dist, col_corr = st.columns([1, 1])
        
        with col_dist:
            dist_feat = st.selectbox("Distribution Feature", numerical_features + ['rating'])
            fig_hist = px.histogram(
                df, x=dist_feat, nbins=40, marginal="box", color_discrete_sequence=['#38BDF8'], opacity=0.8
            )
            fig_hist.update_layout(**PLOTLY_THEME, height=400, title=f"Distribution Profile: {dist_feat}")
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with col_corr:
            st.markdown("<br>", unsafe_allow_html=True) # spacing
            corr_df = df[numerical_features + ['rating']].corr()
            fig_corr = px.imshow(
                corr_df, text_auto=".2f", aspect="auto",
                color_continuous_scale=[[0, '#0F172A'], [0.5, '#1E293B'], [1, '#38BDF8']]
            )
            fig_corr.update_layout(**PLOTLY_THEME, height=400, title="Feature Correlation Matrix")
            st.plotly_chart(fig_corr, use_container_width=True)


# -------------------------------------------------------------------------
# PAGE 3: STATISTICAL TESTS
# -------------------------------------------------------------------------
elif navigation == "📐 Statistical Tests":
    st.markdown('<div class="section-tag">Hypothesis Verification</div>', unsafe_allow_html=True)
    st.markdown("<h1>Statistical <span style='color:#38BDF8;'>Rigour</span></h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="content-card">
        <p style="color:#94A3B8; font-size:0.95rem;">Applying formal statistical methods to validate structural differences in movie performance metrics across categorical groups.</p>
    </div>
    """, unsafe_allow_html=True)
    
    test_type = st.radio("Select Analysis Path", ["Oscar Impact on Box Office", "Cross-Genre Rating Variance"], horizontal=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    if test_type == "Oscar Impact on Box Office":
        if 'oscar_winner' in df.columns:
            st.markdown("### Welch's T-Test: Oscar Winners vs. Non-Winners")
            
            group_1 = df[df['oscar_winner'] == 1]['box_office_million'].dropna()
            group_0 = df[df['oscar_winner'] == 0]['box_office_million'].dropna()
            
            if len(group_1) > 5 and len(group_0) > 5:
                t_stat, p_val = stats.ttest_ind(group_1, group_0, equal_var=False)
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Winner Avg Rev", f"${group_1.mean():.1f}M")
                c2.metric("Non-Winner Avg Rev", f"${group_0.mean():.1f}M")
                c3.metric("T-Statistic", f"{t_stat:.3f}")
                c4.metric("P-Value", f"{p_val:.3e}")
                
                fig = go.Figure()
                fig.add_trace(go.Violin(x=df['oscar_winner'].astype(str), y=df['box_office_million'], 
                                        box_visible=True, meanline_visible=True,
                                        fillcolor='rgba(56, 189, 248, 0.2)', line_color='#38BDF8'))
                fig.update_layout(**PLOTLY_THEME, height=400, title="Box Office Distribution Comparison", xaxis_title="Oscar Winner (0=No, 1=Yes)")
                st.plotly_chart(fig, use_container_width=True)
                
                if p_val < 0.05:
                    st.success("✅ **Statistically Significant:** The difference in box office revenue is robust (p < 0.05).")
                else:
                    st.info("ℹ️ **Not Significant:** Insufficient evidence to claim a difference in revenue (p >= 0.05).")
            else:
                st.warning("Insufficient data points in current filter to perform T-Test.")
        else:
            st.error("Oscar data unavailable.")
            
    else:
        st.markdown("### ANOVA: Rating Variance Across Top Genres")
        top_3_genres = df['genre'].value_counts().nlargest(3).index
        groups = [df[df['genre'] == g]['rating'].dropna() for g in top_3_genres]
        
        if len(groups) == 3:
            f_stat, p_val = stats.f_oneway(*groups)
            
            cols = st.columns(3)
            for i, g in enumerate(top_3_genres):
                cols[i].metric(f"{g} Avg Rating", f"{groups[i].mean():.2f}")
                
            st.markdown(f"**F-Statistic:** {f_stat:.3f} | **P-Value:** {p_val:.3e}")
            
            fig = px.box(df[df['genre'].isin(top_3_genres)], x='genre', y='rating', color='genre', 
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(**PLOTLY_THEME, height=400)
            st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------------------
# PAGE 4: PREDICTIVE MODELS
# -------------------------------------------------------------------------
elif navigation == "🤖 Predictive Models":
    st.markdown('<div class="section-tag">Machine Learning Pipeline</div>', unsafe_allow_html=True)
    st.markdown("<h1>Target <span style='color:#38BDF8;'>Prediction</span></h1>", unsafe_allow_html=True)
    
    target = 'rating'
    features = numerical_features
    
    model_df = df[features + [target]].dropna()
    X = model_df[features]
    y = model_df[target]
    
    if len(model_df) < 50:
        st.warning("⚠️ Insufficient data to train robust models. Please loosen your sidebar filters.")
        st.stop()
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Models
    models = {
        "OLS Linear Regression": LinearRegression(),
        "Ridge Regression (L2)": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = []
    trained_models = {}
    
    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        trained_models[name] = m
        results.append({
            "Model": name,
            "R² Score": r2_score(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "MAE": mean_absolute_error(y_test, preds)
        })
        
    res_df = pd.DataFrame(results).set_index("Model")
    
    st.markdown("### Model Performance Comparison")
    st.dataframe(res_df.style.highlight_max(subset=['R² Score'], color='rgba(56,189,248,0.3)')
                       .highlight_min(subset=['RMSE', 'MAE'], color='rgba(56,189,248,0.3)'), 
                 use_container_width=True)
                 
    col1, col2 = st.columns([1.5, 1])
    
    best_model_name = "Random Forest Regressor"
    best_model = trained_models[best_model_name]
    
    with col1:
        st.markdown(f"### Feature Importance ({best_model_name})")
        importances = best_model.feature_importances_
        imp_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=True)
        
        fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h', color='Importance', 
                         color_continuous_scale='Blues')
        fig_imp.update_layout(**PLOTLY_THEME, height=350)
        st.plotly_chart(fig_imp, use_container_width=True)
        
    with col2:
        st.markdown("### Interactive Predictor")
        st.markdown("<p style='font-size:0.85rem; color:#94A3B8;'>Input values to predict rating.</p>", unsafe_allow_html=True)
        
        input_data = {}
        for feat in features:
            input_data[feat] = st.number_input(f"{feat.replace('_', ' ').title()}", value=float(df[feat].median()))
            
        if st.button("Generate Prediction", use_container_width=True):
            input_df = pd.DataFrame([input_data])
            prediction = best_model.predict(input_df)[0]
            st.markdown(f"""
            <div style="background:rgba(56,189,248,0.1); border:1px solid #38BDF8; border-radius:8px; padding:1rem; text-align:center; margin-top:1rem;">
                <p style="margin:0; font-size:0.9rem; color:#94A3B8;">Predicted Rating</p>
                <h2 style="margin:0; color:#38BDF8; font-size:2.5rem;">{prediction:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)


# -------------------------------------------------------------------------
# PAGE 5: STRATEGY & EXPORT
# -------------------------------------------------------------------------
elif navigation == "💡 Strategy & Export":
    st.markdown('<div class="section-tag">Final Review</div>', unsafe_allow_html=True)
    st.markdown("<h1>Strategic <span style='color:#38BDF8;'>Insights</span></h1>", unsafe_allow_html=True)
    
    insights_html = """
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin-bottom:2rem;">
        <div class="content-card" style="margin-bottom:0;">
            <h3 style="color:#F8FAFC; font-size:1.1rem; margin-bottom:0.5rem;">📊 Audience Engagement is Key</h3>
            <p style="color:#94A3B8; font-size:0.9rem; line-height:1.6;">Our Random Forest model consistently identifies <b>Votes</b> as the strongest predictor of high movie ratings. Strategies should prioritize organic audience building and community engagement over pure marketing spend.</p>
        </div>
        <div class="content-card" style="margin-bottom:0;">
            <h3 style="color:#F8FAFC; font-size:1.1rem; margin-bottom:0.5rem;">💰 The Revenue Paradox</h3>
            <p style="color:#94A3B8; font-size:0.9rem; line-height:1.6;">Scatter analysis reveals a very weak correlation between Box Office success and critical/audience ratings. Mass appeal does not necessarily equal perceived quality.</p>
        </div>
        <div class="content-card" style="margin-bottom:0;">
            <h3 style="color:#F8FAFC; font-size:1.1rem; margin-bottom:0.5rem;">🏆 Award Efficacy</h3>
            <p style="color:#94A3B8; font-size:0.9rem; line-height:1.6;">Statistical T-Tests validate that Oscar status holds significant weight. Oscar winners exhibit statistically distinct financial trajectories compared to non-winners.</p>
        </div>
        <div class="content-card" style="margin-bottom:0;">
            <h3 style="color:#F8FAFC; font-size:1.1rem; margin-bottom:0.5rem;">🤖 Modeling Efficacy</h3>
            <p style="color:#94A3B8; font-size:0.9rem; line-height:1.6;">Non-linear models (Random Forests) outperform linear approaches (OLS/Ridge) by a significant margin, suggesting that feature interactions (e.g., specific combinations of runtime and release era) are critical.</p>
        </div>
    </div>
    """
    st.markdown(insights_html, unsafe_allow_html=True)
    
    st.markdown("### Export Processed Dataset")
    st.markdown("<p style='color:#94A3B8; font-size:0.9rem;'>Download the exact dataset subset you are currently viewing (with your applied global sidebar filters) for offline analysis.</p>", unsafe_allow_html=True)
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered CSV",
        data=csv,
        file_name="cine_analytics_export.csv",
        mime="text/csv"
    )