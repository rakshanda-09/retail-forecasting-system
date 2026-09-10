import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.preprocessing import RobustScaler


st.set_page_config(
    page_title="RetailIQ – Demand Forecasting & Inventory Optimization 📈",
    page_icon="📈",
    layout="wide"
)


st.markdown("""
<style>
#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header{visibility:hidden;}

[data-testid="stAppViewContainer"]{
    background:#f6f8fb;
}

[data-testid="stSidebar"]{
    background:#0f172a;
    border-right:1px solid #1e293b;
}

[data-testid="stSidebar"] *{
    color:#e5e7eb;
}

[data-testid="stSidebar"] .stRadio label{
    color:#cbd5e1;
    font-size:14px;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"]{
    gap:6px;
}

[data-testid="stSidebar"] hr{
    border-color:#334155;
}

[data-testid="stSidebar"] .stSelectbox>div>div{
    background:#1e293b !important;
    border:1px solid #475569 !important;
    border-radius:9px !important;
}

[data-testid="stSidebar"] .stSelectbox>div>div:hover{
    border-color:#64748b !important;
}

[data-testid="stSidebar"] .stSelectbox input{
    color:#ffffff !important;
}

[data-testid="stSidebar"] .stSelectbox svg{
    color:#cbd5e1 !important;
}

[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"]{
    background:#1e293b !important;
}

[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] div{
    color:#ffffff !important;
}

[data-testid="stSidebar"] .stButton button{
    background:#1e293b !important;
    color:#f8fafc !important;
    border:1px solid #475569 !important;
    border-radius:9px !important;
    font-weight:600 !important;
    transition:all .2s ease;
}

[data-testid="stSidebar"] .stButton button:hover{
    background:#334155 !important;
    border-color:#64748b !important;
    color:#ffffff !important;
}

.block-container{
    padding-top:28px;
    padding-bottom:40px;
    max-width:1500px;
}

.topbar{
    display:flex;
    justify-content:space-between;
    align-items:center;
    background:#ffffff;
    border:1px solid #e5e7eb;
    border-radius:16px;
    padding:18px 24px;
    margin-bottom:24px;
    box-shadow:0 4px 18px rgba(15,23,42,.04);
}

.topbar-title{
    font-size:22px;
    font-weight:700;
    color:#111827;
    margin:0;
}

.topbar-subtitle{
    color:#6b7280;
    font-size:13px;
    margin-top:4px;
}

.status-badge{
    display:inline-flex;
    align-items:center;
    padding:6px 11px;
    border-radius:999px;
    font-size:12px;
    font-weight:600;
}

.status-open{
    background:#ecfdf3;
    color:#15803d;
}

.status-active{
    background:#eff6ff;
    color:#2563eb;
}

.status-warning{
    background:#fff7ed;
    color:#c2410c;
}

.status-danger{
    background:#fef2f2;
    color:#dc2626;
}

.kpi-card{
    background:#ffffff;
    border:1px solid #e5e7eb;
    border-radius:16px;
    padding:20px;
    min-height:125px;
    box-shadow:0 4px 18px rgba(15,23,42,.04);
}

.kpi-label{
    color:#6b7280;
    font-size:13px;
    font-weight:500;
}

.kpi-value{
    color:#111827;
    font-size:28px;
    font-weight:700;
    margin-top:8px;
}

.kpi-note{
    color:#9ca3af;
    font-size:12px;
    margin-top:5px;
}

.section-title{
    color:#111827;
    font-size:19px;
    font-weight:700;
    margin-top:28px;
    margin-bottom:5px;
}

.section-description{
    color:#6b7280;
    font-size:13px;
    margin-bottom:16px;
}

.panel{
    background:#ffffff;
    border:1px solid #e5e7eb;
    border-radius:16px;
    padding:20px;
    box-shadow:0 4px 18px rgba(15,23,42,.04);
}

.info-box{
    background:#f8fafc;
    border:1px solid #e2e8f0;
    border-radius:12px;
    padding:15px 17px;
    color:#475569;
    font-size:13px;
    line-height:1.6;
    margin-bottom:18px;
}

.login-page{
    min-height:80vh;
    display:flex;
    align-items:center;
    justify-content:center;
}

[data-testid="stForm"]{
    background:#ffffff;
    border:1px solid #e5e7eb;
    border-radius:18px;
    padding:36px 38px 30px 38px !important;
    box-shadow:0 18px 50px rgba(15,23,42,.08);
}

.login-brand{
    font-size:30px;
    font-weight:800;
    color:#111827;
    text-align:center;
    margin-bottom:5px;
}

.login-title{
    font-size:16px;
    color:#64748b;
    text-align:center;
    margin-bottom:22px;
}

.login-description{
    color:#64748b;
    font-size:13px;
    line-height:1.6;
    text-align:center;
    margin-bottom:22px;
}

.login-footer{
    text-align:center;
    color:#94a3b8;
    font-size:12px;
    margin-top:18px;
}

.metric-positive{
    color:#15803d;
    font-weight:700;
}

.metric-negative{
    color:#dc2626;
    font-weight:700;
}

.alert-card{
    background:#ffffff;
    border:1px solid #e5e7eb;
    border-left:4px solid #f97316;
    border-radius:12px;
    padding:16px;
    margin-bottom:12px;
}

.alert-title{
    color:#111827;
    font-weight:700;
    font-size:14px;
}

.alert-text{
    color:#64748b;
    font-size:13px;
    margin-top:4px;
}

.action-card{
    background:#ffffff;
    border:1px solid #e5e7eb;
    border-radius:16px;
    padding:20px;
    min-height:145px;
}

.action-title{
    color:#111827;
    font-size:15px;
    font-weight:700;
}

.action-description{
    color:#64748b;
    font-size:13px;
    line-height:1.55;
    margin-top:8px;
}

.footer{
    text-align:center;
    color:#94a3b8;
    font-size:12px;
    padding-top:30px;
}

div[data-testid="stDataFrame"]{
    border:1px solid #e5e7eb;
    border-radius:12px;
    overflow:hidden;
}

.stButton button{
    border-radius:9px;
    font-weight:600;
}

.stSelectbox label,
.stMultiSelect label,
.stRadio label{
    font-weight:600;
    color:#374151;
}

</style>
""",unsafe_allow_html=True)


def generate_sample_data():
    np.random.seed(42)

    end_date=pd.Timestamp("2026-09-06")
    dates=pd.date_range(end=end_date,periods=240,freq="D")

    stores=[f"Store {i:02d}" for i in range(1,6)]
    products=[f"P{i:03d}" for i in range(1,25)]

    categories=[
        "Grocery",
        "Beverages",
        "Personal Care",
        "Household",
        "Snacks",
        "Dairy"
    ]

    regions=[
        "West",
        "North",
        "South",
        "East",
        "Central"
    ]

    rows=[]

    store_factors={
        "Store 01":1.00,
        "Store 02":1.12,
        "Store 03":0.92,
        "Store 04":1.20,
        "Store 05":0.84
    }

    category_factors={
        "Grocery":1.10,
        "Beverages":1.25,
        "Personal Care":0.75,
        "Household":0.90,
        "Snacks":1.15,
        "Dairy":1.05
    }

    for store in stores:
        region=regions[stores.index(store)]

        for product_index,product in enumerate(products):
            category=categories[product_index%len(categories)]
            base_price=40+(product_index%12)*8

            for date in dates:
                day_of_week=date.dayofweek
                month=date.month

                weekend_factor=1.12 if day_of_week>=5 else 1.0
                seasonal_factor=1.0+0.12*np.sin((month/12)*2*np.pi)
                trend_factor=1.0+(date-dates.min()).days/240*0.10

                discount=np.random.choice(
                    [0,5,10,15,20],
                    p=[0.45,0.20,0.18,0.12,0.05]
                )

                weather=np.random.choice(
                    ["Clear","Cloudy","Rain","Hot"],
                    p=[0.45,0.20,0.20,0.15]
                )

                weather_factor={
                    "Clear":1.02,
                    "Cloudy":0.98,
                    "Rain":0.92,
                    "Hot":1.06
                }[weather]

                price=base_price*(1-discount/100)

                demand=(
                    22
                    *store_factors[store]
                    *category_factors[category]
                    *weekend_factor
                    *seasonal_factor
                    *trend_factor
                    *weather_factor
                )

                demand*=max(0.65,1+(discount/100)*0.7)
                demand+=np.random.normal(0,3.0)
                units=max(0,int(round(demand)))

                inventory=max(
                    0,
                    int(
                        units*np.random.uniform(5,13)
                        +np.random.normal(0,15)
                    )
                )

                revenue=units*price

                rows.append({
                    "Date":date,
                    "Store":store,
                    "Product":product,
                    "Category":category,
                    "Region":region,
                    "Weather":weather,
                    "Price":round(price,2),
                    "Discount":discount,
                    "Units_Sold":units,
                    "Inventory":inventory,
                    "Revenue":round(revenue,2)
                })

    return pd.DataFrame(rows)


class DemandForecaster:

    def __init__(self):
        self.model=HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.06,
            max_leaf_nodes=25,
            l2_regularization=1.0,
            random_state=42
        )
        self.scaler=RobustScaler()
        self.features=[]
        self.trained=False
        self.validation_r2=np.nan
        self.validation_mae=np.nan

    def create_features(self,data):
        data=data.copy()

        data["Date"]=pd.to_datetime(data["Date"])
        data=data.sort_values(
            ["Store","Product","Date"]
        ).reset_index(drop=True)

        grouped=data.groupby(
            ["Store","Product"],
            group_keys=False
        )

        data["Lag_1"]=grouped["Units_Sold"].shift(1)
        data["Lag_7"]=grouped["Units_Sold"].shift(7)
        data["Lag_14"]=grouped["Units_Sold"].shift(14)

        data["Rolling_7"]=grouped["Units_Sold"].transform(
            lambda x:x.shift(1).rolling(
                7,
                min_periods=3
            ).mean()
        )

        data["Rolling_14"]=grouped["Units_Sold"].transform(
            lambda x:x.shift(1).rolling(
                14,
                min_periods=5
            ).mean()
        )

        data["EWM_7"]=grouped["Units_Sold"].transform(
            lambda x:x.shift(1).ewm(
                span=7,
                adjust=False,
                min_periods=3
            ).mean()
        )

        data["Recent_Demand"]=data["Rolling_7"]

        data["Inv_Demand_Ratio"]=(
            data["Inventory"]/
            (data["Recent_Demand"]+1)
        )

        data["DayOfWeek"]=data["Date"].dt.dayofweek
        data["DayOfMonth"]=data["Date"].dt.day
        data["Month"]=data["Date"].dt.month
        data["WeekOfYear"]=data["Date"].dt.isocalendar().week.astype(int)
        data["Quarter"]=data["Date"].dt.quarter
        data["IsWeekend"]=(data["DayOfWeek"]>=5).astype(int)

        data["MonthSin"]=np.sin(
            2*np.pi*data["Month"]/12
        )

        data["MonthCos"]=np.cos(
            2*np.pi*data["Month"]/12
        )

        data["DaySin"]=np.sin(
            2*np.pi*data["DayOfWeek"]/7
        )

        data["DayCos"]=np.cos(
            2*np.pi*data["DayOfWeek"]/7
        )

        data["Effective_Price"]=data["Price"]

        return data

    def prepare_features(self,data,fit=False):
        feature_data=self.create_features(data)

        feature_data=pd.get_dummies(
            feature_data,
            columns=[
                "Store",
                "Product",
                "Category",
                "Region",
                "Weather"
            ],
            drop_first=False,
            dtype=float
        )

        excluded=[
            "Date",
            "Units_Sold",
            "Revenue"
        ]

        if fit:
            self.features=[
                column
                for column in feature_data.columns
                if column not in excluded
            ]

        for column in self.features:
            if column not in feature_data.columns:
                feature_data[column]=0.0

        return feature_data

    def train(self,data):

        feature_data=self.prepare_features(
            data,
            fit=True
        )

        feature_data=feature_data.dropna(
            subset=[
                "Lag_1",
                "Lag_7",
                "Rolling_7"
            ]
        ).copy()

        feature_data=feature_data.sort_values(
            "Date"
        ).reset_index(drop=True)

        if len(feature_data)<30:
            raise ValueError(
                "Not enough historical observations to train the demand model."
            )

        split_index=int(
            len(feature_data)*0.80
        )

        train_data=feature_data.iloc[
            :split_index
        ].copy()

        valid_data=feature_data.iloc[
            split_index:
        ].copy()

        X_train=train_data[
            self.features
        ].copy()

        y_train=train_data[
            "Units_Sold"
        ].astype(float)

        X_valid=valid_data[
            self.features
        ].copy()

        y_valid=valid_data[
            "Units_Sold"
        ].astype(float)

        self.scaler.fit(
            X_train
        )

        X_train_scaled=self.scaler.transform(
            X_train
        )

        X_valid_scaled=self.scaler.transform(
            X_valid
        )

        self.model.fit(
            X_train_scaled,
            y_train
        )

        predictions=self.model.predict(
            X_valid_scaled
        )

        predictions=np.maximum(
            predictions,
            0
        )

        self.validation_r2=r2_score(
            y_valid,
            predictions
        )

        self.validation_mae=mean_absolute_error(
            y_valid,
            predictions
        )

        self.trained=True

        return (
            self.validation_r2,
            self.validation_mae
        )

    def predict(self,data):

        if not self.trained:
            return None

        work=data.copy()

        work["_original_order"]=np.arange(
            len(work)
        )

        feature_data=self.create_features(
            work
        )

        feature_data=pd.get_dummies(
            feature_data,
            columns=[
                "Store",
                "Product",
                "Category",
                "Region",
                "Weather"
            ],
            drop_first=False,
            dtype=float
        )

        for column in self.features:
            if column not in feature_data.columns:
                feature_data[column]=0.0

        feature_data=feature_data.sort_values(
            "_original_order"
        ).reset_index(drop=True)

        X=feature_data[
            self.features
        ].copy()

        valid_mask=(
            X["Lag_1"].notna()
            &
            X["Lag_7"].notna()
            &
            X["Rolling_7"].notna()
        )

        predictions=np.full(
            len(feature_data),
            np.nan
        )

        if valid_mask.any():

            X_valid=X.loc[
                valid_mask
            ].copy()

            X_valid_scaled=self.scaler.transform(
                X_valid
            )

            predictions[valid_mask]=self.model.predict(
                X_valid_scaled
            )

        return np.maximum(
            predictions,
            0
        )


class InventoryOptimizer:

    def calculate(self,data):

        data=data.copy()

        data["Date"]=pd.to_datetime(
            data["Date"]
        )

        data=data.sort_values(
            ["Store","Product","Date"]
        ).reset_index(drop=True)

        grouped=data.groupby(
            ["Store","Product"],
            group_keys=False
        )

        data["Recent_Demand"]=grouped[
            "Units_Sold"
        ].transform(
            lambda x:x.shift(1).rolling(
                7,
                min_periods=3
            ).mean()
        )

        fallback=grouped[
            "Units_Sold"
        ].transform(
            lambda x:x.shift(1).expanding(
                min_periods=1
            ).mean()
        )

        data["Recent_Demand"]=data[
            "Recent_Demand"
        ].fillna(
            fallback
        )

        data["Recent_Demand"]=data[
            "Recent_Demand"
        ].fillna(0)

        data["DaysToStockout"]=np.where(
            data["Recent_Demand"]>0,
            data["Inventory"]/
            data["Recent_Demand"],
            np.inf
        )

        demand_std=grouped[
            "Units_Sold"
        ].transform(
            lambda x:x.shift(1).rolling(
                14,
                min_periods=5
            ).std()
        )

        demand_std=demand_std.fillna(
            data["Recent_Demand"]*0.20
        )

        data["SafetyStock"]=np.maximum(
            demand_std.fillna(0)*1.65,
            2
        )

        lead_time_days=7

        data["RecommendedReorder"]=np.maximum(
            0,
            (
                data["Recent_Demand"]*
                lead_time_days+
                data["SafetyStock"]-
                data["Inventory"]
            )
        ).round().astype(int)

        data["Risk"]=np.select(
            [
                data["DaysToStockout"]<=3,
                data["DaysToStockout"]<=7,
                data["DaysToStockout"]<=14
            ],
            [
                "Critical",
                "High",
                "Medium"
            ],
            default="Low"
        )

        return data


def generate_alerts(data):

    alerts=[]

    critical=data[
        data["Risk"]=="Critical"
    ].copy()

    high=data[
        data["Risk"]=="High"
    ].copy()

    if not critical.empty:

        for _,row in critical.nlargest(
            8,
            "RecommendedReorder"
        ).iterrows():

            alerts.append({
                "Severity":"Critical",
                "Store":row["Store"],
                "Product":row["Product"],
                "Message":(
                    f"Inventory may run out in "
                    f"{max(0,row['DaysToStockout']):.1f} days. "
                    f"Consider replenishing "
                    f"{int(row['RecommendedReorder'])} units."
                )
            })

    if not high.empty:

        for _,row in high.nlargest(
            8,
            "RecommendedReorder"
        ).iterrows():

            alerts.append({
                "Severity":"High",
                "Store":row["Store"],
                "Product":row["Product"],
                "Message":(
                    f"Inventory coverage is approximately "
                    f"{row['DaysToStockout']:.1f} days."
                )
            })

    return pd.DataFrame(
        alerts
    )


def metric_card(label,value,note):

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def page_header(title,description):

    st.markdown(
        f"""
        <div class="topbar">
            <div>
                <div class="topbar-title">{title}</div>
                <div class="topbar-subtitle">{description}</div>
            </div>
            <div>
                <span class="status-badge status-open">System active</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_header(title,description):

    st.markdown(
        f"""
        <div class="section-title">{title}</div>
        <div class="section-description">{description}</div>
        """,
        unsafe_allow_html=True
    )


def initialize_session():

    if "logged_in" not in st.session_state:
        st.session_state.logged_in=False

    if "role" not in st.session_state:
        st.session_state.role=None

    if "store" not in st.session_state:
        st.session_state.store="All Stores"

    if "model" not in st.session_state:
        st.session_state.model=None

    if "model_r2" not in st.session_state:
        st.session_state.model_r2=np.nan

    if "model_mae" not in st.session_state:
        st.session_state.model_mae=np.nan


@st.cache_data
def load_data():

    return generate_sample_data()


def login_screen(data):

    st.markdown(
        '<div class="login-page">',
        unsafe_allow_html=True
    )

    left,center,right=st.columns(
        [1,1.05,1]
    )

    with center:

        with st.form("login_form"):

            st.markdown(
                '<div class="login-brand">RetailIQ</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="login-title">Demand & Inventory Intelligence</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                """
                <div class="login-description">
                RetailIQ helps store teams understand demand,
                monitor inventory risk, and identify replenishment
                priorities using historical sales data.
                </div>
                """,
                unsafe_allow_html=True
            )

            role=st.selectbox(
                "Access level",
                [
                    "Store Manager",
                    "Administrator"
                ]
            )

            stores=[
                "All Stores"
            ]+sorted(
                data["Store"].unique().tolist()
            )

            store=st.selectbox(
                "Store",
                stores
            )

            st.caption(
                "Select the access level and store view you want to use."
            )

            submitted=st.form_submit_button(
                "Sign in",
                width="stretch"
            )

            if submitted:

                st.session_state.logged_in=True
                st.session_state.role=role
                st.session_state.store=store

                st.rerun()

        st.markdown(
            '<div class="login-footer">Retail operations workspace</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


initialize_session()

data=load_data()

if not st.session_state.logged_in:

    login_screen(data)
    st.stop()


model=st.session_state.model

optimizer=InventoryOptimizer()

inventory_data=optimizer.calculate(
    data
)

alerts_data=generate_alerts(
    inventory_data
)


with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:24px;
            font-weight:800;
            color:#ffffff;
            padding:10px 0 4px 0;
        ">
        RetailIQ
        </div>
        <div style="
            color:#94a3b8;
            font-size:12px;
            margin-bottom:24px;
        ">
        Demand & Inventory Intelligence
        </div>
        """,
        unsafe_allow_html=True
    )

    navigation=st.radio(
        "Workspace",
        [
            "Dashboard",
            "Demand Forecast",
            "Inventory",
            "Alerts",
            "Business Insights"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.markdown(
        "**Current View**"
    )

    store_options=[
        "All Stores"
    ]+sorted(
        data["Store"].unique().tolist()
    )

    selected_store=st.selectbox(
        "Store",
        store_options,
        index=store_options.index(
            st.session_state.store
        )
    )

    st.session_state.store=selected_store

    st.divider()

    st.markdown(
        "**Model**"
    )

    if model is not None and model.trained:

        st.markdown(
            '<span class="status-badge status-active">Active</span>',
            unsafe_allow_html=True
        )

        st.caption(
            "Demand model is active."
        )

    else:

        st.markdown(
            '<span class="status-badge status-warning">Not trained</span>',
            unsafe_allow_html=True
        )

        st.caption(
            "Train the demand model to generate demand estimates."
        )

    st.divider()

    st.markdown(
        "**Account**"
    )

    st.caption(
        f"{st.session_state.role} · {st.session_state.store}"
    )

    if st.button(
        "Sign out",
        width="stretch"
    ):

        st.session_state.logged_in=False
        st.session_state.model=None
        st.rerun()


if navigation=="Dashboard":

    page_header(
        "Dashboard",
        "A high-level view of demand, inventory and operational priorities."
    )

    if selected_store=="All Stores":
        view_data=data.copy()
        view_inventory=inventory_data.copy()
    else:
        view_data=data[
            data["Store"]==selected_store
        ].copy()

        view_inventory=inventory_data[
            inventory_data["Store"]==selected_store
        ].copy()

    total_units=int(
        view_data["Units_Sold"].sum()
    )

    total_revenue=float(
        view_data["Revenue"].sum()
    )

    avg_daily_units=float(
        view_data.groupby("Date")[
            "Units_Sold"
        ].sum().mean()
    )

    critical_count=int(
        (
            view_inventory["Risk"]=="Critical"
        ).sum()
    )

    high_count=int(
        (
            view_inventory["Risk"]=="High"
        ).sum()
    )

    c1,c2,c3,c4=st.columns(4)

    with c1:
        metric_card(
            "Units Sold",
            f"{total_units:,}",
            "Selected historical period"
        )

    with c2:
        metric_card(
            "Revenue",
            f"₹{total_revenue:,.0f}",
            "Historical sales value"
        )

    with c3:
        metric_card(
            "Average Daily Demand",
            f"{avg_daily_units:,.1f}",
            "Units sold per day"
        )

    with c4:
        metric_card(
            "Critical Inventory",
            str(critical_count),
            "Items requiring immediate attention"
        )

    section_header(
        "Demand overview",
        "Daily sales activity across the selected store view."
    )

    daily=view_data.groupby(
        "Date",
        as_index=False
    )["Units_Sold"].sum()

    fig=px.line(
        daily,
        x="Date",
        y="Units_Sold",
        labels={
            "Units_Sold":"Units Sold",
            "Date":"Date"
        }
    )

    fig.update_layout(
        height=360,
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10
        ),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    section_header(
        "Operational priorities",
        "Inventory categories requiring the most attention."
    )

    a,b,c=st.columns(3)

    with a:

        st.markdown(
            """
            <div class="action-card">
                <div class="action-title">Critical stock</div>
                <div class="action-description">
                Items with very low inventory coverage and a high
                probability of stockout.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.metric(
            "Items",
            critical_count
        )

    with b:

        st.markdown(
            """
            <div class="action-card">
                <div class="action-title">High stockout risk</div>
                <div class="action-description">
                Products with approximately one week or less
                of estimated inventory coverage.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.metric(
            "Items",
            high_count
        )

    with c:

        st.markdown(
            """
            <div class="action-card">
                <div class="action-title">Model status</div>
                <div class="action-description">
                Demand estimates are available after the forecasting
                model has been trained.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if model is not None and model.trained:

            st.metric(
                "Validation R²",
                f"{model.validation_r2:.3f}"
            )

        else:

            st.metric(
                "Status",
                "Not trained"
            )


elif navigation=="Demand Forecast":

    page_header(
        "Demand Forecast",
        "Review model-estimated demand and recent sales patterns."
    )

    st.markdown(
        """
        <div class="info-box">
        Demand estimates are generated from historical sales,
        recent demand patterns, pricing, discounts, calendar effects,
        store characteristics, product information and weather.
        Model performance is evaluated using a chronological holdout
        period rather than a random split.
        </div>
        """,
        unsafe_allow_html=True
    )

    train_col,info_col=st.columns(
        [1,2]
    )

    with train_col:

        if st.button(
            "Train demand model",
            width="stretch"
        ):

            with st.spinner(
                "Training demand model..."
            ):

                new_model=DemandForecaster()

                try:

                    r2,mae=new_model.train(
                        data
                    )

                    st.session_state.model=new_model
                    st.session_state.model_r2=r2
                    st.session_state.model_mae=mae

                    st.success(
                        "Demand model trained successfully."
                    )

                    st.rerun()

                except Exception as e:

                    st.error(
                        f"Model training failed: {e}"
                    )

    with info_col:

        st.caption(
            "Training uses the first 80% of observations chronologically. "
            "The remaining observations are held out for validation."
        )

    model=st.session_state.model

    c1,c2,c3=st.columns(3)

    with c1:

        if model is not None and model.trained:

            metric_card(
                "Validation R²",
                f"{model.validation_r2:.3f}",
                "Chronological holdout performance"
            )

        else:

            metric_card(
                "Validation R²",
                "—",
                "Train the model to calculate"
            )

    with c2:

        if model is not None and model.trained:

            metric_card(
                "Validation MAE",
                f"{model.validation_mae:.2f}",
                "Average units of prediction error"
            )

        else:

            metric_card(
                "Validation MAE",
                "—",
                "Train the model to calculate"
            )

    with c3:

        metric_card(
            "Model status",
            "Active" if model is not None and model.trained else "Not trained",
            "Demand model availability"
        )

    if model is not None and model.trained:

        section_header(
            "Model-estimated demand",
            "Historical demand estimates for the selected store view."
        )

        if selected_store=="All Stores":

            prediction_df=data.copy()

        else:

            prediction_df=data[
                data["Store"]==selected_store
            ].copy()

        prediction_df=prediction_df.sort_values(
            "Date"
        ).reset_index(drop=True)

        predictions=model.predict(
            prediction_df
        )

        prediction_df["Predicted_Demand"]=predictions

        prediction_df=prediction_df.dropna(
            subset=["Predicted_Demand"]
        )

        daily_prediction=prediction_df.groupby(
            "Date",
            as_index=False
        ).agg(
            Actual_Demand=(
                "Units_Sold",
                "sum"
            ),
            Predicted_Demand=(
                "Predicted_Demand",
                "sum"
            )
        )

        fig=go.Figure()

        fig.add_trace(
            go.Scatter(
                x=daily_prediction["Date"],
                y=daily_prediction["Actual_Demand"],
                mode="lines",
                name="Actual demand"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=daily_prediction["Date"],
                y=daily_prediction["Predicted_Demand"],
                mode="lines",
                name="Model estimate"
            )
        )

        fig.update_layout(
            height=390,
            margin=dict(
                l=10,
                r=10,
                t=20,
                b=10
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        section_header(
            "Recent product-level estimates",
            "Compare recent actual demand with the model estimate."
        )

        latest_date=prediction_df["Date"].max()

        latest=prediction_df[
            prediction_df["Date"]==latest_date
        ].copy()

        latest=latest[
            [
                "Store",
                "Product",
                "Category",
                "Units_Sold",
                "Predicted_Demand",
                "Inventory"
            ]
        ].sort_values(
            "Predicted_Demand",
            ascending=False
        )

        latest["Predicted_Demand"]=latest[
            "Predicted_Demand"
        ].round(1)

        st.dataframe(
            latest,
            width="stretch",
            hide_index=True
        )

    else:

        st.info(
            "The demand model is not trained yet. "
            "Select a store if needed and click "
            "'Train demand model' to activate forecasting."
        )


elif navigation=="Inventory":

    page_header(
        "Inventory",
        "Monitor stock coverage and replenishment requirements."
    )

    if selected_store=="All Stores":

        view_inventory=inventory_data.copy()

    else:

        view_inventory=inventory_data[
            inventory_data["Store"]==selected_store
        ].copy()

    c1,c2,c3,c4=st.columns(4)

    critical=int(
        (
            view_inventory["Risk"]=="Critical"
        ).sum()
    )

    high=int(
        (
            view_inventory["Risk"]=="High"
        ).sum()
    )

    medium=int(
        (
            view_inventory["Risk"]=="Medium"
        ).sum()
    )

    reorder_units=int(
        view_inventory[
            "RecommendedReorder"
        ].sum()
    )

    with c1:
        metric_card(
            "Critical",
            critical,
            "Immediate replenishment attention"
        )

    with c2:
        metric_card(
            "High Risk",
            high,
            "Low inventory coverage"
        )

    with c3:
        metric_card(
            "Medium Risk",
            medium,
            "Monitor stock position"
        )

    with c4:
        metric_card(
            "Recommended Reorder",
            f"{reorder_units:,}",
            "Estimated replenishment units"
        )

    section_header(
        "Inventory position",
        "Use inventory coverage and reorder quantity to prioritize products."
    )

    risk_filter=st.multiselect(
        "Risk level",
        [
            "Critical",
            "High",
            "Medium",
            "Low"
        ],
        default=[
            "Critical",
            "High",
            "Medium",
            "Low"
        ]
    )

    table=view_inventory[
        view_inventory["Risk"].isin(
            risk_filter
        )
    ].copy()

    table=table[
        [
            "Store",
            "Product",
            "Category",
            "Inventory",
            "Recent_Demand",
            "DaysToStockout",
            "SafetyStock",
            "RecommendedReorder",
            "Risk"
        ]
    ]

    table["Recent_Demand"]=table[
        "Recent_Demand"
    ].round(1)

    table["DaysToStockout"]=table[
        "DaysToStockout"
    ].replace(
        np.inf,
        np.nan
    ).round(1)

    table["SafetyStock"]=table[
        "SafetyStock"
    ].round(1)

    table=table.sort_values(
        [
            "Risk",
            "RecommendedReorder"
        ],
        ascending=[
            True,
            False
        ]
    )

    st.dataframe(
        table,
        width="stretch",
        hide_index=True
    )

    section_header(
        "Inventory coverage",
        "Products with the shortest estimated time to stockout."
    )

    chart_data=view_inventory[
        view_inventory["DaysToStockout"]<30
    ].copy()

    chart_data=chart_data.nsmallest(
        20,
        "DaysToStockout"
    )

    if not chart_data.empty:

        fig=px.bar(
            chart_data,
            x="DaysToStockout",
            y="Product",
            color="Risk",
            orientation="h",
            labels={
                "DaysToStockout":"Estimated Days to Stockout",
                "Product":"Product"
            }
        )

        fig.update_layout(
            height=500,
            margin=dict(
                l=10,
                r=10,
                t=20,
                b=10
            ),
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


elif navigation=="Alerts":

    page_header(
        "Alerts",
        "Review inventory conditions that may require operational action."
    )

    if alerts_data.empty:

        st.success(
            "No critical or high-risk inventory alerts were identified."
        )

    else:

        if selected_store!="All Stores":

            filtered_alerts=alerts_data[
                alerts_data["Store"]==selected_store
            ].copy()

        else:

            filtered_alerts=alerts_data.copy()

        if filtered_alerts.empty:

            st.success(
                "No high-priority alerts are currently associated with the selected store."
            )

        else:

            section_header(
                "Priority alerts",
                "These alerts are based on recent demand and current inventory coverage."
            )

            for _,row in filtered_alerts.iterrows():

                severity_class=(
                    "status-danger"
                    if row["Severity"]=="Critical"
                    else "status-warning"
                )

                st.markdown(
                    f"""
                    <div class="alert-card">
                        <div>
                            <span class="status-badge {severity_class}">
                            {row["Severity"]}
                            </span>
                        </div>
                        <div class="alert-title" style="margin-top:9px;">
                        {row["Store"]} · {row["Product"]}
                        </div>
                        <div class="alert-text">
                        {row["Message"]}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


elif navigation=="Business Insights":

    page_header(
        "Business Insights",
        "Understand demand drivers, store performance and inventory exposure."
    )

    if selected_store=="All Stores":

        view_data=data.copy()
        view_inventory=inventory_data.copy()

    else:

        view_data=data[
            data["Store"]==selected_store
        ].copy()

        view_inventory=inventory_data[
            inventory_data["Store"]==selected_store
        ].copy()

    section_header(
        "Category performance",
        "Historical unit sales by product category."
    )

    category_data=view_data.groupby(
        "Category",
        as_index=False
    )["Units_Sold"].sum()

    category_data=category_data.sort_values(
        "Units_Sold",
        ascending=False
    )

    fig=px.bar(
        category_data,
        x="Category",
        y="Units_Sold",
        labels={
            "Units_Sold":"Units Sold",
            "Category":"Category"
        }
    )

    fig.update_layout(
        height=380,
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10
        ),
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    section_header(
        "Store performance",
        "Compare historical sales volume across stores."
    )

    store_data=data.groupby(
        "Store",
        as_index=False
    ).agg(
        Units_Sold=(
            "Units_Sold",
            "sum"
        ),
        Revenue=(
            "Revenue",
            "sum"
        ),
        Avg_Daily_Demand=(
            "Units_Sold",
            "mean"
        )
    )

    store_data["Revenue"]=store_data[
        "Revenue"
    ].round(0)

    store_data["Avg_Daily_Demand"]=store_data[
        "Avg_Daily_Demand"
    ].round(2)

    st.dataframe(
        store_data.sort_values(
            "Units_Sold",
            ascending=False
        ),
        width="stretch",
        hide_index=True
    )

    section_header(
        "Inventory exposure",
        "Identify where current stock positions create operational risk."
    )

    exposure=view_inventory.groupby(
        "Risk",
        as_index=False
    ).agg(
        Products=(
            "Product",
            "count"
        ),
        Inventory=(
            "Inventory",
            "sum"
        ),
        RecommendedReorder=(
            "RecommendedReorder",
            "sum"
        )
    )

    st.dataframe(
        exposure,
        width="stretch",
        hide_index=True
    )

    if model is not None and model.trained:

        section_header(
            "Model interpretation",
            "Summary of the current demand model performance."
        )

        m1,m2=st.columns(2)

        with m1:

            st.markdown(
                f"""
                <div class="panel">
                    <div class="kpi-label">
                    Chronological validation R²
                    </div>
                    <div class="kpi-value">
                    {model.validation_r2:.3f}
                    </div>
                    <div class="kpi-note">
                    Measures how much variation in the held-out
                    demand observations is explained by the model.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with m2:

            st.markdown(
                f"""
                <div class="panel">
                    <div class="kpi-label">
                    Validation MAE
                    </div>
                    <div class="kpi-value">
                    {model.validation_mae:.2f}
                    </div>
                    <div class="kpi-note">
                    Average absolute difference between actual
                    and model-estimated demand.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.info(
            "Train the demand model to make model performance available "
            "in the business insights section."
        )


st.markdown(
    """
    <div class="footer">
    RetailIQ · Demand forecasting and inventory intelligence
    </div>
    """,
    unsafe_allow_html=True
)