import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc # dcc will be dash.dcc in newer versions
import dash_html_components as html # html will be dash.html in newer versions
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 1. Generate Sample Data (Enhanced for more KPIs)
np.random.seed(42)
num_rows = 250
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)

data = {
    'OrderID': range(1, num_rows + 1),
    'OrderDate': [start_date + timedelta(days=np.random.randint(0, (end_date - start_date).days)) for _ in range(num_rows)],
    'Region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], num_rows, p=[0.20, 0.25, 0.20, 0.15, 0.20]),
    'ProductCategory': np.random.choice(['Electronics', 'Apparel', 'Home Goods', 'Books', 'Software'], num_rows, p=[0.25, 0.20, 0.25, 0.15, 0.15]),
    'Salesperson': np.random.choice(['Alice', 'Bob', 'Charlie', 'Diana', 'Edward', 'Fiona'], num_rows),
    'UnitsSold': np.random.randint(1, 20, num_rows),
    'UnitPrice': np.random.choice([19.99, 29.99, 49.99, 79.99, 129.99, 199.99], num_rows),
    'CustomerSatisfaction': np.random.randint(1, 6, num_rows) # 1 to 5 scale
}
df = pd.DataFrame(data)
df['SalesAmount'] = df['UnitsSold'] * df['UnitPrice']
df['OrderMonth'] = df['OrderDate'].dt.to_period('M')
df['OrderMonthStr'] = df['OrderMonth'].astype(str) # For grouping and display

# --- Power BI Style Color Palette ---
power_bi_colors = ['#01B8AA', '#374649', '#FD625E', '#F2C80F', '#5F6B6D', '#8AD4EB', '#FE9666', '#A66999']
power_bi_bg_color = '#F3F2F1'
power_bi_card_bg = '#FFFFFF'
power_bi_text_color = '#252423'
power_bi_title_color = '#0078D4' # A common Power BI blue

# 2. Calculate KPIs & Data for KPI Tiles
total_sales = df['SalesAmount'].sum()
total_units_sold = df['UnitsSold'].sum()
avg_order_value = df['SalesAmount'].mean()
unique_customers = df['Salesperson'].nunique() # Assuming salesperson is a proxy for unique interactions or key accounts
avg_satisfaction = df['CustomerSatisfaction'].mean()

# For trend indicators (simplified - comparing last month to previous month)
current_month_period = df['OrderMonth'].max()
previous_month_period = current_month_period - 1

sales_current_month = df[df['OrderMonth'] == current_month_period]['SalesAmount'].sum()
sales_previous_month = df[df['OrderMonth'] == previous_month_period]['SalesAmount'].sum()
sales_trend_icon = "▲" if sales_current_month > sales_previous_month else "▼"
sales_trend_color = "green" if sales_current_month > sales_previous_month else "red"
sales_trend_text = f"{sales_trend_icon} vs Prev. Month"

units_current_month = df[df['OrderMonth'] == current_month_period]['UnitsSold'].sum()
units_previous_month = df[df['OrderMonth'] == previous_month_period]['UnitsSold'].sum()
units_trend_icon = "▲" if units_current_month > units_previous_month else "▼"
units_trend_color = "green" if units_current_month > units_previous_month else "red"
units_trend_text = f"{units_trend_icon} vs Prev. Month"


# --- Helper function for creating styled KPI tiles ---
def create_styled_kpi_tile(title, main_value, secondary_text="", trend_indicator="", trend_color="grey", icon_class=""):
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.I(className=f"{icon_class} mr-2", style={'fontSize': '1.5rem', 'color': power_bi_title_color}) if icon_class else None,
                html.H5(title, className="card-title mb-1", style={'color': power_bi_title_color, 'fontSize': '1rem', 'fontWeight':'500'}),
            ], style={'display': 'flex', 'alignItems': 'center'}),
            html.H3(main_value, className="card-text my-2", style={'color': power_bi_text_color, 'fontSize': '1.8rem', 'fontWeight': 'bold'}),
            html.P(secondary_text, className="card-text_secondary_text", style={'fontSize': '0.8rem', 'color': 'grey'}),
            html.Small(trend_indicator, style={'color': trend_color, 'fontWeight': 'bold', 'fontSize': '0.9rem'})
        ]),
        className="m-2", # Margin for spacing between tiles
        style={
            'backgroundColor': power_bi_card_bg,
            'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.05)',
            'border': 'none',
            'minHeight': '150px', # Ensure consistent height for tiles in a row
            'display': 'flex',
            'flexDirection': 'column',
            'justifyContent': 'space-between'
        }
    )

# 3. Create Figures for Main Charts
chart_layout_defaults = {
    'plot_bgcolor': power_bi_card_bg,
    'paper_bgcolor': power_bi_card_bg,
    'font_color': power_bi_text_color,
    'title_font_color': power_bi_title_color,
    'legend_font_color': power_bi_text_color,
    'xaxis': {'gridcolor': '#e0e0e0'},
    'yaxis': {'gridcolor': '#e0e0e0'}
}

# Sales by Product Category
category_sales = df.groupby('ProductCategory')['SalesAmount'].sum().reset_index().sort_values(by='SalesAmount', ascending=False)
fig_category_sales = px.bar(category_sales, x='ProductCategory', y='SalesAmount', title='Revenue by Product Category', color_discrete_sequence=[power_bi_colors[0]])
fig_category_sales.update_layout(**chart_layout_defaults)

# Sales Over Time (Monthly)
monthly_sales = df.groupby('OrderMonthStr')['SalesAmount'].sum().reset_index().sort_values(by='OrderMonthStr')
fig_monthly_sales = px.area(monthly_sales, x='OrderMonthStr', y='SalesAmount', title='Monthly Revenue Trend', markers=True, color_discrete_sequence=[power_bi_colors[1]])
fig_monthly_sales.update_layout(**chart_layout_defaults)
fig_monthly_sales.update_xaxes(type='category') # Treat month string as category for proper sorting

# Sales by Region (Donut Chart)
region_sales = df.groupby('Region')['SalesAmount'].sum().reset_index()
fig_region_sales = px.pie(region_sales, values='SalesAmount', names='Region', title='Revenue Share by Region', hole=0.4, color_discrete_sequence=power_bi_colors[2:])
fig_region_sales.update_layout(**chart_layout_defaults)

# Customer Satisfaction Distribution
satisfaction_counts = df['CustomerSatisfaction'].value_counts().reset_index()
satisfaction_counts.columns = ['Rating', 'Count']
satisfaction_counts = satisfaction_counts.sort_values('Rating')
fig_satisfaction = px.bar(satisfaction_counts, x='Rating', y='Count', title='Customer Satisfaction Ratings', color_discrete_sequence=[power_bi_colors[3]])
fig_satisfaction.update_layout(**chart_layout_defaults)


# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME])
app.title = "Sales Performance Dashboard"

# 4. App Layout - More Tiled
app.layout = html.Div(style={'backgroundColor': power_bi_bg_color, 'padding': '15px', 'color': power_bi_text_color, 'minHeight':'100vh'}, children=[
    dbc.Row(
        dbc.Col(html.H2("Sales Performance Dashboard", className="text-center mb-4 p-2", style={'color': power_bi_card_bg, 'backgroundColor': power_bi_title_color, 'fontWeight': 'bold'}), width=12)
    ),

    # Row for KPI Tiles
    dbc.Row([
        dbc.Col(create_styled_kpi_tile("Total Revenue", f"${total_sales:,.0f}", "All-time total revenue", sales_trend_text, sales_trend_color, icon_class="fas fa-dollar-sign"), lg=3, md=6, sm=12),
        dbc.Col(create_styled_kpi_tile("Units Sold", f"{total_units_sold:,}", "All-time units sold", units_trend_text, units_trend_color, icon_class="fas fa-shopping-cart"), lg=3, md=6, sm=12),
        dbc.Col(create_styled_kpi_tile("Avg. Order Value", f"${avg_order_value:,.2f}", "Average per order", icon_class="fas fa-file-invoice-dollar"), lg=3, md=6, sm=12),
        dbc.Col(create_styled_kpi_tile("Avg. Satisfaction", f"{avg_satisfaction:.1f} / 5", f"{num_rows} responses", icon_class="fas fa-smile"), lg=3, md=6, sm=12),
    ], className="mb-3"),

    # Row for Main Charts (2 charts side-by-side)
    dbc.Row([
        dbc.Col(
            dbc.Card([dbc.CardBody(dcc.Graph(figure=fig_category_sales, config={'displayModeBar': False}))], className="shadow-sm border-0 m-2"),
            lg=7, md=12
        ),
        dbc.Col(
            dbc.Card([dbc.CardBody(dcc.Graph(figure=fig_region_sales, config={'displayModeBar': False}))], className="shadow-sm border-0 m-2"),
            lg=5, md=12
        ),
    ], className="mb-3"),

    # Row for Time Series and another chart
    dbc.Row([
        dbc.Col(
            dbc.Card([dbc.CardBody(dcc.Graph(figure=fig_monthly_sales, config={'displayModeBar': False}))], className="shadow-sm border-0 m-2"),
            lg=8, md=12
        ),
        dbc.Col(
            dbc.Card([dbc.CardBody(dcc.Graph(figure=fig_satisfaction, config={'displayModeBar': False}))], className="shadow-sm border-0 m-2"),
            lg=4, md=12
        ),
    ], className="mb-3"),

    html.Footer(
        "Fictional Company Sales Data © 2025 | Confidential",
        className="text-center text-muted mt-4 p-3", style={'fontSize': '0.8em'}
    )
])

if __name__ == '__main__':
    app.run(debug=True) # Use app.run instead of app.run_server