# app.py - Academic Research Analytics Dashboard
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from mysql_utils import MySQLUtils

app = dash.Dash(__name__)
server = app.server

# Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'newuser',
    'password': 'password',
    'db': 'academicworld'
}

app.layout = html.Div([
    html.H1("Academic Research Analytics Dashboard", className='main-header'),
    
    dcc.Tabs([
    #Publication Explorer Tab
    dcc.Tab(label='Publication Explorer', children=[
        html.Div([
            html.Div([
                html.Label("Keyword Search:", className='control-label'),
                dcc.Input(
                    id='keyword-input',
                    type='text',
                    placeholder='Enter research keyword...',
                    className='keyword-input'
                )
            ], className='search-column'),
            
            html.Div([
                html.Label("Publications to Show:", className='control-label'),
                dcc.Slider(
                    id='pub-count-slider',
                    min=5,
                    max=100,
                    step=5,
                    value=25,
                    marks={i: str(i) for i in range(5, 105, 20)}
                )
            ], className='control-column')
        ], className='control-panel'),
        
        dcc.Graph(id='publication-scores-chart'),
        html.Div(id='publication-meta', className='stats-panel')
    ])
  ])
], style={'fontFamily': 'Arial, sans-serif'})

# Shared Database function
def execute_query(query, params=None):
    """Universal query executor with enhanced error handling"""
    mysql = None  # Initialize outside try block to ensure variable existence
    try:
        mysql = MySQLUtils(**DB_CONFIG)
        if not mysql.connect():
            raise ConnectionError("Failed to connect to database")
            
        result = mysql.execute_query(query, params)
        print(pd.DataFrame(result))
        return pd.DataFrame(result) if result else pd.DataFrame()

    except Exception as e:
        print(f"Database Error: {str(e)}")
        return pd.DataFrame()
        
    finally:
        if mysql and hasattr(mysql, 'connection'):
            mysql.close()

# Publication Analysis Functions
def get_top_publications(keyword, top_n):
    """Retrieve publications with highest keyword relevance"""
    query = """
	SELECT p.title,
        pk.score AS keyword_score,  -- Direct score from junction table
        p.num_citations,
        p.year,
        GROUP_CONCAT(DISTINCT f.name) AS authors
	FROM publication p
	JOIN Publication_Keyword pk ON p.id = pk.publication_id
	JOIN keyword k ON pk.keyword_id = k.id
	LEFT JOIN faculty_publication fp ON p.id = fp.publication_id
	LEFT JOIN faculty f ON fp.faculty_id = f.id
	WHERE k.name = %s
	GROUP BY p.id, pk.score, p.title, p.year, p.num_citations  -- Added pk.score to GROUP BY
	ORDER BY keyword_score DESC
	LIMIT %s;
    """
    return execute_query(query, (keyword, top_n))

# Callbacks for Publication Analysis Tab
@app.callback(
    [Output('publication-scores-chart', 'figure'),
     Output('publication-meta', 'children')],
    [Input('keyword-input', 'value'),
     Input('pub-count-slider', 'value')]
)
def update_publication_analysis(keyword, top_n):
    if not keyword:
        return px.scatter(title="Enter a keyword to begin analysis"), ""
    
    df = get_top_publications(keyword, top_n)
    if df.empty:
        return px.scatter(title=f"No publications found for '{keyword}'"), ""
    
    # Visualization updates
    fig = px.bar(
        df,
        x='keyword_score',
        y='title',
        color='year',
        hover_data=['authors', 'num_citations'],  # Changed 'citations' → 'num_citations'
        labels={
            'keyword_score': 'Keyword Relevance Score',
            'title': 'Publication Title',
            'year': 'Publication Year',
            'authors': 'Authors',
            'num_citations': 'Citations'  # Label mapping update
        },
        title=f"Top {top_n} Publications for '{keyword.title()}'"
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        hovermode='closest'
    )

    # Metadata updates
    stats = [
        html.H4("Analysis Insights:"),
        html.P(f"Median Keyword Score: {df['keyword_score'].median():.1f}"),
        html.P(f"Total Citations Across Selection: {df['num_citations'].sum()}"),  # Field name update
        html.P(f"Publication Span: {df['year'].min()} - {df['year'].max()}"),
        html.P(f"Most Frequent Author: {df['authors'].str.split(', ').explode().mode()[0]}")  # Improved author parsing
    ]
    
    return fig, stats



# Style Configuration
#app.css.append_css({
#    'external_url': [
#        'https://codepen.io/chriddyp/pen/bWLwgP.css',
#        '/assets/custom.css'
#    ]
#})

if __name__ == '__main__':
    app.run(debug=True, dev_tools_hot_reload = False, port=8050)
