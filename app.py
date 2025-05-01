# app.py - Academic Research Analytics Dashboard
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
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
        # Keyword Analysis Tab
        dcc.Tab(label='Keyword Insights', children=[
            html.Div([
                html.Div([
                    html.Label("Analysis Scope:", className='control-label'),
                    dcc.Dropdown(
                        id='scope-selector',
                        options=[
                            {'label': 'Faculty Keywords', 'value': 'faculty'},
                            {'label': 'Publication Keywords', 'value': 'publications'},
                            {'label': 'Combined Analysis', 'value': 'combined'}
                        ],
                        value='combined',
                        clearable=False
                    )
                ], className='control-column'),
                
                html.Div([
                    html.Label("Results Limit:", className='control-label'),
                    dcc.Slider(
                        id='top-n-selector',
                        min=5,
                        max=50,
                        step=5,
                        value=15,
                        marks={i: str(i) for i in range(5, 55, 5)}
                    )
                ], className='control-column'),
                
                html.Div([
                    html.Label("Filters:", className='control-label'),
                    dcc.Dropdown(
                        id='university-filter',
                        placeholder="University...",
                        multi=True,
			options=[],
			value=[]
                    ),
                    dcc.Dropdown(
                        id='professor-filter',
                        placeholder="Professor...",
                        multi=True,
			options=[],
			value=[]
                    ),
                    dcc.Dropdown(
                        id='publication-filter',
                        placeholder="Publication...",
                        multi=True,
			options=[],
			value=[]
                    )
                ], className='filter-column')
            ], className='control-panel'),
            
            dcc.Graph(id='keyword-visualization'),
            html.Div(id='context-stats', className='stats-panel')
        ]),
        
        # Publication Analysis Tab
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

# Keyword Analysis Functions
def get_filtered_keywords(filters):
    """Dynamic score aggregation based on entity filters"""
    base_query = """
        SELECT k.name, {metric} AS value
        FROM keyword k
        {joins}
        WHERE {conditions}
        GROUP BY k.name
        ORDER BY value DESC
        LIMIT %s
    """
    
    metric = "COUNT(*)"  # Default faculty metric
    joins = []
    conditions = ["1=1"]  # Default condition
    params = []
    limit = filters.get('limit', 15)

    # Entity-specific score aggregation
    if filters.get('publications'):
        metric = "SUM(pk.score)"
        joins.extend([
            "JOIN Publication_Keyword pk ON k.id = pk.keyword_id",
            "JOIN publication p ON pk.publication_id = p.id"
        ])
        conditions.append("p.title IN %s")
        params.append(tuple(filters['publications']))
        
    elif filters.get('professors'):
        metric = "SUM(fk.score)"
        joins.extend([
            "JOIN faculty_keyword fk ON k.id = fk.keyword_id",
            "JOIN faculty f ON fk.faculty_id = f.id"
        ])
        conditions.append("f.name IN %s")
        params.append(tuple(filters['professors']))
        
    elif filters.get('universities'):
        joins.extend([
            "JOIN faculty_keyword fk ON k.id = fk.keyword_id",
            "JOIN faculty f ON fk.faculty_id = f.id",
            "JOIN university u ON f.university = u.id"
        ])
        conditions.append("u.name IN %s")
        params.append(tuple(filters['universities']))
        
    else:  # Cross-entity aggregation
        if filters['scope'] == 'faculty':
            joins.append("JOIN faculty_keyword fk ON k.id = fk.keyword_id")
            metric = "SUM(fk.score)"
        elif filters['scope'] == 'publication':
            joins.append("JOIN Publication_Keyword pk ON k.id = pk.keyword_id")
            metric = "SUM(pk.score)"
        else:  # Global aggregation
            return execute_query("""
                SELECT name, SUM(score) AS value FROM (
                    SELECT k.name, fk.score 
                    FROM faculty_keyword fk
                    JOIN keyword k ON fk.keyword_id = k.id
                    UNION ALL
                    SELECT k.name, pk.score
                    FROM Publication_Keyword pk
                    JOIN keyword k ON pk.keyword_id = k.id
                ) AS combined
                GROUP BY name
                ORDER BY value DESC
                LIMIT %s
            """, (limit,))
    
    query = base_query.format(
        metric=metric,
        joins="\n".join(joins),
        conditions=" AND ".join(conditions)
    )
    return execute_query(query, tuple(params) + (limit,))

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

# Callbacks for Keyword Analysis Tab
@app.callback(
    [Output('university-filter', 'options'),
     Output('professor-filter', 'options'),
     Output('publication-filter', 'options')],
    Input('scope-selector', 'value')
)

def populate_filters(_):
    # Add error handling and case normalization
    def safe_query(query, col_name='name'):
        try:
            df = execute_query(query)
            if not df.empty:
                return [{'label': v, 'value': v} for v in df[col_name].str.strip().str.title()]
            return []
        except Exception as e:
            print(f"Filter population error: {str(e)}")
            return []

    return (
        safe_query("SELECT name FROM university", 'name'),
        safe_query("SELECT name FROM faculty", 'name'),
        safe_query("SELECT title FROM publication", 'title')
    )

@app.callback(
    [Output('keyword-visualization', 'figure'),
     Output('context-stats', 'children')],
    [Input('scope-selector', 'value'),
     Input('top-n-selector', 'value'),
     Input('university-filter', 'value'),
     Input('professor-filter', 'value'),
     Input('publication-filter', 'value')]
)

def update_keyword_analysis(scope, limit, universities, professors, publications):
    filters = {
        'scope': scope,
        'limit': limit,
        'universities': universities,
        'professors': professors,
        'publications': publications
    }
    
    df = get_filtered_keywords(filters)
    
    fig = px.treemap(
        df,
        path=['name'],
        values='value',
        color='value',
        color_continuous_scale='Teal',
        title='Keyword Impact Distribution',
        labels={'value': 'Aggregated Score', 'name': 'Keyword'}
    )
    
    stats = [
        html.H5("Aggregation Context:", className="stats-header"),
        html.P(f"Total Keywords: {len(df):,}", className="stat-item"),
        html.P(f"Score Range: {df['value'].min():.1f} - {df['value'].max():.1f}", className="stat-item"),
        html.P(f"75th Percentile Score: {df['value'].quantile(0.75):.1f}", className="stat-item"),
        html.Div([
            html.Span("Dominant Entities:", className="stat-label"),
            html.Ul([
                html.Li(f"Publications: {len(filters['publications'])}" if filters['publications'] else "All Publications"),
                html.Li(f"Professors: {len(filters['professors'])}" if filters['professors'] else "All Faculty"),
                html.Li(f"Universities: {len(filters['universities'])}" if filters['universities'] else "All Institutions")
            ], className="entity-list")
        ], className="stat-item")
    ]
    
    return fig, stats


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
