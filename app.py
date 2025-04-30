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
    'database': 'MP3'
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
                        multi=True
                    ),
                    dcc.Dropdown(
                        id='professor-filter',
                        placeholder="Professor...",
                        multi=True
                    ),
                    dcc.Dropdown(
                        id='publication-filter',
                        placeholder="Publication...",
                        multi=True
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
        return pd.DataFrame(result) if result else pd.DataFrame()

    except Exception as e:
        print(f"Database Error: {str(e)}")
        return pd.DataFrame()
        
    finally:
        if mysql and hasattr(mysql, 'connection'):
            mysql.close()

# Keyword Analysis Functions
def get_filtered_keywords(filters):
    """Dynamic query builder for keyword analysis"""
    """Dynamic query builder based on active filters"""
    base_query = """
        SELECT k.name, {metric} AS value
        FROM keywords k
        {joins}
        WHERE {conditions}
        GROUP BY k.name
        ORDER BY value DESC
        LIMIT %s
    """
    
    metric = "SUM(k.score)"  # Default metric for faculty-related filters
    joins = []
    conditions = []
    params = []
    limit = filters.get('limit', 15)

    if filters.get('publications'):
        metric = "COUNT(*)"
        joins.append("JOIN publication_keywords pk ON k.id = pk.keyword_id")
        joins.append("JOIN publications p ON pk.publication_id = p.id")
        conditions.append("p.title IN %s")
        params.append(tuple(filters['publications']))
        
    elif filters.get('professors'):
        joins.append("JOIN faculty_keywords fk ON k.id = fk.keyword_id")
        joins.append("JOIN faculty f ON fk.faculty_id = f.id")
        conditions.append("f.name IN %s")
        params.append(tuple(filters['professors']))
        
    elif filters.get('universities'):
        joins.append("JOIN faculty_keywords fk ON k.id = fk.keyword_id")
        joins.append("JOIN faculty f ON fk.faculty_id = f.id")
        joins.append("JOIN affiliation a ON f.affiliation = a.id")
        conditions.append("a.name IN %s")
        params.append(tuple(filters['universities']))
        
    else:  # Default scope-based query
        if filters['scope'] == 'faculty':
            joins.append("JOIN faculty_keywords fk ON k.id = fk.keyword_id")
        elif filters['scope'] == 'publications':
            joins.append("JOIN publication_keywords pk ON k.id = pk.keyword_id")
        else:
            return execute_query("""
                (SELECT k.name, COUNT(*) AS value 
                 FROM faculty_keywords fk JOIN keywords k ON fk.keyword_id = k.id 
                 GROUP BY k.name)
                UNION ALL
                (SELECT k.name, COUNT(*) AS value 
                 FROM publication_keywords pk JOIN keywords k ON pk.keyword_id = k.id 
                 GROUP BY k.name)
                GROUP BY name 
                ORDER BY SUM(value) DESC 
                LIMIT %s
            """, (limit,))
    
    query = base_query.format(
        metric=metric,
        joins="\n".join(joins),
        conditions=" AND ".join(conditions) if conditions else "1=1"
    )
    params.append(limit)
    
    return execute_query(query, tuple(params))

# Publication Analysis Functions
def get_top_publications(keyword, top_n):
    """Retrieve publications with highest keyword relevance"""
    query = """
        SELECT p.title, 
               SUM(k.score) AS score,
               p.year,
               GROUP_CONCAT(DISTINCT f.name) AS authors,
               COUNT(DISTINCT c.id) AS citations
        FROM publications p
        JOIN publication_keywords pk ON p.id = pk.publication_id
        JOIN keywords k ON pk.keyword_id = k.id
        LEFT JOIN faculty_publications fp ON p.id = fp.publication_id
        LEFT JOIN faculty f ON fp.faculty_id = f.id
        LEFT JOIN citations c ON p.id = c.publication_id
        WHERE k.name = %s
        GROUP BY p.title, p.year
        ORDER BY score DESC
        LIMIT %s
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
    return (
        [{'label': uni, 'value': uni} 
         for uni in execute_query("SELECT name FROM affiliation").name],
        [{'label': prof, 'value': prof} 
         for prof in execute_query("SELECT name FROM faculty").name],
        [{'label': pub, 'value': pub} 
         for pub in execute_query("SELECT title FROM publications").title]
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
    metric = 'Score' if any([universities, professors]) else 'Frequency'
    
    # Visualization
    fig = px.bar(
        df,
        x='value',
        y='name',
        orientation='h',
        color='value',
        color_continuous_scale='Teal',
        labels={'name': 'Keyword', 'value': metric},
        title=f'Keyword {metric} Distribution'
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        plot_bgcolor='rgba(240,240,240,0.9)',
        hoverlabel={'bgcolor': 'white'}
    )

    # Statistics
    stats = [
        html.H4("Current Context:", style={'color': '#2c3e50'}),
        html.P(f"Active Filters: {get_active_filters(filters)}"),
        html.P(f"Total Keywords: {len(df)}"),
        html.P(f"Average {metric}: {df['value'].mean():.1f}"),
        html.P(f"Maximum {metric}: {df['value'].max()} ({df.iloc[0]['name']})")
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
    
    # Visualization
    fig = px.bar(
        df,
        x='score',
        y='title',
        color='year',
        hover_data=['authors', 'citations'],
        labels={
            'score': 'Relevance Score',
            'title': 'Publication',
            'year': 'Publication Year',
            'authors': 'Authors',
            'citations': 'Citations'
        },
        title=f"Top {top_n} Publications for '{keyword.title()}'"
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})

    # Metadata
    stats = [
        html.H4("Publication Insights:"),
        html.P(f"Average Citations: {df['citations'].mean():.1f}"),
        html.P(f"Most Recent Publication: {df['year'].max()}"),
        html.P(f"Authors with Most Publications: {df['authors'].explode().mode()[0]}")
    ]
    
    return fig, stats

# Style Configuration
app.css.append_css({
    'external_url': [
        'https://codepen.io/chriddyp/pen/bWLwgP.css',
        '/assets/custom.css'
    ]
})

if __name__ == '__main__':
    app.run(debug=True, port=8050)
