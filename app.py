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
    #Faculty Explorder Tab
    # Add to existing tabs list after Publication Explorer
dcc.Tab(label='Faculty Explorer', children=[
    html.Div([
        html.Div([
            html.Label("Search Faculty Expertise:", className='control-label'),
            dcc.Input(
                id='faculty-keyword-input',
                type='text',
                placeholder='Enter keyword...',
                className='keyword-input'
            )
        ], className='search-column'),
        
        html.Div([
            html.Label("Faculty to Display:", className='control-label'),
            dcc.Slider(
                id='faculty-count-slider',
                min=5,
                max=50,
                step=5,
                value=15,
                marks={i: str(i) for i in range(5, 55, 10)}
            )
        ], className='control-column')
    ], className='control-panel'),
    
    dcc.Graph(id='faculty-analysis-chart'),
    html.Div(id='faculty-details', className='stats-panel')
]),

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
#        print(pd.DataFrame(result))
        return pd.DataFrame(result) if result else pd.DataFrame()

    except Exception as e:
        print(f"Database Error: {str(e)}")
        return pd.DataFrame()
        
    finally:
        if mysql and hasattr(mysql, 'connection'):
            mysql.close()
# Faculty Analysis Functions
def get_top_faculty(search_term, top_n):
    """Retrieve faculty with strongest keyword associations for specific keyword"""
    query = """
    SELECT 
        f.name,
        f.position,
	f.photo_url,
        u.name AS university,
        MAX(fk.score) AS keyword_score,  # Changed to MAX
        k.name AS target_keyword,  # Explicitly show matched keyword
        GROUP_CONCAT(DISTINCT k_all.name) AS related_keywords
    FROM faculty f
    JOIN faculty_keyword fk ON f.id = fk.faculty_id
    JOIN keyword k ON fk.keyword_id = k.id
    JOIN university u ON f.university_id = u.id
    LEFT JOIN faculty_keyword fk_all ON f.id = fk_all.faculty_id
    LEFT JOIN keyword k_all ON fk_all.keyword_id = k_all.id
    WHERE k.name LIKE %s
    GROUP BY f.id, u.name, k.name
    ORDER BY keyword_score DESC
    LIMIT %s;
    """
    search_pattern = f"%{search_term}%"
    return execute_query(query, (search_pattern, top_n))

def create_faculty_card(faculty_row):
    """Generate detailed faculty profile card"""
    return dbc.Card(
        [
            dbc.Row([
                dbc.Col(
                    html.Img(
                        src=faculty_row['photo_url'],
                        className='faculty-photo',
                        style={'height':'150px', 'objectFit':'cover'}
                    ), 
                    width=3
                ),
                dbc.Col([
                    html.H4(faculty_row['name'], className='card-title'),
                    html.P(f"{faculty_row['position']} @ {faculty_row['university']}", 
                          className='text-muted'),
                    html.P(f"Keyword Score: {faculty_row['keyword_score']:.2f}"),
                    html.Hr()
                ], width=9)
            ])
        ],
        className='mb-3 shadow-sm'
    )

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

# Callbacks for Faculty Analysis Tab
@app.callback(
    [Output('faculty-analysis-chart', 'figure'),
     Output('faculty-details', 'children')],
    [Input('faculty-keyword-input', 'value'),
     Input('faculty-count-slider', 'value')]
)
def update_faculty_analysis(search_term, top_n):
    if not search_term:
        return px.scatter(title="Enter research domain to begin analysis"), ""
    
    df = get_top_faculty(search_term, top_n)
    if df.empty:
        return px.scatter(title=f"No faculty found for '{search_term}'"), ""
    
    # Visualization with enhanced styling
    fig = px.bar(
        df,
        x='keyword_score',
        y='name',
        color='university',
        hover_data=['university', 'position', 'related_keywords'],
        labels={
            'keyword_score': 'Keyword Score',
            'name': 'Faculty Member',
            'keyword_count': 'Related Keywords',
            'position': 'Academic Position'
        },
        title=f"Top {top_n} Faculty in '{search_term.title()}' Domain"
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        hoverlabel=dict(bgcolor="white", font_size=12),
        coloraxis_colorbar=dict(title="Keyword Count")
    )

    # Interactive detail cards
    cards = [create_faculty_card(row) for _, row in df.iterrows()]
    
    return fig, cards

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
