# app.py - Academic Research Analytics Dashboard (Modified Faculty Explorer)
import dash
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
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
        # Modified Faculty Explorer Tab
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
                    html.Label("Results Limit:", className='control-label'),
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
            
            # List and Details Container
            html.Div([
                html.Div(id='faculty-list-container', className='scrollable-list'),
                html.Div(id='faculty-details-container', className='details-panel')
            ], className='list-detail-container'),
            
            dcc.Store(id='faculty-cache-store')
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
    ]),
    # New University Analysis Tab
    dcc.Tab(label='University Analysis', children=[
        html.Div([
            html.H3("University Research Strength Analyzer", className='tab-header'),
            # Photo Card (new)
            html.Div(
                id='university-photo-card',
                className='photo-card',
                children=[
                    html.Img(
                        id='university-photo',
                        className='university-photo',
                        style={'display': 'none'}  # Hidden by default
                    )
                ]
            ),
	    # Control Panel
            html.Div([
                html.Div([
                    html.Label("University Name:", className='control-label'),
                    dcc.Input(
                        id='university-tab-input',
                        type='text',
                        placeholder='Start typing university name...',
                        className='university-input',
                        debounce=True
                    )
                ], className='search-column'),
                
                html.Div([
                    html.Label("Top Keywords to Display:", className='control-label'),
                    dcc.Slider(
                        id='university-tab-top-n',
                        min=5,
                        max=25,
                        step=1,
                        value=15,
                        marks={i: {'label': str(i)} for i in range(5, 26, 5)},
                        tooltip={"placement": "bottom"}
                    )
                ], className='control-column')
            ], className='control-panel'),
            
            # Visualization Area
            dcc.Graph(
                id='university-tab-pie-chart',
                config={'displayModeBar': False},
                className='pie-container'
            ),
            
            # Supplemental Info
            html.Div(id='university-stats-panel', className='stats-panel')
        ], className='university-tab-content')
    ])

        ])
], style={'fontFamily': 'Arial, sans-serif'})

# Shared Database function
def execute_query(query, params=None):
    #print(query)
    """Universal query executor with enhanced error handling"""
    mysql = None  # Initialize outside try block to ensure variable existence
    try:
        mysql = MySQLUtils(**DB_CONFIG)
        if not mysql.connect():
            raise ConnectionError("Failed to connect to database")
            
        result = mysql.execute_query(query, params)
        #print(pd.DataFrame(result))
        return pd.DataFrame(result) if result else pd.DataFrame()

    except Exception as e:
        print(f"Database Error: {str(e)}")
        return pd.DataFrame()
        
    finally:
        if mysql and hasattr(mysql, 'connection'):
            mysql.close()


# Faculty Analysis Functions (modified)
def get_faculty_list(search_term, limit):
    """Retrieve simplified faculty list with basic info"""
    query = """
    SELECT 
        f.name,
        f.position,
        u.name AS university,
        MAX(fk.score) AS keyword_score
    FROM faculty f
    JOIN faculty_keyword fk ON f.id = fk.faculty_id
    JOIN keyword k ON fk.keyword_id = k.id
    JOIN university u ON f.university_id = u.id
    WHERE k.name LIKE %s
    GROUP BY f.id, u.name
    ORDER BY keyword_score DESC
    LIMIT %s;
    """
    search_pattern = f"%{search_term}%"
    return execute_query(query, (search_pattern, limit))

def get_full_faculty_details(name):
    """Retrieve complete details for selected faculty"""
    query = """
    SELECT 
        f.name,
        f.position,
        f.photo_url,
        u.name AS university,
        MAX(fk.score) AS keyword_score,
        k.name AS target_keyword,
        GROUP_CONCAT(DISTINCT k_all.name) AS related_keywords
    FROM faculty f
    JOIN faculty_keyword fk ON f.id = fk.faculty_id
    JOIN keyword k ON fk.keyword_id = k.id
    JOIN university u ON f.university_id = u.id
    LEFT JOIN faculty_keyword fk_all ON f.id = fk_all.faculty_id
    LEFT JOIN keyword k_all ON fk_all.keyword_id = k_all.id
    WHERE f.name = %s
    GROUP BY f.id, u.name, k.name;
    """
    return execute_query(query, (name,)).iloc[0]

# Modified Faculty Callbacks
@app.callback(
    [Output('faculty-list-container', 'children'),
     Output('faculty-cache-store', 'data')],
    [Input('faculty-keyword-input', 'value'),
     Input('faculty-count-slider', 'value')]
)
def update_faculty_list(search_input, limit):
    if not search_input:
        return html.Div("Enter a research keyword to begin search"), None
    
    df = get_faculty_list(search_input, limit)
    if df.empty:
        return html.Div(f"No results found for '{search_input}'"), None
    
    # Generate clickable list items
    list_items = [
        html.Li(
            [
                html.Div(
                    [
                        html.Strong(row['name']),
                        html.Br(),
                        html.Span(f"{row['position']} @ {row['university']}", 
                                className='text-muted'),
                        html.Div(f"Expertise Score: {row['keyword_score']:.2f}",
                               className='score-indicator')
                    ],
                    className='list-item-content'
                )
            ],
            className='faculty-list-item',
            id={'type': 'faculty-name', 'index': row['name']},
            n_clicks=0
        ) for _, row in df.iterrows()
    ]
    
    return html.Ul(list_items, className='faculty-list'), df.to_dict('records')

@app.callback(
    Output('faculty-details-container', 'children'),
    [Input({'type': 'faculty-name', 'index': dash.dependencies.ALL}, 'n_clicks')],
    [State('faculty-cache-store', 'data')]
)
def show_faculty_details(clicks, cached_data):
    if not cached_data or not any(clicks):
        return html.Div("Select a faculty member from the list to view details",
                       className='placeholder-text')
    
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    selected_name = eval(triggered_id)['index']
    
    try:
        full_details = get_full_faculty_details(selected_name)
        return create_faculty_card(full_details)
    except Exception as e:
        print(f"Detail retrieval error: {str(e)}")
        return html.Div("Error loading faculty details", className='error-message')

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
    
# security-enhanced database function
def get_university_keyword_scores(university_name, top_n):
    """Get aggregated keyword scores for a university's faculty"""
    query = """
    SELECT
        k.name AS keyword,
        SUM(fk.score) as total_score,
        COUNT(DISTINCT f.id) as professor_count,
        GROUP_CONCAT(DISTINCT f.name) as professors,
        MAX(u.photo_url) as photo_url
    FROM faculty f
    JOIN faculty_keyword fk ON f.id = fk.faculty_id
    JOIN keyword k ON fk.keyword_id = k.id
    JOIN university u ON u.id = f.university_id
    WHERE u.name LIKE %s
    GROUP BY k.name
    ORDER BY total_score DESC
    LIMIT %s;
    """
    
    return execute_query(query, (f"%{university_name.strip()}%", top_n))

# accessibility-enhanced visualization
def create_keyword_pie(df, university_name):
    if df.empty:
        return px.pie(title="No Data Available").update_layout(
            annotations=[dict(text="No research data found", showarrow=False)]
        )

    fig = px.pie(
        df,
        names='keyword',
        values='total_score',
        hover_data=['professor_count', 'professors'],
        title=f"Research Keywords Distribution at {university_name}",
        hole=0.35,
        labels={
            'keyword': 'Research Area',
            'total_score': 'Aggregated Score',
            'professor_count': 'Professors'
        },
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    fig.update_traces(
        texttemplate='%{label}<br>%{percent}',
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Total Score: %{value}<br>"
            "Professors: %{customdata[0]}<br>"
            "Contributors: %{customdata[1]}<br>"
            "<extra></extra>"
        ),
        marker=dict(line=dict(color='#ffffff', width=1))
    )

    fig.update_layout(
        uniformtext_minsize=12,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=14
        )
    )
    
    return fig

# secured callback with validation
@app.callback(
    [Output('university-tab-pie-chart', 'figure'),
     Output('university-stats-panel', 'children'),
     Output('university-photo', 'src'),
     Output('university-photo', 'style')],
    [Input('university-tab-input', 'value'),
     Input('university-tab-top-n', 'value')]
)

def update_university_tab(name_input, top_n=10):
    # Input validation
    if not name_input or len(name_input.strip()) < 3:
        raise PreventUpdate
    
    try:
        top_n = int(top_n)
        top_n = max(1, min(top_n, 50))  # Enforce 1-50 range
    except (TypeError, ValueError):
        top_n = 10

    clean_name = name_input.strip().title()
    df = get_university_keyword_scores(clean_name, top_n)

    if df.empty:
        return (
            px.pie(title=f"No Data Found").update_layout(
                annotations=[dict(text=f"No results for '{clean_name}'", showarrow=False)]
            ),
            html.Div(
                "Please verify the university name",
                className='error-message',
                style={'color': '#dc3545'}
            )
        )
    # Extract photo URL from first result
    photo_url = df.iloc[0]['photo_url'] if 'photo_url' in df.columns else None
    # Generate visualization
    pie_fig = create_keyword_pie(df, clean_name)
    
    # Create sanitized stats panel
    stats_content = html.Div([
        html.H4(f"{clean_name} Research Summary", className='summary-title'),
        html.Div([
            html.Div([
                html.Span("Total Keywords Analyzed:", className='stat-label'),
                html.Span(f"{len(df)}", className='stat-value')
            ], className='stat-item'),
            html.Div([
                html.Span("Average Score:", className='stat-label'),
                html.Span(f"{df['total_score'].mean():.1f}", className='stat-value')
            ], className='stat-item'),
            html.Div([
                html.Span("Top Research Area:", className='stat-label'),
                html.Span(df.iloc[0]['keyword'], className='stat-value')
            ], className='stat-item')
        ], className='stats-grid')
    ])
    
    # Show/hide photo based on availability
    photo_style = {'height': '200px', 'objectFit': 'cover'} if photo_url else {'display': 'none'}
    
    return pie_fig, stats_content, photo_url, photo_style

if __name__ == '__main__':
    app.run(debug=True, dev_tools_hot_reload=False, port=8050)


