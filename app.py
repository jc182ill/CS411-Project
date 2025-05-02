# app.py - Academic Research Analytics Dashboard (Modified Faculty Explorer)
import dash
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
from datetime import datetime
from bson import ObjectId
from collections import Counter
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
import json
import re
from neo4j import GraphDatabase
from mysql_utils import MySQLUtils
from mongodb_utils import MongoDBUtils
from neo4j_utils import Neo4jUtils

app = dash.Dash(__name__)
server = app.server

# Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'newuser',
    'password': 'password',
    'db': 'academicworld'
}

MONGO_URI = "mongodb://localhost:27017"

NEO4J_CONFIG = {
     'database': 'academicworld',
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
                    id='publication-keyword-input',
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
        html.Div(id='publication-meta', className='stats-panel'),
        html.Div(id='publication-detail-card', className='detail-panel')
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
    ]),
    dcc.Tab(label='Related Keywords', children=[
	html.Div([
	    html.H4("Semantic Keyword Explorer", className='widget-header'),
	    html.Div([
		dcc.Input(
		    id='keyword-input',
		    type='text',
		    placeholder='Enter research keyword...',
		    debounce=True,  # Prevents rapid firing
		    style={'width': '100%', 'padding': '10px'}
		),
		html.Div([
		    html.Label("Minimum Connections:", className='control-label'),
		    dcc.Slider(
		        id='connection-threshold',
		        min=2,
		        max=20,
		        step=2,
		        value=3,
		        marks={i: str(i) for i in range(2, 21)},
		        tooltip={"placement": "bottom"}
		    )
		], className='threshold-control')
	    ], className='control-bar'),
	    dcc.Graph(
		id='keyword-network',
		config={'displayModeBar': False},
		className='network-graph'
	    )
	], className='similarity-widget')
    ]),
    dcc.Tab(label='Admin Editor', children=[
    html.Div([
        html.H3("Data Override & Approval Panel"),

        dcc.Dropdown(
            id='admin-entity-type',
            options=[
                {'label': 'Faculty', 'value': 'faculty'},
                {'label': 'Publication', 'value': 'publication'},
                {'label': 'University', 'value': 'university'}
            ],
            placeholder="Select entity type"
        ),

        dcc.Input(id='admin-entity-id', type='number', placeholder='Enter entity ID'),

        dcc.Textarea(
            id='admin-json-editor',
            placeholder='Enter field-value JSON (e.g., {"email": "new@edu.edu"})',
            style={'width': '100%', 'height': '200px'}
        ),
	html.H4("Current Values"),
	html.Pre(id='admin-current-values', style={'background': '#f7f7f7', 'padding': '10px'}),
        html.Button("Submit Override", id='admin-submit-btn'),
        html.Button("Delete Override", id='admin-delete-btn'),
        html.Button("Approve Override", id='admin-approve-btn'),
        html.Div(id='admin-action-status', style={'marginTop': '10px'}),
        html.Hr(),
	html.H4("Override History"),
	html.Div(id='admin-history-list'),
        html.H4("Pending Approvals"),
        html.Div(id='admin-pending-list')
    ])
]),
	dcc.Tab(label='Keyword Matcher', children=[
	    html.Div([
		html.H3("Keyword Relevance Matcher"),
		dcc.Input(id='kw-match-input', type='text', placeholder='Enter comma-separated keywords...'),
		dcc.Slider(id='kw-match-limit', min=5, max=50, value=10, step=5, marks={i: str(i) for i in range(5, 55, 5)}),
		html.Button("Find Matches", id='kw-match-btn'),
		html.Div(id='kw-match-status', style={'marginTop': '10px'}),
		html.H4("Top Professors"),
		html.Div(id='kw-match-faculty'),
		html.H4("Top Publications"),
		html.Div(id='kw-match-publications'),
		html.Hr(),
                html.Div(id='kw-match-faculty-card', className='details-panel')
	    ])
	])
    ])
], style={'fontFamily': 'Arial, sans-serif'})

# Shared Database function
def execute_query(query, params=None, db_type='mysql'):
    #print(query)
    #print(params)
    """Universal executor supporting both MySQL and Neo4j"""
    if db_type == 'mysql':
        # Existing MySQL logic
        mysql = None
        try:
            mysql = MySQLUtils(**DB_CONFIG)
            if not mysql.connect():
                raise ConnectionError("MySQL connection failed")
            result = mysql.execute_query(query, params)
            #print(pd.DataFrame(result)
            return pd.DataFrame(result) if result else pd.DataFrame()
            
        except Exception as e:
            print(f"MySQL Error: {str(e)}")
            return pd.DataFrame()
            
        finally:
            if mysql and hasattr(mysql, 'connection'):
                mysql.close()
                
    elif db_type == 'neo4j':
        # Neo4j execution path
        try:
            n4j = Neo4jUtils(**NEO4J_CONFIG)
            if not n4j.connect():
            	raise ConnectionError("Neo4j connection failed")
            records = n4j.execute_query(query, params)
            # Convert Neo4j records to DataFrame
            data = [dict(rec) for rec in records]
            print(pd.DataFrame(data))
            return pd.DataFrame(data)
                
        except Exception as e:
            print(f"Neo4j Error: {str(e)}")
            return pd.DataFrame()
            
    else:
        raise ValueError("Invalid database type. Use 'mysql' or 'neo4j'")

# Faculty Analysis Functions (modified)
def get_faculty_list(search_term, limit):
    """Retrieve simplified faculty list with basic info"""
    query = """
    SELECT
        f.id, 
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
    """Retrieve full faculty profile with top keywords by score"""
    profile_query = """
    SELECT
        f.id, 
        f.name,
        f.position,
        f.photo_url,
        f.email,
        f.phone,
        f.research_interest,
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
    WHERE f.id = %s
    GROUP BY f.id, u.name, k.name;

    """

    keywords_query = """
    SELECT k.name, fk.score
    FROM faculty_keyword fk
    JOIN keyword k ON fk.keyword_id = k.id
    WHERE fk.faculty_id = %s
    ORDER BY fk.score DESC
    LIMIT 10;
    """

    publications_query = """
    SELECT p.title, p.venue, p.year, p.num_citations
    FROM faculty_publication fp
    JOIN publication p ON fp.publication_id = p.id
    WHERE fp.faculty_id = %s
    ORDER BY p.year DESC
    LIMIT 5;
    """

    profile_df = execute_query(profile_query, (name,))
    keywords_df = execute_query(keywords_query, (name,))
    pubs_df = execute_query(publications_query, (name,))

    if profile_df.empty:
        raise ValueError("Faculty not found.")

    profile = profile_df.iloc[0].to_dict()
    profile['top_keywords'] = keywords_df.to_dict('records') if not keywords_df.empty else []
    profile["matched_publications"] = pubs_df.to_dict('records') if pubs_df is not None and not pubs_df.empty else []
    return profile

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
                        html.Div(f"Keyword Score: {row['keyword_score']}",
                               className='score-indicator')
                    ],
                    className='list-item-content'
                )
            ],
            className='faculty-list-item',
            id={'type': 'faculty-name', 'index': row['id']},
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
    """Generate detailed faculty profile card with contact info, keywords, and relevant publications"""
    keyword_items = [html.Li(f"{kw['name']} — Score: {kw['score']:.2f}")
                     for kw in faculty_row.get('top_keywords', [])]

    publication_items = [
        html.Li(f"{pub['title']} ({pub['venue']}, {pub['year']}) — Citations: {pub['num_citations']}")
        for pub in faculty_row.get("matched_publications", [])
    ]

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
                    html.P(f"Email: {faculty_row.get('email', 'N/A')}"),
                    html.P(f"Phone: {faculty_row.get('phone', 'N/A')}"),
                    html.P(f"Research Interests: {faculty_row.get('research_interest', 'N/A')}"),
                    html.H5("Top Keywords"),
                    html.Ul(keyword_items),
                    html.H5("Publications"),
                    html.Ul(publication_items if publication_items else [html.Li("None found")]),
                    html.Hr()
                ], width=9)
            ])
        ],
        className='mb-3 shadow-sm'
    )

def create_publication_card(pub_row):
    """Generate a card with publication metadata and author/keyword details."""
    raw_authors = pub_row.get("authors", [])
    cleaned_authors = [clean_author_name(a) for a in raw_authors if a and clean_author_name(a).lower() not in ['ph.d.', 'phd', 'dr.', 'dr']]
    authors = html.Ul([html.Li(a) for a in cleaned_authors]) if cleaned_authors else html.P("No authors listed.")

    keywords = html.Ul([html.Li(k) for k in pub_row.get("keywords", [])])

    return dbc.Card(
        dbc.CardBody([
            html.H4(pub_row["title"], className="card-title"),
            html.P(f"Venue: {pub_row['venue']}"),
            html.P(f"Year: {pub_row['year']}"),
            html.P(f"Citations: {pub_row['num_citations']}"),
            html.H5("Authors"),
            authors,
            html.H5("Keywords"),
            keywords
        ]),
        className="mt-3 shadow-sm"
    )

# Publication Analysis Functions
def get_top_publications(keyword, top_n):
    """Retrieve publications with highest keyword relevance and extra metadata"""
    query = """
    SELECT 
        p.id,
        p.title,
        p.venue,
        p.year,
        p.num_citations,
        MAX(pk.score) AS keyword_score,
        GROUP_CONCAT(DISTINCT f.name) AS authors,
        GROUP_CONCAT(DISTINCT k_all.name) AS keywords
    FROM publication p
    JOIN Publication_Keyword pk ON p.id = pk.publication_id
    JOIN keyword k ON pk.keyword_id = k.id
    LEFT JOIN Publication_Keyword pk_all ON p.id = pk_all.publication_id
    LEFT JOIN keyword k_all ON pk_all.keyword_id = k_all.id
    LEFT JOIN faculty_publication fp ON p.id = fp.publication_id
    LEFT JOIN faculty f ON fp.faculty_id = f.id
    WHERE k.name = %s
    GROUP BY p.id
    ORDER BY keyword_score DESC
    LIMIT %s;
    """
    return execute_query(query, (keyword, top_n))

def clean_author_name(name):
    # Remove common suffixes like ", Ph.D.", "PhD", "Dr.", etc.
    name = re.sub(r',?\s*(Ph\.?D\.?|Dr\.?|M\.?D\.?)$', '', name, flags=re.IGNORECASE)
    return name.strip()

# Callbacks for Publication Analysis Tab
@app.callback(
    [Output('publication-scores-chart', 'figure'),
     Output('publication-meta', 'children')],
    [Input('publication-keyword-input', 'value'),
     Input('pub-count-slider', 'value')]
)
def update_publication_analysis(keyword, top_n):
    if not keyword:
        return px.scatter(title="Enter a keyword to begin analysis"), ""

    df = get_top_publications(keyword, top_n)
    if df.empty:
        return px.scatter(title=f"No publications found for '{keyword}'"), ""

    # Pre-process authors/keywords for card display
    df["authors"] = df["authors"].fillna("").apply(lambda a: a.split(','))
    df["keywords"] = df["keywords"].fillna("").apply(lambda k: k.split(','))

    # Scatter Plot: X = year, Y = score
    fig = px.scatter(
        df,
        x="year",
        y="keyword_score",
        hover_name="title",
        size_max=10,
        labels={
            "keyword_score": "Keyword Relevance",
            "year": "Publication Year"
        },
        title=f"Top {top_n} Publications for '{keyword.title()}'"
    )    
    
    fig.update_xaxes(categoryorder='category ascending')  # ✅ Ensures year is sorted
    fig.update_traces(marker=dict(size=10), mode='markers')
    fig.update_layout(clickmode='event+select')
    author_series = df.explode("authors")["authors"].dropna().apply(clean_author_name)
    most_common_author = author_series.mode()[0] if not author_series.empty else "N/A"
    author_counts = Counter(author_series)
    top_authors = author_counts.most_common(5)

    # Insights summary
    stats = [
        html.H4("Analysis Insights:"),
        html.P(f"Median Keyword Score: {df['keyword_score'].median():.1f}"),
        html.P(f"Total Citations: {df['num_citations'].sum()}"),
        html.P(f"Year Range: {df['year'].min()} - {df['year'].max()}"),
        html.P("Top Authors:"),
        html.Ul([
            html.Li(f"{author} — {count} publication(s)")
            for author, count in top_authors
         ]) if top_authors else html.P("No authors found.")
    ]

    return fig, stats
    
@app.callback(
    Output('publication-detail-card', 'children'),
    Input('publication-scores-chart', 'clickData'),
    prevent_initial_call=True
)
def display_publication_card(clickData):
    if not clickData:
        raise PreventUpdate

    title = clickData['points'][0]['hovertext']
    query = """
    SELECT 
        p.title, p.venue, p.year, p.num_citations,
        GROUP_CONCAT(DISTINCT f.name) AS authors,
        GROUP_CONCAT(DISTINCT k.name) AS keywords
    FROM publication p
    LEFT JOIN faculty_publication fp ON p.id = fp.publication_id
    LEFT JOIN faculty f ON fp.faculty_id = f.id
    LEFT JOIN Publication_Keyword pk ON p.id = pk.publication_id
    LEFT JOIN keyword k ON pk.keyword_id = k.id
    WHERE p.title = %s
    GROUP BY p.id;
    """
    df = execute_query(query, (title,))
    if df.empty:
        return html.Div("No details available.")

    row = df.iloc[0].to_dict()
    row["authors"] = row["authors"].split(',') if row.get("authors") else []
    row["keywords"] = row["keywords"].split(',') if row.get("keywords") else []

    return create_publication_card(row)
    
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
    """
    all_keywords = execute_query(query, (university_name,))
    total_keywords = len(all_keywords)
    return all_keywords.head(top_n), total_keywords

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
    if not name_input or len(name_input.strip()) < 3:
        raise PreventUpdate

    try:
        top_n = int(top_n)
        top_n = max(1, min(top_n, 50))
    except (TypeError, ValueError):
        top_n = 10

    clean_name = name_input.strip().title()
    df, total_keywords = get_university_keyword_scores(clean_name, top_n)

    if df.empty:
        return (
            px.pie(title="No Data Found").update_layout(
                annotations=[dict(text=f"No results for '{clean_name}'", showarrow=False)]
            ),
            html.Div(
                "Please verify the university name",
                className='error-message',
                style={'color': '#dc3545'}
            ),
            None,
            {'display': 'none'}
        )

    photo_url = df.iloc[0]['photo_url'] if 'photo_url' in df.columns else None
    pie_fig = create_keyword_pie(df, clean_name)

    stats_content = html.Div([
        html.H4(f"{clean_name} Research Summary", className='summary-title'),
        html.Div([
            html.Div([
                html.Span("Total Keywords Available:", className='stat-label'),
                html.Span(f"{total_keywords}", className='stat-value')
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

    photo_style = {'height': '200px', 'objectFit': 'cover'} if photo_url else {'display': 'none'}

    return pie_fig, stats_content, photo_url, photo_style
    
def find_similar_keywords(keyword: str, min_connections: int = 3) -> list:
    """
    Returns a list of keywords that share at least `min_connections` professors or publications
    with the input keyword.
    """
    query = """
    // === Professor-based keyword similarity ===
    MATCH (k:KEYWORD {name: $keyword})
    MATCH (k)<-[:INTERESTED_IN]-(f:FACULTY)-[:INTERESTED_IN]->(other:KEYWORD)
    WHERE other.name <> $keyword
    WITH other.name AS keyword, COUNT(DISTINCT f) AS connCount
    WHERE connCount >= $min_conn
    RETURN keyword, ['Faculty'] AS viaEntities, connCount AS totalScore

    UNION

    // === Publication-based keyword similarity ===
    MATCH (k:KEYWORD {name: $keyword})
    MATCH (k)<-[:LABEL_BY]-(p:PUBLICATION)-[:LABEL_BY]->(other:KEYWORD)
    WHERE other.name <> $keyword
    WITH other.name AS keyword, COUNT(DISTINCT p) AS connCount
    WHERE connCount >= $min_conn
    RETURN keyword, ['Publication'] AS viaEntities, connCount AS totalScore

    ORDER BY totalScore DESC
    LIMIT 15
    """

    params = {
        "keyword": keyword.strip().lower(),
        "min_conn": min_connections
    }

    df = execute_query(query=query, params=params, db_type='neo4j')

    if df.empty:
        return []
    
    return df.to_dict('records')
    
def empty_figure(message: str = "No data available") -> go.Figure:
    """Return an empty Plotly figure with a centered message"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=20)
    )
    fig.update_layout(
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, visible=False),
        yaxis=dict(showgrid=False, visible=False)
    )
    return fig
    
def create_network_graph(nodes, links, root_keyword):
    # Create graph object
    G = nx.Graph()

    # Add nodes
    for node in nodes:
        G.add_node(node['id'], size=node['size'])

    # Add edges
    for link in links:
        G.add_edge(link['source'], link['target'], weight=link['value'], label=link['type'])

    # Positioning with spring layout
    pos = nx.spring_layout(G, seed=42)

    # Extract node and edge data
    edge_x = []
    edge_y = []
    edge_text = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_text.append(f"{edge[0]} ⇄ {edge[1]}<br>Type: {edge[2]['label']}")

    node_x = []
    node_y = []
    node_size = []
    node_text = []

    for node in G.nodes(data=True):
        x, y = pos[node[0]]
        node_x.append(x)
        node_y.append(y)
        node_size.append(node[1]['size'])
        node_text.append(node[0])

    # Create figure
    fig = go.Figure()

    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='gray'),
        hoverinfo='text',
        mode='lines'
    ))

    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition='top center',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=node_size,
            color=node_size,
            colorbar=dict(
                thickness=15,
                title='Connection Strength',
                xanchor='left',
            ),
            line_width=2
        )
    ))

    fig.update_layout(
        title=f"Keyword Network: {root_keyword}",
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode='closest'
    )

    return fig

# Callback Logic
@app.callback(
    Output('keyword-network', 'figure'),
   [Input('keyword-input', 'value'),  # Keyword entry
     Input('connection-threshold', 'value')], 
    prevent_initial_call=True
)
def update_keyword_network(selected_keyword, min_conn):
    if not selected_keyword:
        return empty_figure("Select a keyword to begin analysis")
    
    results = find_similar_keywords(selected_keyword, min_conn)
    
    if not results:
        return empty_figure("No significant connections found")
    
    # Visualization processing
    nodes = [{'id': selected_keyword, 'size': 40}] + [
        {'id': kw['keyword'], 'size': 20 + kw['totalScore']*2}
        for kw in results
    ]
    
    links = [{
        'source': selected_keyword,
        'target': kw['keyword'],
        'value': kw['totalScore'],
        'type': ' / '.join(kw['viaEntities'])
    } for kw in results]
    
    return create_network_graph(nodes, links, selected_keyword)
    
def process_admin_action(action: str, entity_type: str, entity_id: int, json_input: str = "") -> str:
    mongo = MongoDBUtils(MONGO_URI)
    if not mongo.connect():
        return "MongoDB connection failed"

    collection_map = {
        "faculty": "faculty_overrides",
        "publication": "publication_overrides",
        "university": "university_overrides"
    }

    collection = collection_map.get(entity_type)
    if not collection:
        return "Invalid entity type selected."

    try:
        if action == "submit":
            updates = json.loads(json_input)
            mongo.upsert_override(collection, entity_id, updates)

            # Log submission
            mongo.db["audit_log"].insert_one({
                "entity_type": entity_type,
                "entity_id": entity_id,
                "action": "submit",
                "timestamp": datetime.utcnow(),
                "changes": updates
            })

            return f"Override submitted for {entity_type.title()} ID {entity_id}."

        elif action == "delete":
            mongo.delete_override(collection, entity_id)

            mongo.db["audit_log"].insert_one({
                "entity_type": entity_type,
                "entity_id": entity_id,
                "action": "delete",
                "timestamp": datetime.utcnow()
            })

            return f"Override deleted for {entity_type.title()} ID {entity_id}."

        elif action == "approve":
            mongo.approve_override(collection, entity_id)

            # Log approval
            mongo.db["audit_log"].insert_one({
                "entity_type": entity_type,
                "entity_id": entity_id,
                "action": "approve",
                "timestamp": datetime.utcnow()
            })

            return f"Override approved for {entity_type.title()} ID {entity_id}."

        else:
            return "Unknown action."
    except json.JSONDecodeError:
        return "Invalid JSON input."
    except Exception as e:
        return f"Error: {str(e)}"
        
def sanitize_mongo_doc(doc):
    return {
        k: (str(v) if isinstance(v, ObjectId) else v)
        for k, v in doc.items()
    }

def reset_all_mongo_admin_data():
    mongo = MongoDBUtils(MONGO_URI)
    if mongo.connect():
        for collection in [
            "audit_log",
            "faculty_overrides",
            "publication_overrides",
            "university_overrides"
        ]:
            mongo.db[collection].delete_many({})
        mongo.close()

reset_all_mongo_admin_data()

@app.callback(
    [Output('admin-action-status', 'children'),
     Output('admin-pending-list', 'children'),
     Output('admin-history-list', 'children'),
     Output('admin-current-values', 'children')],
    [Input('admin-submit-btn', 'n_clicks'),
     Input('admin-delete-btn', 'n_clicks'),
     Input('admin-approve-btn', 'n_clicks')],
    [State('admin-entity-type', 'value'),
     State('admin-entity-id', 'value'),
     State('admin-json-editor', 'value')]
)
def handle_admin_actions(submit_click, delete_click, approve_click, entity_type, entity_id, json_input):
    triggered = dash.callback_context.triggered
    if not triggered or not entity_type or entity_id is None:
        raise PreventUpdate

    trigger_id = triggered[0]['prop_id'].split('.')[0]
    action = None
    if trigger_id == 'admin-submit-btn':
        action = "submit"
    elif trigger_id == 'admin-delete-btn':
        action = "delete"
    elif trigger_id == 'admin-approve-btn':
        action = "approve"
    else:
        raise PreventUpdate

    # Execute override action
    status = process_admin_action(action, entity_type, entity_id, json_input)

    mongo = MongoDBUtils(MONGO_URI)
    pending_list, history, current_json = [], [], "No override found."

    if mongo.connect():
        collection = f"{entity_type}_overrides"

        # Pending overrides
        pending = mongo.get_pending_overrides(collection)
        pending_list = [
            html.Li(f"{entity_type.title()} ID {item['entity_id']}: {item}")
            for item in pending
        ]

        # Current override preview
	# Lookup original entity
        original_doc = mongo.find(entity_type, {"id": entity_id})
        override_doc = mongo.find(collection, {"entity_id": entity_id})

	# Merge override (if any)
        if original_doc:
            merged = original_doc[0]
            if override_doc:
                for key, val in override_doc[0].items():
                    if key not in ["_id", "entity_id", "approved"]:
                        merged[key] = val
                        sanitized = sanitize_mongo_doc(merged)
            current_json = json.dumps(sanitized, indent=2)
        else:
            current_json = "Entity not found in MongoDB."


        # History from audit log
        audit = mongo.db["audit_log"].find(
            {"entity_type": entity_type, "entity_id": entity_id}
        ).sort("timestamp", -1).limit(5)
        history = [
            html.Li(f"{doc['timestamp']} - {doc['action'].title()}: {doc.get('changes', {})}")
            for doc in audit
        ]

    return status, html.Ul(pending_list), html.Ul(history), current_json
    
def get_top_faculty_and_publications_by_keywords(keywords: list[str], top_n: int = 10):
    faculty_query = """
    SELECT 
        f.id,
        f.name,
        f.position,
        u.name AS university,
        SUM(fk.score) AS total_score,
        COUNT(DISTINCT k.name) AS matched_keywords
    FROM faculty f
    JOIN faculty_keyword fk ON f.id = fk.faculty_id
    JOIN keyword k ON fk.keyword_id = k.id
    JOIN university u ON f.university_id = u.id
    WHERE k.name IN %(keywords)s
    GROUP BY f.id, u.name
    ORDER BY total_score DESC
    LIMIT %(limit)s;
    """

    pub_query = """
    SELECT 
        p.id,
        p.title,
        p.venue,
        p.year,
        SUM(pk.score) AS total_score,
        COUNT(DISTINCT k.name) AS matched_keywords
    FROM publication p
    JOIN Publication_Keyword pk ON p.id = pk.publication_id
    JOIN keyword k ON pk.keyword_id = k.id
    WHERE k.name IN %(keywords)s
    GROUP BY p.id
    ORDER BY total_score DESC
    LIMIT %(limit)s;
    """

    params = {"keywords": tuple(keywords), "limit": top_n}
    top_faculty = execute_query(faculty_query, params)
    top_pubs = execute_query(pub_query, params)

    return top_faculty, top_pubs
    
def display_clicked_faculty_from_match(clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    faculty_name = eval(triggered_id)['index']
    
    try:
        details = get_full_faculty_details(faculty_name)
        return create_faculty_card(details)
    except Exception as e:
        return html.Div("Error loading faculty details.")

@app.callback(
    [Output('kw-match-status', 'children'),
     Output('kw-match-faculty', 'children'),
     Output('kw-match-publications', 'children'),
     Output('kw-match-faculty-card', 'children')],
    [Input('kw-match-btn', 'n_clicks'),
     Input({'type': 'kwmatch-faculty-name', 'index': dash.dependencies.ALL}, 'n_clicks')],
    [State('kw-match-input', 'value'),
     State('kw-match-limit', 'value')]
)
def keyword_match(btn_clicks, faculty_clicks, keyword_input, limit):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    # Check if the trigger came from a clicked faculty name
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if "kwmatch-faculty-name" in triggered_id:
        faculty_name = eval(triggered_id)['index']
        try:
            details = get_full_faculty_details(faculty_name)
            return dash.no_update, dash.no_update, dash.no_update, create_faculty_card(details)
        except Exception:
            return dash.no_update, dash.no_update, dash.no_update, html.Div("Error loading faculty details.")

    # Otherwise, it's a keyword search
    if not keyword_input:
        return "Enter keywords to begin search.", "", "", ""

    keywords = [kw.strip().lower() for kw in keyword_input.split(',') if kw.strip()]
    if len(keywords) < 2:
        return "Enter at least two keywords.", "", "", ""

    faculty_df, pub_df = get_top_faculty_and_publications_by_keywords(keywords, limit)

    faculty_list = [
        html.Div([
            html.A(
                row['name'],
                href="#",
                id={'type': 'kwmatch-faculty-name', 'index': row['id']},
                n_clicks=0,
                style={'fontWeight': 'bold'}
            ),
            html.Span(f" — {row['position']} at {row['university']} — Score: {row['total_score']}")
        ]) for _, row in faculty_df.iterrows()
    ] if not faculty_df.empty else [html.Div("No matching faculty found.")]

    pub_list = [
        html.Div(f"{row['title']} ({row['venue']}, {row['year']}) — Score: {row['total_score']}")
        for _, row in pub_df.iterrows()
    ] if not pub_df.empty else [html.Div("No matching publications found.")]

    return f"Results for: {', '.join(keywords)}", faculty_list, pub_list, ""

if __name__ == '__main__':
    app.run(debug=True, dev_tools_hot_reload=False, port=8050)


