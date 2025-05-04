# app.py - Academic Research Analytics Dashboard (Modified Faculty Explorer)
import dash
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
from datetime import datetime
from bson import ObjectId
from collections import Counter
from pymongo import MongoClient
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
mongo = MongoDBUtils(MONGO_URI)
mongo.connect()

NEO4J_CONFIG = {
     'database': 'academicworld',
}

app.layout = html.Div([
    html.H1("Academic Keywords Analytics Dashboard", className='main-header'),
    html.Hr(),
    dbc.Container([
        # === Top Row ===
        html.Div([
                	html.Div([
            	html.H2("Faculty Explorer"),
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
                    	dcc.Input(
                        	id='faculty-count-input',
                        	type='number',
                        	value=10,
                        	min=1,
                        	max=1000,
                        	step=1,
                        	debounce=True,
                        	style={'width': '100px'}
                    	)
                	], className='control-column')
            	], className='control-panel'),

            	html.Div([
                	html.Div(id='faculty-list-container', className='scrollable-list'),
                	html.Div(id='faculty-details-container', className='details-panel')
            	], className='list-detail-container'),

            	dcc.Store(id='faculty-cache-store'),
        	],style={'width': '30%'}),

        	html.Div([
            	html.H2("Publication Explorer"),
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
                    	dcc.Input(
                        	id='pub-count-input',
                        	type='number',
                        	value=20,
                        	min=1,
                        	max=1000,
                        	step=1,
                        	debounce=True,
                        	style={'width': '100px'}
                    	)
                	], className='control-column')
            	], className='control-panel'),

            	dcc.Graph(id='publication-scores-chart'),
            	html.Div(id='publication-meta', className='stats-panel'),
            	html.Div(id='publication-detail-card', className='detail-panel')
        	],style={'width': '30%'}),
        	html.Div([
            	html.H2("University Analysis"),
            	html.Div([
                	html.Div(
                    	id='university-photo-card',
                    	className='photo-card',
                    	children=[
                        	html.Img(
                            	id='university-photo',
                            	className='university-photo',
                            	style={'display': 'none'}
                        	)
                    	]
                	),
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
                        	html.Label("Top Keywords to Display (up to 20):", className='control-label'),
                        	dcc.Input(
                            	id='university-tab-top-n',
                            	type='number',
                            	value=10,
                            	min=1,
                            	max=20,
                            	step=1,
                            	debounce=True,
                            	style={'width': '100px'}
                        	)
                    	], className='control-column')
                	], className='control-panel'),

                	dbc.Row([
                    	dbc.Col(
                        	dcc.Graph(id='university-tab-pie-chart', config={'displayModeBar': False}, className='pie-container'),
                        	style={'height': '500px'}
                    	),
                    	dbc.Col(
                        	html.Div(id='university-keyword-list', style={'overflowY': 'auto', 'maxHeight': '500px'}),
                        	width=6
                    	)
                	]),
                	html.Div(id='university-stats-panel', className='stats-panel')
            	], className='university-tab-content')
        	], style={'width': '30%'})
        ],style={
    'display': 'flex',
    'flexWrap': 'nowrap',
    'justifyContent': 'space-between',
    'alignItems': 'flex-start',
    'marginBottom': '30px'
        }
        ),

        html.Hr(),

        # === Bottom Row ===
        dbc.Row([
            dbc.Col([
                html.H2("Related Keyword Explorer"),
                html.Div([
                    dcc.Input(
                        id='keyword-input',
                        type='text',
                        placeholder='Enter research keyword...',
                        debounce=True,
                        style={'width': '100%', 'padding': '10px'}
                    ),
                    dcc.Dropdown(
                        id='keyword-filter-type',
                        options=[
                            {'label': 'Both Faculty and Publications', 'value': 'both'},
                            {'label': 'Faculty Only', 'value': 'faculty'},
                            {'label': 'Publications Only', 'value': 'publication'}
                        ],
                        value='both',
                        clearable=False,
                        style={'width': '300px', 'marginTop': '10px'}
                    ),
                    html.Div([
                        html.Label("Minimum Connections:", className='control-label'),
                        dcc.Input(
                            id='connection-threshold',
                            type='number',
                            value=2,
                            min=1,
                            max=1000,
                            step=1,
                            debounce=True,
                            style={'width': '100px'}
                        )
                    ], className='threshold-control')
                ], className='control-bar'),

                dcc.Graph(
                    id='keyword-network',
                    style={'height': '400px', 'width': '100%'},
                    config={'displayModeBar': False},
                    className='network-graph'
                ),
                html.Div(id='kw-match-faculty-card'),
                html.Div(id='related-keyword-results'),
                html.Br(),
                html.H4("Select a Related Keyword"),
                dcc.Dropdown(id='related-keyword-dropdown', placeholder="Select from related keywords", multi=True),
                html.Div(id='keyword-matching-results')
            ], width=4),
            html.Hr(),

# === Database Editor and Keyword Editor in one row ===
html.H2("Database Editor & Keyword Editor"),
html.Div([
    html.Div([
        html.H3("Data Override & Approval Panel"),
        dcc.Dropdown(
            id='admin-entity-type',
            options=[
                {'label': 'Faculty', 'value': 'faculty'},
                {'label': 'Publication', 'value': 'publications'}
            ],
            placeholder="Select entity type"
        ),
        dcc.Input(id='admin-entity-name', type='text', placeholder='Enter entity name'),
        dcc.Textarea(
            id='admin-json-editor',
            placeholder='Enter field-value JSON (e.g., {"email": "new@edu.edu"})',
            style={'width': '100%', 'height': '50px'}
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
    ], style={'width': '48%'}),
    
    html.Div(style={
        'width': '1px',
        'backgroundColor': '#ccc',
        'margin': '0 10px'
    }),
    html.Div([
        html.H3("Keyword Editor"),
        dcc.Input(id='entity-type-input', type='text', placeholder='Enter "faculty" or "publication"'),
        dcc.Input(id='entity-name-input', type='text', placeholder='Enter professor or publication name'),
        dcc.Input(id='keyword-name-input', type='text', placeholder='Enter keyword name'),
        dcc.Input(id='new-score-input', type='number', placeholder='Enter new score'),
        html.Button('Update Score', id='update-score-btn'),
        html.Div(id='update-score-status')
    ], style={'width': '48%'})
], style={'display': 'flex', 'flexWrap': 'wrap'}),
        ])
    ], fluid=True)
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
            if not n4j.connect(): raise ConnectionError("Neo4j connection failed")
            records = n4j.execute_query(query, params)
            # Convert Neo4j records to DataFrame
            data = [dict(rec) for rec in records]
            #print(pd.DataFrame(data))
            return pd.DataFrame(data)
                
        except Exception as e:
            print(f"Neo4j Error: {str(e)}")
            return pd.DataFrame()
            
    else:
        raise ValueError("Invalid database type. Use 'mysql' or 'neo4j'")

# Faculty Analysis Functions (modified)
def get_faculty_list(search_term, limit):
    """Retrieve simplified faculty list with basic info using a view"""
    query = """
    SELECT
        faculty_id AS id,
        faculty_name AS name,
        position,
        university_name as university,
        MAX(score) AS keyword_score
    FROM faculty_keyword_score_view
    WHERE keyword LIKE %s
    GROUP BY faculty_id, university_name
    ORDER BY keyword_score DESC
    LIMIT %s;
    """
    search_pattern = f"%{search_term}%"
    return execute_query(query, (search_pattern, limit))


def get_full_faculty_details(faculty_id: int) -> dict[str, any]:
    """Retrieve full faculty profile with top keywords and recent publications using MongoDB."""

    faculty_result = mongo.find("faculty", {"id": faculty_id})
    
    if not faculty_result:
        raise ValueError("Faculty not found.")
    
    faculty = faculty_result[0]
    name = faculty.get("name")
    override_doc = mongo.find("faculty_overrides", {"entity_str": name, "approved": True})
    if override_doc:
        for k, v in override_doc[0].items():
            if k not in ["_id", "entity_id", "approved"]:
                faculty[k] = v

    # Extract and sort top keywords by score
    keywords = sorted(
        faculty.get("keywords", []),
        key=lambda k: k.get("score", 0),
        reverse=True
    )[:10]

    # Retrieve publications by matching publication IDs
    profile = {
        "id": faculty.get("id"),
        "name": faculty.get("name"),
        "position": faculty.get("position"),
        "photo_url": faculty.get("photoUrl"),
        "email": faculty.get("email"),
        "phone": faculty.get("phone"),
        "research_interest": faculty.get("researchInterest"),
        "university": faculty.get("affiliation", {}).get("name", "Unknown"),
        "top_keywords": [{"name": k["name"], "score": k["score"]} for k in keywords],
    }
    # Get top 5 publication details by publication IDs
    publication_ids = faculty.get("publications", [])[:5]  # Limit to 5 recent ones
    publications = mongo.find(
        "publications",
        {"id": {"$in": publication_ids}},
        projection={"_id": 0}
    )
# Make sure the result is a list and sort by year
    if publications:
        sorted_pubs = sorted(publications, key=lambda x: x.get("year", 0), reverse=True)
        profile["matched_publications"] = sorted_pubs
    else:
        profile["matched_publications"] = []

    return profile

# Modified Faculty Callbacks
@app.callback(
    [Output('faculty-list-container', 'children'),
     Output('faculty-cache-store', 'data')],
    [Input('faculty-keyword-input', 'value'),
     Input('faculty-count-input', 'value')]
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
    keyword_items = [
        html.Li(f"{kw['name']} — Score: {kw['score']:.2f}")
        for kw in faculty_row.get('top_keywords', [])
    ]

    publication_items = [
        html.Li(f"{pub['title']} ({pub['venue']}, {pub['year']}) — Citations: {pub['numCitations']}")
        for pub in faculty_row.get("matched_publications", [])
    ]

    return dbc.Card([
        dbc.Row([
            dbc.Col(
                html.Img(
                    src=faculty_row.get('photo_url', ''),
                    className='faculty-photo',
                    style={'height': '150px', 'objectFit': 'cover'}
                ),
                width=3
            ),
            dbc.Col([
                html.H4(faculty_row.get('name', 'N/A'), className='card-title'),
                html.P(f"{faculty_row.get('position', '')} @ {faculty_row.get('university', 'Unknown')}",
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
    ], className='mb-3 shadow-sm')

def create_publication_card(pub_row):
    cleaned_authors = [clean_author_name(a) for a in pub_row.get("authors", []) if a]
    authors = html.Ul([html.Li(a) for a in cleaned_authors]) if cleaned_authors else html.P("No authors listed.")

    keywords = html.Ul([html.Li(k) for k in pub_row.get("keywords", [])])

    return dbc.Card(
        dbc.CardBody([
            html.H4(pub_row["title"], className="card-title"),
            html.P(f"Venue: {pub_row['venue']}"),
            html.P(f"Year: {pub_row['year']}"),
            html.P(f"Citations: {pub_row.get('numCitations', 0)}"),
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
    publication_id,
    title,
    venue,
    year,
    num_citations,
    MAX(score) AS keyword_score,
    GROUP_CONCAT(DISTINCT faculty_name) AS authors,
    GROUP_CONCAT(DISTINCT all_keyword) AS keywords
    FROM publication_keyword_detail_view
    WHERE keyword = %s
    GROUP BY publication_id
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
     Input('pub-count-input', 'value')]
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
    
    results = mongo.find("publications", {"title": title})
    if not results:
        return html.Div("No details available.")

    pub = results[0]
    name = pub.get("title")
    override_doc = mongo.find("publications_overrides", {"entity_str": name, "approved": True})
    if override_doc:
        for k, v in override_doc[0].items():
            if k not in ["_id", "entity_id", "approved"]:
                pub[k] = v
    
    # Retrieve author names from faculty documents
    faculty_docs = mongo.find("faculty", {"publications": pub.get("id", -1)})
    authors = [f.get("name") for f in faculty_docs if f.get("name")]

    pub["authors"] = authors
    pub["keywords"] = [k["name"] for k in pub.get("keywords", [])]

    return create_publication_card(pub)
    
# security-enhanced database function
def get_university_keyword_scores(university_name, top_n):
    """Get aggregated keyword scores for a university's faculty"""
    query = """
    SELECT
    keyword,
    SUM(score) AS total_score,
    COUNT(DISTINCT faculty_id) AS professor_count,
    GROUP_CONCAT(DISTINCT faculty_name) AS professors,
    MAX(photo_url) AS photo_url
    FROM university_keyword_summary_view
    WHERE university_name LIKE %s
    GROUP BY keyword
    ORDER BY total_score DESC;
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
        orientation="v",       # Vertical layout
        yanchor="top",
        y=1,
        xanchor="left",
        x=1.05,                # Shift to the right of the pie chart
        font=dict(size=12)
    ),
    hoverlabel=dict(
        bgcolor="white",
        font_size=14
    ),
    margin=dict(l=40, r=120, t=40, b=40)  # Add right margin to make room for legend
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
                html.Span("Total Score:", className='stat-label'),
                html.Span(f"{df['total_score'].sum()}", className='stat-value')
            ], className = 'stat-item'),
            html.Div([
                html.Span("Top Research Area:", className='stat-label'),
                html.Span(df.iloc[0]['keyword'], className='stat-value')
            ], className='stat-item')
        ], className='stats-grid')
    ])

    photo_style = {'height': '200px', 'objectFit': 'cover'} if photo_url else {'display': 'none'}

    return pie_fig, stats_content, photo_url, photo_style
    
def find_similar_keywords(keyword: str, min_connections: int = 3, filter_type: str = "both") -> list:
    """
    Returns a list of keywords that share at least `min_connections` professors or publications
    with the input keyword. Filter by 'faculty', 'publication', or 'both'.
    """
    query = """
// === Faculty and publication keyword similarity ===
MATCH (k:KEYWORD {name: $keyword})
WITH k

// Faculty relationships
OPTIONAL MATCH (k)<-[:INTERESTED_IN]-(f:FACULTY)-[:INTERESTED_IN]->(other1:KEYWORD)
WHERE other1.name <> $keyword
WITH k, other1.name AS keyword, COUNT(DISTINCT f) AS facultyCount

// Publication relationships
OPTIONAL MATCH (k)<-[:LABEL_BY]-(p:PUBLICATION)-[:LABEL_BY]->(other2:KEYWORD)
WHERE other2.name <> $keyword AND other2.name = keyword
WITH keyword,
     facultyCount,
     COUNT(DISTINCT p) AS pubCount,
     CASE $filter_type
         WHEN 'faculty' THEN facultyCount
         WHEN 'publication' THEN COUNT(DISTINCT p)
         ELSE (facultyCount + COUNT(DISTINCT p))
     END AS totalScore,
     COUNT(DISTINCT p) AS pubCountFinal  // Needed separately
WITH keyword,
     facultyCount,
     pubCountFinal,
     totalScore,
     [x IN ['Faculty', 'Publication']
      WHERE (x = 'Faculty' AND facultyCount > 0) OR (x = 'Publication' AND pubCountFinal > 0)] AS viaEntities
WHERE totalScore >= $min_conn
RETURN keyword, viaEntities, totalScore
ORDER BY totalScore DESC
    """

    params = {
        "keyword": keyword.strip().lower(),
        "min_conn": min_connections,
        "filter_type": filter_type.lower()
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

    # Add nodes with metadata
    for node in nodes:
        G.add_node(node['id'], size=node['size'], score=node.get('score', 0))

    # Add edges with metadata
    for link in links:
        G.add_edge(link['source'], link['target'], weight=link['value'], label=link['type'])

    # Layout for visualization
    pos = nx.spring_layout(G, seed=42)

    # Edges for visualization
    edge_x = []
    edge_y = []
    edge_text = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_text.append(f"{edge[0]} ⇄ {edge[1]}<br>Type: {edge[2]['label']}<br>Strength: {edge[2]['weight']}")

    # Nodes for visualization
    node_x = []
    node_y = []
    node_size = []
    node_text = []
    for node in G.nodes(data=True):
        x, y = pos[node[0]]
        node_x.append(x)
        node_y.append(y)
        node_size.append(node[1]['score'])
        hover_info = f"{node[0]}<br>Connection Strength: {node[1].get('score', 0)}"
        node_text.append(hover_info)

    # Build figure
    fig = go.Figure()

    # Edge trace
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=1, color='gray'),
        hoverinfo='text',
        text=edge_text,
        name='Edges'
    ))

    # Node trace
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[n['id'] for n in nodes],
        hovertext=node_text,
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
    
def get_top_faculty_and_publications_by_keywords(keywords: list[str], keyword_count: int = 2, top_n: int = 10):
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
    HAVING COUNT(DISTINCT k.name) >= %(keyword_count)s
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
    HAVING COUNT(DISTINCT k.name) >= %(keyword_count)s
    ORDER BY total_score DESC
    LIMIT %(limit)s;
    """

    params = {"keywords": tuple(keywords), "keyword_count": len(keywords),"limit": top_n}
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
# Callback Logic
@app.callback(
    [
        Output('keyword-network', 'figure'),
        Output('related-keyword-results', 'children'),
        Output('related-keyword-dropdown', 'options'),
        Output('keyword-matching-results', 'children'),
        Output('kw-match-faculty-card', 'children')
    ],
    [
        Input('keyword-input', 'value'),
        Input('connection-threshold', 'value'),
        Input('keyword-filter-type', 'value'),
        Input('related-keyword-dropdown', 'value'),
        Input({'type': 'kwmatch-faculty-name', 'index': dash.dependencies.ALL}, 'n_clicks')
    ],
    prevent_initial_call=True
)
def update_keyword_and_match(selected_keyword, min_conn, filter_type, dropdown_value, faculty_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Handle faculty card click
    if "kwmatch-faculty-name" in triggered_id:
        faculty_name = eval(triggered_id)['index']
        try:
            details = get_full_faculty_details(faculty_name)
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, create_faculty_card(details)
        except Exception:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, html.Div("Error loading faculty details.")

    # If no input keyword, halt
    if not selected_keyword:
        return empty_figure("Select a keyword to begin analysis"), [], [], "", ""

    # Run Neo4j keyword network
    results = find_similar_keywords(selected_keyword, min_conn, filter_type)
    if not results:
        return empty_figure("No significant connections found"), [], [], "", ""

    # Build graph data
    nodes = [{'id': selected_keyword, 'size': 40, 'score': 0}] + [
        {'id': kw['keyword'], 'size': 20, 'score': kw['totalScore']} for kw in results
    ]
    links = [{
        'source': selected_keyword,
        'target': kw['keyword'],
        'value': kw['totalScore'],
        'type': ' / '.join(kw['viaEntities'])
    } for kw in results]

    # Dropdown options
    dropdown_options = [{'label': kw['keyword'], 'value': kw['keyword']} for kw in results]

    # Run matcher only if a dropdown keyword is selected
    if dropdown_value:
        keywords = [selected_keyword.lower()] + [kw.lower() for kw in dropdown_value] if isinstance(dropdown_value, list) else [selected_keyword.lower(), dropdown_value.lower()]
        faculty_df, pub_df = get_top_faculty_and_publications_by_keywords(keywords)

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

        match_output = html.Div([
            html.H5(f"Results for: {selected_keyword}, {dropdown_value}"),
            html.Br(),
            html.Div(faculty_list, style={'marginBottom': '30px'}),
            html.Div(pub_list)
        ])
    else:
        match_output = ""

    return create_network_graph(nodes, links, selected_keyword), None, dropdown_options, match_output, ""
    
def process_admin_action(action: str, entity_type: str, entity_name: str, json_input: str = "") -> str:
    collection_map = {
        "faculty": "faculty_overrides",
        "publications": "publications_overrides",
    }

    collection = collection_map.get(entity_type)
    if not collection:
        return "Invalid entity type selected."

    try:
        if action == "submit":
            updates = json.loads(json_input)
            mongo.upsert_override(collection, entity_name, updates)

            # Log submission
            mongo.db["audit_log"].insert_one({
                "entity_type": entity_type,
                "entity_name": entity_name,
                "action": "submit",
                "timestamp": datetime.utcnow(),
                "changes": updates
            })

            return f"Override submitted for {entity_type.title()} Name {entity_name}."

        elif action == "delete":
            mongo.delete_override(collection, entity_name)

            mongo.db["audit_log"].insert_one({
                "entity_type": entity_type,
                "entity_name": entity_name,
                "action": "delete",
                "timestamp": datetime.utcnow()
            })

            return f"Override deleted for {entity_type.title()} Name {entity_name}."

        elif action == "approve":
            mongo.approve_override(collection, entity_name)

            # Log approval
            mongo.db["audit_log"].insert_one({
                "entity_type": entity_type,
                "entity_name": entity_name,
                "action": "approve",
                "timestamp": datetime.utcnow()
            })

            return f"Override approved for {entity_type.title()} Name {entity_name}."

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
            "publications_overrides",
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
     State('admin-entity-name', 'value'),
     State('admin-json-editor', 'value')]
)
def handle_admin_actions(submit_click, delete_click, approve_click, entity_type, entity_name, json_input):
    triggered = dash.callback_context.triggered
    if not triggered or not entity_type or entity_name is None:
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
    status = process_admin_action(action, entity_type, entity_name, json_input)

    mongo = MongoDBUtils(MONGO_URI)
    pending_list, history, current_json = [], [], "No override found."

    if mongo.connect():
        collection = f"{entity_type}_overrides"

        # Pending overrides
        pending = mongo.get_pending_overrides(collection)
        pending_list = [
            html.Li(f"{entity_type.title()} ID {item['entity_str']}: {item}")
            for item in pending
        ]

        # Current override preview
	# Lookup original entity
        if entity_type == 'faculty':
            original_doc = mongo.find(entity_type, {"name": entity_name})
        else:
            original_doc = mongo.find(entity_type, {"title": entity_name})
        override_doc = mongo.find(collection, {"entity_name": entity_name})

	# Merge override (if any)
        if original_doc:
            merged = original_doc[0]
            if override_doc:
                for key, val in override_doc[0].items():
                    if key not in ["_id", "entity_name", "approved"]:
                        merged[key] = val
            sanitized = sanitize_mongo_doc(merged)
            current_json = json.dumps(sanitized, indent=2)
        else:
            current_json = "Entity not found in MongoDB."


        # History from audit log
        audit = mongo.db["audit_log"].find(
            {"entity_type": entity_type, "entity_name": entity_name}
        ).sort("timestamp", -1).limit(5)
        history = [
            html.Li(f"{doc['timestamp']} - {doc['action'].title()}: {doc.get('changes', {})}")
            for doc in audit
        ]

    return status, html.Ul(pending_list), html.Ul(history), current_json

@app.callback(
    Output('update-score-status', 'children'),
    Input('update-score-btn', 'n_clicks'),
    State('entity-type-input', 'value'),
    State('entity-name-input', 'value'),
    State('keyword-name-input', 'value'),
    State('new-score-input', 'value')
)
def update_keyword_score(n_clicks, entity_type, entity_name, keyword_name, new_score):
    if not all([entity_type, entity_name, keyword_name, new_score]):
        raise dash.exceptions.PreventUpdate

    entity_type = entity_type.lower().strip()
    if entity_type not in ['faculty', 'publication']:
        return "Invalid entity type. Use 'faculty' or 'publications'."

    # Look up entity ID
    entity_table = 'faculty' if entity_type == 'faculty' else 'publication'
    entity_id_result = execute_query(
        f"SELECT id FROM {entity_table} WHERE name = %(name)s",
        {'name': entity_name}
    )
    if entity_id_result.empty:
        return f"{entity_type.capitalize()} '{entity_name}' not found."
    entity_id = entity_id_result.iloc[0]['id']

    # Look up keyword ID
    keyword_result = execute_query(
        "SELECT id FROM keyword WHERE name = %(name)s",
        {'name': keyword_name}
    )
    if keyword_result.empty:
        return f"Keyword '{keyword_name}' not found."
    keyword_id = keyword_result.iloc[0]['id']

    # Perform update
    mapping_table = 'faculty_keyword' if entity_type == 'faculty' else 'Publication_Keyword'
    id_field = f"{entity_type}_id"
    update_query = f"""
        UPDATE {mapping_table}
        SET score = %(new_score)s
        WHERE {id_field} = %(entity_id)s AND keyword_id = %(keyword_id)s
    """
    execute_query(update_query, {
        'new_score': new_score,
        'entity_id': entity_id,
        'keyword_id': keyword_id
    })

    return f"Updated score for {entity_type} '{entity_name}' and keyword '{keyword_name}' to {new_score}."


if __name__ == '__main__':
    app.run(debug=True, dev_tools_hot_reload=False, port=8050)

