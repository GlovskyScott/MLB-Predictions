_TEAMS = {
    110: {'abbr': 'BAL', 'name': 'Baltimore Orioles',      'primary': '#DF4601', 'secondary': '#000000'},
    111: {'abbr': 'BOS', 'name': 'Boston Red Sox',         'primary': '#BD3039', 'secondary': '#0C2340'},
    133: {'abbr': 'OAK', 'name': 'Oakland Athletics',      'primary': '#003831', 'secondary': '#EFB21E'},
    136: {'abbr': 'SEA', 'name': 'Seattle Mariners',       'primary': '#0C2C56', 'secondary': '#005C5C'},
    108: {'abbr': 'LAA', 'name': 'Los Angeles Angels',     'primary': '#BA0021', 'secondary': '#003263'},
    117: {'abbr': 'HOU', 'name': 'Houston Astros',         'primary': '#002D62', 'secondary': '#EB6E1F'},
    140: {'abbr': 'TEX', 'name': 'Texas Rangers',          'primary': '#003278', 'secondary': '#C0111F'},
    141: {'abbr': 'TOR', 'name': 'Toronto Blue Jays',      'primary': '#134A8E', 'secondary': '#1D2D5C'},
    139: {'abbr': 'TBR', 'name': 'Tampa Bay Rays',         'primary': '#092C5C', 'secondary': '#8FBCE6'},
    142: {'abbr': 'MIN', 'name': 'Minnesota Twins',        'primary': '#002B5C', 'secondary': '#D31145'},
    145: {'abbr': 'CWS', 'name': 'Chicago White Sox',      'primary': '#27251F', 'secondary': '#C4CED4'},
    116: {'abbr': 'DET', 'name': 'Detroit Tigers',         'primary': '#0C2340', 'secondary': '#FA4616'},
    118: {'abbr': 'KCR', 'name': 'Kansas City Royals',     'primary': '#004687', 'secondary': '#BD9B60'},
    114: {'abbr': 'CLE', 'name': 'Cleveland Guardians',    'primary': '#00385D', 'secondary': '#E31937'},
    147: {'abbr': 'NYY', 'name': 'New York Yankees',       'primary': '#132448', 'secondary': '#C4CED4'},
    144: {'abbr': 'ATL', 'name': 'Atlanta Braves',         'primary': '#CE1141', 'secondary': '#13274F'},
    146: {'abbr': 'MIA', 'name': 'Miami Marlins',          'primary': '#00A3E0', 'secondary': '#EF3340'},
    121: {'abbr': 'NYM', 'name': 'New York Mets',          'primary': '#002D72', 'secondary': '#FF5910'},
    143: {'abbr': 'PHI', 'name': 'Philadelphia Phillies',  'primary': '#E81828', 'secondary': '#002D72'},
    120: {'abbr': 'WSN', 'name': 'Washington Nationals',   'primary': '#AB0003', 'secondary': '#14225A'},
    112: {'abbr': 'CHC', 'name': 'Chicago Cubs',           'primary': '#0E3386', 'secondary': '#CC3433'},
    113: {'abbr': 'CIN', 'name': 'Cincinnati Reds',        'primary': '#C6011F', 'secondary': '#000000'},
    158: {'abbr': 'MIL', 'name': 'Milwaukee Brewers',      'primary': '#12284B', 'secondary': '#FFC52F'},
    134: {'abbr': 'PIT', 'name': 'Pittsburgh Pirates',     'primary': '#FDB827', 'secondary': '#27251F'},
    138: {'abbr': 'STL', 'name': 'St. Louis Cardinals',    'primary': '#C41E3A', 'secondary': '#0C2340'},
    109: {'abbr': 'ARI', 'name': 'Arizona Diamondbacks',   'primary': '#A71930', 'secondary': '#E3D4AD'},
    115: {'abbr': 'COL', 'name': 'Colorado Rockies',       'primary': '#333366', 'secondary': '#C4CED4'},
    119: {'abbr': 'LAD', 'name': 'Los Angeles Dodgers',    'primary': '#005A9C', 'secondary': '#EF3E42'},
    135: {'abbr': 'SDP', 'name': 'San Diego Padres',       'primary': '#2F241D', 'secondary': '#FFC425'},
    137: {'abbr': 'SFG', 'name': 'San Francisco Giants',   'primary': '#FD5A1E', 'secondary': '#27251F'},
}

_FALLBACK = {'abbr': '???', 'name': 'Unknown', 'primary': '#888888', 'secondary': '#444444'}


def get_team_meta(team_id: int) -> dict:
    t = _TEAMS.get(team_id, _FALLBACK)
    return {
        'abbr': t['abbr'],
        'name': t['name'],
        'primary': t['primary'],
        'secondary': t['secondary'],
        'logo_url': f'https://www.mlbstatic.com/team-logos/{team_id}.svg',
    }
