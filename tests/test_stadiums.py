import json
import pytest
from src.stadiums import get_stadium, get_all_stadiums, WIND_DIRECTIONS

def test_get_stadium_returns_dict_for_valid_team_id():
    stadium = get_stadium(147)  # Yankees
    assert isinstance(stadium, dict)
    assert 'name' in stadium
    assert 'lat' in stadium
    assert 'lon' in stadium
    assert 'roof' in stadium

def test_get_stadium_has_correct_yankees_data():
    stadium = get_stadium(147)
    assert stadium['name'] == 'Yankee Stadium'
    assert abs(stadium['lat'] - 40.8296) < 0.01
    assert abs(stadium['lon'] - (-73.9262)) < 0.01

def test_get_stadium_returns_none_for_invalid_id():
    assert get_stadium(9999) is None

def test_get_all_stadiums_has_30_teams():
    stadiums = get_all_stadiums()
    assert len(stadiums) == 30

def test_all_stadiums_have_required_fields():
    stadiums = get_all_stadiums()
    required = {'name', 'team', 'city', 'lat', 'lon', 'roof'}
    for team_id, data in stadiums.items():
        assert required.issubset(data.keys()), f"Team {team_id} missing fields"

def test_roof_types_are_valid():
    stadiums = get_all_stadiums()
    valid_roofs = {'open', 'retractable', 'dome'}
    for team_id, data in stadiums.items():
        assert data['roof'] in valid_roofs, f"Team {team_id} has invalid roof: {data['roof']}"
