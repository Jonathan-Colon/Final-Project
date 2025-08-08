#Import all libraries
import requests
import json
import random
import itertools
import pprint

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.cluster import KMeans

import ipywidgets as widgets
from IPython.display import display, clear_output, Markdown

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State


#API call for PokeAPI
def get_pokemon_data(name):
    url = f"https://pokeapi.co/api/v2/pokemon/{name.lower()}"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Data not found for {name}")
        return None

    data = response.json()

    stats = {stat['stat']['name']: stat['base_stat'] for stat in data['stats']}
    types = [type_info['type']['name'] for type_info in data['types']]
    moves = [move_info['move']['name'] for move_info in data['moves']]
    abilities = [ability_info['ability']['name'] for ability_info in data['abilities']] # Added abilities
    species_url = data["species"]["url"]

    return {
        'Name': data['name'],
        'Base Stats': stats,
        'Types': types,
        'Moves': moves,
        'Abilities': abilities,
        "species_url": species_url
    }

def get_move_data(move_name):
    """
    Fetch detailed move data from PokéAPI with name normalization.
    """
    import re
    base_url = "https://pokeapi.co/api/v2/move/"

    # Normalize name: lowercase, hyphens instead of spaces, remove apostrophes/periods
    normalized_name = move_name.strip().lower()
    normalized_name = normalized_name.replace(" ", "-").replace("'", "").replace(".", "")
    normalized_name = re.sub(r"[^\w\-]", "", normalized_name)

    url = f"{base_url}{normalized_name}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        move = response.json()
        return {
            "name": move["name"],
            "type": move["type"]["name"],
            "power": move["power"],
            "accuracy": move["accuracy"],
            "pp": move["pp"],
            "damage_class": move["damage_class"]["name"],
            "effect_chance": move.get("effect_chance"),
            "short_effect": (
                move["effect_entries"][0]["short_effect"]
                if move.get("effect_entries") else "No effect info."
            ),
            "meta": move.get("meta", {})
        }
    except Exception as e:
        print(f"⚠️ Could not retrieve move data for '{move_name}': {e}")
        return None


def get_nature_details(nature_name):
    """
    Fetches nature details from the PokeAPI.

    Args:
        nature_name (str): The name of the nature.

    Returns:
        dict: A dictionary containing increased and decreased stats, or None if not found.
    """
    url = f"https://pokeapi.co/api/v2/nature/{nature_name.lower()}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        increased_stat = data["increased_stat"]["name"] if data["increased_stat"] else None
        decreased_stat = data["decreased_stat"]["name"] if data["decreased_stat"] else None
        return {"increased": increased_stat, "decreased": decreased_stat}
    else:
        print(f"Nature details not found for {nature_name}")
        return None

def get_item_data(item_name):
    url = f"https://pokeapi.co/api/v2/item/{item_name}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            "name": data["name"],
            "category": data["category"]["name"],
            "effect": data["effect_entries"][0]["effect"] if data["effect_entries"] else "",
            "attributes": [attr["name"] for attr in data["attributes"]]
        }
    else:
        return None

def get_type_effectiveness_data():
    """
    Gets type effectiveness data from the PokeAPI.

    Returns:
        dict: A dictionary where keys are attacking type names and values are
              dictionaries containing damage multipliers against other types.
              Returns an empty dictionary if data cannot be fetched.
    """
    type_effectiveness = {}
    url = "https://pokeapi.co/api/v2/type/"
    response = requests.get(url)

    if response.status_code != 200:
        print("Could not fetch list of types.")
        return type_effectiveness

    types_data = response.json()
    type_urls = [t['url'] for t in types_data['results']]

    for type_url in type_urls:
        type_response = requests.get(type_url)
        if type_response.status_code == 200:
            type_details = type_response.json()
            type_name = type_details['name']
            damage_relations = type_details['damage_relations']

            effectiveness_for_type = {}
            for rel_type in damage_relations['double_damage_to']:
                effectiveness_for_type[rel_type['name']] = 2.0
            for rel_type in damage_relations['half_damage_to']:
                effectiveness_for_type[rel_type['name']] = 0.5
            for rel_type in damage_relations['no_damage_to']:
                effectiveness_for_type[rel_type['name']] = 0.0

            type_effectiveness[type_name] = effectiveness_for_type
        else:
            print(f"Could not fetch details for type: {type_url}")

    return type_effectiveness

def get_type_effectiveness(attacking_type, defender_types):
    """
    Calculates the overall damage multiplier based on attacking and defending types.

    Args:
        attacking_type (str): The name of the attacking type.
        defender_types (list): A list of names of the defending types.

    Returns:
        float: The overall damage multiplier.
    """
    total_effectiveness = 1.0

    if attacking_type not in type_effectiveness_data:
        return total_effectiveness

    effectiveness_for_attacking_type = type_effectiveness_data[attacking_type]

    for defender_type in defender_types:
        if defender_type in effectiveness_for_attacking_type:
            total_effectiveness *= effectiveness_for_attacking_type[defender_type]

    return total_effectiveness

def calculate_hp(base_hp, iv, ev, level):
    """Calculates a Pokémon's actual HP stat."""
    return int(((2 * base_hp + iv + int(ev/4)) * level) / 100) + level + 10

def calculate_stat(base_stat, iv, ev, level, nature_mod):
    """Calculates a Pokémon's actual non-HP stat."""
    return int((((2 * base_stat + iv + int(ev/4)) * level) / 100) + 5) * nature_mod

# Define item multipliers
item_multipliers = {
    "choice-band": {"stat": "attack", "multiplier": 1.5},
    "choice-specs": {"stat": "special-attack", "multiplier": 1.5},
    "choice-scarf": {"stat": "speed", "multiplier": 1.5},
    "life-orb": {"stat": "damage", "multiplier": 1.3},
    "muscle-band": {"stat": "attack", "multiplier": 1.1},
    "wise-glasses": {"stat": "special-attack", "multiplier": 1.1},
    "assault-vest": {"stat": "special-defense", "multiplier": 1.5},
    "quick-claw": {"stat": "priority", "multiplier": 1.0},
    "light-clay": {"stat": "screens", "multiplier": 1.0},
}

# Get type effectiveness
type_effectiveness_data = get_type_effectiveness_data()

#get all names to align
def get_all_pokemon_names(limit=10000):
    url = f"https://pokeapi.co/api/v2/pokemon?limit={limit}"
    response = requests.get(url).json()
    names = [pokemon['name'] for pokemon in response['results']]
    return names

all_pokemon_names = get_all_pokemon_names()

all_pokemon_data = []
for name in all_pokemon_names:
    pokemon_data = get_pokemon_data(name)
    if pokemon_data:
        all_pokemon_data.append(pokemon_data)

def get_generation_from_species_url(species_url):
    response = requests.get(species_url)
    if response.status_code != 200:
        print(f"Generation info not found for species: {species_url}")
        return None

    data = response.json()
    return data["generation"]["name"]

gen_map = {
    "generation-i": "Gen I",
    "generation-ii": "Gen II",
    "generation-iii": "Gen III",
    "generation-iv": "Gen IV",
    "generation-v": "Gen V",
    "generation-vi": "Gen VI",
    "generation-vii": "Gen VII",
    "generation-viii": "Gen VIII",
    "generation-ix": "Gen IX"
}

for pokemon in all_pokemon_data:
    pokemon["generation"] = get_generation_from_species_url(pokemon["species_url"])
    pokemon["generation"] = gen_map.get(pokemon["generation"], pokemon["generation"])
    pokemon["Total Base Stats"] = sum(pokemon["Base Stats"].values())

display(all_pokemon_data[0])

#defines battle simulation data
def prepare_pokemon_for_battle(pokemon_data, level, selected_moves, selected_nature, selected_item, iv_dict, ev_dict):
    if not pokemon_data:
        print("Error: pokemon_data is None.")
        return None

    base_stats = pokemon_data.get('Base Stats')
    if not base_stats:
        print(f"Error: 'Base Stats' not found for {pokemon_data.get('Name', 'Unknown Pokemon')}.")
        return None

    nature_details = get_nature_details(selected_nature) if selected_nature != "Select Nature" else None

    calculated_stats = {
        'hp': calculate_hp(base_stats.get('hp', 0), iv_dict.get('hp', 0), ev_dict.get('hp', 0), level)
    }

    stat_names = ['attack', 'defense', 'special-attack', 'special-defense', 'speed']
    for stat in stat_names:
        nature_mod = 1.0
        if nature_details:
            if nature_details.get('increased') == stat:
                nature_mod = 1.1
            elif nature_details.get('decreased') == stat:
                nature_mod = 0.9
        calculated_stats[stat] = calculate_stat(base_stats.get(stat, 0), iv_dict.get(stat, 0), ev_dict.get(stat, 0), level, nature_mod)


    # Fetch details for selected moves
    move_details = [get_move_data(move_name) for move_name in selected_moves]
    move_details = [move for move in move_details if move] # Filter out any moves not found


    # Apply item effects to calculated stats if applicable
    effective_stats = calculated_stats.copy()
    if selected_item != "None" and selected_item in item_multipliers:
        item_effect = item_multipliers[selected_item]
        if item_effect["stat"] in effective_stats:
             effective_stats[item_effect["stat"]] = int(effective_stats[item_effect["stat"]] * item_effect["multiplier"])
        # Damage and priority modifiers are handled in damage calculation/turn order

    # Get abilities from the provided pokemon_data
    abilities = pokemon_data.get('Abilities', [])

    return {
        'Name': pokemon_data['Name'],
        'Types': pokemon_data['Types'],
        'Base Stats': base_stats, # Keep base stats for reference
        'Calculated Stats': calculated_stats, # Stats before item
        'Effective Stats': effective_stats, # Stats after item
        'Moves': move_details,
        'Level': level,
        'Nature': selected_nature,
        'Item': selected_item,
        'IVs': iv_dict,
        'EVs': ev_dict,
        'current_hp': calculated_stats['hp'], # Start with full HP
        'Abilities': abilities # Include abilities
    }

#calculate damage
def calculate_damage(attacker, defender, move, type_effectiveness_data):
    """
    Calculates the damage a move deals, considering stats, types, and modifiers.
    Simplified damage calculation based on typical in-game mechanics (Gen 5+).
    """
    if not move or move.get('power') is None:
        return 0  # Cannot calculate damage without a valid move with power

    attack_stat = attacker['Effective Stats']['attack'] if move.get('damage_class') == 'physical' else attacker['Effective Stats']['special-attack']
    defense_stat = defender['Effective Stats']['defense'] if move.get('damage_class') == 'physical' else defender['Effective Stats']['special-defense']
    move_power = move['power']
    attacker_level = attacker['Level']
    move_type = move['type']
    defender_types = defender['Types']

    # Base Damage Calculation
    # Formula: (((2 * Level / 5 + 2) * AttackStat * MovePower / DefenseStat) / 50 + 2) * Modifiers
    damage = ((((2 * attacker_level / 5 + 2) * attack_stat * move_power / defense_stat) / 50) + 2)

    # Modifiers: STAB (Same Type Attack Bonus), Type Effectiveness, Crit, Random, Other
    stab = 1.5 if move_type in attacker['Types'] else 1.0
    type_eff = get_type_effectiveness(move_type, defender_types)

    # Random factor (0.85 to 1.0)
    random_factor = random.uniform(0.85, 1.0)

    # Item modifier (Life Orb, etc.)
    item_multiplier = 1.0
    if attacker['Item'] in item_multipliers:
        item_effect = item_multipliers[attacker['Item']]
        if item_effect['stat'] == 'damage':
            item_multiplier = item_effect['multiplier']

    final_damage = damage * stab * type_eff * random_factor * item_multiplier

    # Ensure minimum damage is 1 if the move has power and is not 0 effectiveness
    if move_power > 0 and type_eff > 0 and int(final_damage) == 0:
        final_damage = 1
    elif type_eff == 0:
        final_damage = 0 # Ensure 0 damage if type effectiveness is 0


    return int(final_damage)

#Simulate battle turns
def simulate_battle_turn(pokemon1, pokemon2, type_effectiveness_data):
    """
    Simulates a single turn of a Pokémon battle.
    Determines who attacks first based on speed (and potential priority moves/items),
    calculates damage, and updates HP.
    Returns a tuple: (description of the turn, winner or None if battle continues)
    """
    turn_log = []
    winner = None

    # Determine turn order based on Speed (and eventually move priority, items)
    # Simplistic speed comparison for now
    pokemon1_speed = pokemon1['Effective Stats']['speed']
    pokemon2_speed = pokemon2['Effective Stats']['speed']

    # Select a random available move for each Pokémon
    move1 = random.choice(pokemon1['Moves']) if pokemon1['Moves'] else None
    move2 = random.choice(pokemon2['Moves']) if pokemon2['Moves'] else None

    # Basic turn order: faster Pokémon attacks first. Tie goes to random.
    if pokemon1_speed > pokemon2_speed:
        first_pokemon, second_pokemon = pokemon1, pokemon2
        first_move, second_move = move1, move2
    elif pokemon2_speed > pokemon1_speed:
        first_pokemon, second_pokemon = pokemon2, pokemon1
        first_move, second_move = move2, move1
    else: # Speed tie
        if random.random() < 0.5:
            first_pokemon, second_pokemon = pokemon1, pokemon2
            first_move, second_move = move1, move2
        else:
            first_pokemon, second_pokemon = pokemon2, pokemon1
            first_move, second_move = move2, move1

    # --- First Pokémon attacks ---
    if first_move:
        damage_to_second = calculate_damage(first_pokemon, second_pokemon, first_move, type_effectiveness_data)
        second_pokemon['current_hp'] -= damage_to_second
        turn_log.append(f"{first_pokemon['Name'].capitalize()} used {first_move['name'].capitalize()}!")
        turn_log.append(f"{second_pokemon['Name'].capitalize()} took {damage_to_second} damage. (HP: {max(0, second_pokemon['current_hp'])}/{second_pokemon['Calculated Stats']['hp']})")

        if second_pokemon['current_hp'] <= 0:
            turn_log.append(f"{second_pokemon['Name'].capitalize()} fainted!")
            winner = first_pokemon['Name']
            return turn_log, winner

    # --- Second Pokémon attacks (if not fainted) ---
    if second_pokemon['current_hp'] > 0 and second_move:
        damage_to_first = calculate_damage(second_pokemon, first_pokemon, second_move, type_effectiveness_data)
        first_pokemon['current_hp'] -= damage_to_first
        turn_log.append(f"{second_pokemon['Name'].capitalize()} used {second_move['name'].capitalize()}!")
        turn_log.append(f"{first_pokemon['Name'].capitalize()} took {damage_to_first} damage. (HP: {max(0, first_pokemon['current_hp'])}/{first_pokemon['Calculated Stats']['hp']})")

        if first_pokemon['current_hp'] <= 0:
            turn_log.append(f"{first_pokemon['Name'].capitalize()} fainted!")
            winner = second_pokemon['Name']
            return turn_log, winner

    return turn_log, winner

#Full team battles
def simulate_battle(team1, team2, type_effectiveness_data, max_turns=100):
    """
    Simulates a full battle between two teams.
    Teams are lists of battle-ready Pokémon dictionaries.
    Returns a tuple: (winning team, battle_log)
    """
    team1_current = [p.copy() for p in team1] # Create copies to avoid modifying original data
    team2_current = [p.copy() for p in team2]

    battle_log = [] # Initialize battle log

    team1_active_index = 0
    team2_active_index = 0

    for turn in range(max_turns):
        if team1_active_index >= len(team1_current) or team2_active_index >= len(team2_current):
             # One team is out of Pokémon before max_turns reached
             break

        battle_log.append(f"\n--- Turn {turn + 1} ---")

        pokemon1 = team1_current[team1_active_index]
        pokemon2 = team2_current[team2_active_index]

        # Simulate a turn between the active Pokémon
        turn_log, winner = simulate_battle_turn(pokemon1, pokemon2, type_effectiveness_data)
        battle_log.extend(turn_log) # Add battle log play- by- play

        # Check for fainted Pokémon and switch if necessary
        if pokemon1['current_hp'] <= 0:
            battle_log.append(f"{pokemon1['Name'].capitalize()} fainted!")
            team1_active_index += 1
            if team1_active_index < len(team1_current):
                 battle_log.append(f"Team 1 sent out {team1_current[team1_active_index]['Name'].capitalize()}!")
            else:
                 battle_log.append("Team 1 is out of Pokémon!")
                 return 'Team 2', battle_log # Team 1 is out of Pokémon


        if pokemon2['current_hp'] <= 0:
            battle_log.append(f"{pokemon2['Name'].capitalize()} fainted!")
            team2_active_index += 1
            if team2_active_index < len(team2_current):
                battle_log.append(f"Team 2 sent out {team2_current[team2_active_index]['Name'].capitalize()}!")
            else:
                 battle_log.append("Team 2 is out of Pokémon!")
                 return 'Team 1', battle_log # Team 2 is out of Pokémon


    # If max_turns reached, check HP totals
    if team1_active_index < len(team1_current) or team2_active_index < len(team2_current):
        # One team still has Pokémon but max turns reached. Winner is the one with remaining Pokémon.
        battle_log.append("\n--- Max turns reached! ---")
        if team1_active_index < len(team1_current) and team2_active_index >= len(team2_current):
             battle_log.append("Team 1 has remaining Pokémon.")
             return 'Team 1', battle_log
        elif team1_active_index >= len(team1_current) and team2_active_index < len(team2_current):
             battle_log.append("Team 2 has remaining Pokémon.")
             return 'Team 2', battle_log
        else: # Both teams still have Pokémon, compare remaining HP
             team1_remaining_hp = sum(p['current_hp'] for p in team1_current[team1_active_index:])
             team2_remaining_hp = sum(p['current_hp'] for p in team2_current[team2_active_index:])
             battle_log.append(f"Team 1 remaining HP: {team1_remaining_hp}")
             battle_log.append(f"Team 2 remaining HP: {team2_remaining_hp}")


             if team1_remaining_hp > team2_remaining_hp:
                 battle_log.append("Team 1 has more remaining HP.")
                 return 'Team 1', battle_log
             elif team2_remaining_hp > team1_remaining_hp:
                 battle_log.append("Team 2 has more remaining HP.")
                 return 'Team 2', battle_log
             else:
                 battle_log.append("Remaining HP is equal.")
                 return 'Draw', battle_log
    else:
         # Both teams are out of Pokémon simultaneously (unlikely in this simplified model but possible)
         battle_log.append("Both teams ran out of Pokémon.")
         return 'Draw', battle_log

#offensive coverage
def analyze_offensive_coverage(team, type_effectiveness_data):
    """
    Analyzes the offensive type coverage of a team's moves.
    Returns a dictionary indicating how effectively the team's moves hit each type.
    """
    coverage = {}
    all_types = list(type_effectiveness_data.keys()) # Get all possible types

    for target_type in all_types:
        max_effectiveness = 0.0
        attacking_types = set() # Keep track of unique attacking types

        for pokemon in team:
            for move in pokemon.get('Moves', []):
                move_type = move.get('type')
                if move_type in type_effectiveness_data:
                    effectiveness = get_type_effectiveness(move_type, [target_type])
                    if effectiveness > max_effectiveness:
                        max_effectiveness = effectiveness
                        attacking_types.add(move_type) # Add the type that achieved this effectiveness

        coverage[target_type] = {'max_effectiveness': max_effectiveness, 'attacking_types': list(attacking_types)}

    return coverage

#Defensive synergy
def analyze_defensive_synergy(team, type_effectiveness_data):
    """
    Analyzes the defensive synergy of a team, considering weaknesses and resistances.
    Returns a dictionary indicating the minimum damage multiplier the team takes from each type.
    """
    defensive_synergy = {}
    all_types = list(type_effectiveness_data.keys()) # Get all possible types

    for attacking_type in all_types:
        min_multiplier = 4.0 # Start high, assuming a 4x weakness is possible
        resisting_pokemon = [] # List of Pokémon that resist this type

        for pokemon in team:
            pokemon_types = pokemon.get('Types', [])
            effectiveness = get_type_effectiveness(attacking_type, pokemon_types)

            if effectiveness < min_multiplier:
                min_multiplier = effectiveness
                resisting_pokemon = [pokemon['Name']] # Start a new list of resisting Pokémon
            elif effectiveness == min_multiplier and effectiveness < 1.0:
                 # Add Pokémon if they share the minimum (resistance/immunity)
                 resisting_pokemon.append(pokemon['Name'])


        defensive_synergy[attacking_type] = {'min_multiplier': min_multiplier, 'resisting_pokemon': list(set(resisting_pokemon))} # Use set to get unique names

    return defensive_synergy

#Generating team synergy report
def generate_team_synergy_report(team, type_effectiveness_data, level):
    """
    Generates a markdown report analyzing the offensive coverage and defensive synergy of a team.
    Includes details about each Pokémon's stats, moves, abilities, nature, and item.
    Includes a summarized type synergy analysis.
    """
    report = []
    report.append(f"## Team Synergy Report (Level {level})")
    report.append("") # Add a newline

    report.append("### Team Members:")
    for pokemon in team:
        report.append(f"- **{pokemon['Name'].capitalize()}** (Types: {', '.join([t.capitalize() for t in pokemon['Types']])})")
        report.append(f"  - Nature: {pokemon['Nature']}, Item: {pokemon['Item'].replace('-', ' ').title()}")
        report.append(f"  - Abilities: {', '.join([a.replace('-', ' ').title() for a in pokemon.get('Abilities', [])])}")
        report.append(f"  - Stats: HP: {pokemon['Calculated Stats']['hp']}, Atk: {pokemon['Calculated Stats']['attack']}, Def: {pokemon['Calculated Stats']['defense']}, SpA: {pokemon['Calculated Stats']['special-attack']}, SpD: {pokemon['Calculated Stats']['special-defense']}, Spe: {pokemon['Calculated Stats']['speed']}")
        report.append(f"  - Effective Stats (with item): HP: {pokemon['Effective Stats']['hp']}, Atk: {pokemon['Effective Stats']['attack']}, Def: {pokemon['Effective Stats']['defense']}, SpA: {pokemon['Effective Stats']['special-attack']}, SpD: {pokemon['Effective Stats']['special-defense']}, Spe: {pokemon['Effective Stats']['speed']}")
        report.append(f"  - Moves:")
        if pokemon['Moves']:
            for move in pokemon['Moves']:
                report.append(f"    - {move['name'].capitalize()} (Type: {move['type'].capitalize()}, Power: {move['power'] or 'N/A'}, Accuracy: {move['accuracy'] or 'N/A'}, Damage Class: {move['damage_class'].capitalize()})")
        else:
            report.append("    - No moves selected.")
        report.append("")

    report.append("### Offensive Coverage Analysis:")
    offensive_coverage = analyze_offensive_coverage(team, type_effectiveness_data)
    if offensive_coverage:
        report.append("| Attacking Type | Max Effectiveness | Attacking Pokémon Types |")
        report.append("|---|---|---|")
        for target_type in sorted(offensive_coverage.keys()):
            coverage_info = offensive_coverage[target_type]
            max_eff = coverage_info['max_effectiveness']
            attacking_types = ", ".join([t.capitalize() for t in coverage_info['attacking_types']]) or "None"
            report.append(f"| {target_type.capitalize()} | {max_eff:.1f}x | {attacking_types} |")
    else:
        report.append("Could not perform offensive coverage analysis.")
    report.append("")


    report.append("### Defensive Synergy Analysis:")
    defensive_synergy = analyze_defensive_synergy(team, type_effectiveness_data)
    if defensive_synergy:
        report.append("| Attacking Type | Minimum Damage Multiplier | Resisting Pokémon |")
        report.append("|---|---|---|")
        for attacking_type in sorted(defensive_synergy.keys()):
            synergy_info = defensive_synergy[attacking_type]
            min_mult = synergy_info['min_multiplier']
            resisting_pokemon = ", ".join([p.capitalize() for p in synergy_info['resisting_pokemon']]) or "None"
            report.append(f"| {attacking_type.capitalize()} | {min_mult:.1f}x | {resisting_pokemon} |")
    else:
        report.append("Could not perform defensive synergy analysis.")
    report.append("")

    #Summarized Type Synergy Analysis
    report.append("--- Team Synergy Report ---")
    report.append("\n1. Type Synergy Analysis:")

    # Prepare data for offensive summary
    super_effective_4x = []
    super_effective_2x = []
    for target_type, coverage_info in offensive_coverage.items():
        if coverage_info['max_effectiveness'] >= 4.0:
            super_effective_4x.append(target_type.capitalize())
        elif coverage_info['max_effectiveness'] >= 2.0:
            super_effective_2x.append(target_type.capitalize())

    # Offensive Coverage Summary
    report.append("\n  Offensive Coverage:")
    if super_effective_4x:
        report.append(f"    - Can hit for 4x super effective damage against: {', '.join(super_effective_4x)}")
    if super_effective_2x:
        report.append(f"    - Can hit for 2x super effective damage against: {', '.join(super_effective_2x)}")
    if not super_effective_4x and not super_effective_2x:
         report.append("    - Limited super effective coverage based on the analyzed moves.")


    # Prepare data for defensive summary
    weaknesses_by_type = {}
    resistances_by_type = {}
    immunities_by_type = {}

    for attacking_type, synergy_info in defensive_synergy.items():
        min_mult = synergy_info['min_multiplier']
        resisting_pokemon = synergy_info['resisting_pokemon']

        if min_mult >= 2.0:
             weaknesses_by_type[attacking_type.capitalize()] = resisting_pokemon


#individual matchups
    individual_weaknesses = {}
    individual_resistances = {}
    individual_immunities = {}

    for attacking_type in type_effectiveness_data.keys():
        weak_pokemon = []
        resistant_pokemon = []
        immune_pokemon = []
        for pokemon in team:
            effectiveness = get_type_effectiveness(attacking_type, pokemon.get('Types', []))
            if effectiveness >= 2.0:
                weak_pokemon.append(pokemon['Name'].capitalize())
            elif effectiveness <= 0.5 and effectiveness > 0:
                resistant_pokemon.append(pokemon['Name'].capitalize())
            elif effectiveness == 0:
                immune_pokemon.append(pokemon['Name'].capitalize())

        if weak_pokemon:
            individual_weaknesses[attacking_type.capitalize()] = list(set(weak_pokemon)) # Use set for unique names
        if resistant_pokemon:
             individual_resistances[attacking_type.capitalize()] = list(set(resistant_pokemon))
        if immune_pokemon:
             individual_immunities[attacking_type.capitalize()] = list(set(immune_pokemon))


    # Defensive Matchups Summary
    report.append("\n  Defensive Matchups:")
    if individual_weaknesses:
        report.append("    - Weaknesses (2x or 4x damage received):")
        for type_, pokemon_list in sorted(individual_weaknesses.items()):
            report.append(f"      - {type_}: {', '.join(pokemon_list)}")
    else:
        report.append("    - No notable type weaknesses identified.")

    if individual_resistances:
        report.append("    - Resistances (0.5x or 0.25x damage received):")
        for type_, pokemon_list in sorted(individual_resistances.items()):
            report.append(f"      - {type_}: {', '.join(pokemon_list)}")
    else:
        report.append("    - No notable type resistances identified.")

    if individual_immunities:
        report.append("    - Immunities (0x damage received):")
        for type_, pokemon_list in sorted(individual_immunities.items()):
            report.append(f"      - {type_}: {', '.join(pokemon_list)}")
    else:
        report.append("    - No notable type immunities identified.")


    # Convert the list of strings into a single markdown string
    return "\n".join(report)

#Simulate battles and predicts outcome
def predict_battle_outcome(team1, team2, level, num_simulations, type_effectiveness_data):
    team1_wins = 0
    team2_wins = 0
    draws = 0
    team1_remaining_hp_total = 0
    team2_remaining_hp_total = 0
    team1_pokemon_remaining_total = 0
    team2_pokemon_remaining_total = 0


    for _ in range(num_simulations):
        # Create fresh copies of the teams for each simulation
        team1_sim = [p.copy() for p in team1]
        team2_sim = [p.copy() for p in team2]

        winner, battle_log = simulate_battle(team1_sim, team2_sim, type_effectiveness_data)

        if winner == 'Team 1':
            team1_wins += 1
            # Calculate remaining HP and Pokémon for the winning team
            team1_remaining_hp_total += sum(p['current_hp'] for p in team1_sim if p['current_hp'] > 0)
            team1_pokemon_remaining_total += sum(1 for p in team1_sim if p['current_hp'] > 0)

        elif winner == 'Team 2':
            team2_wins += 1
            # Calculate remaining HP and Pokémon for the winning team
            team2_remaining_hp_total += sum(p['current_hp'] for p in team2_sim if p['current_hp'] > 0)
            team2_pokemon_remaining_total += sum(1 for p in team2_sim if p['current_hp'] > 0)
        else:
            draws += 1
            # For draws, you might sum remaining HP/Pokémon for both or handle differently
            team1_remaining_hp_total += sum(p['current_hp'] for p in team1_sim if p['current_hp'] > 0)
            team1_pokemon_remaining_total += sum(1 for p in team1_sim if p['current_hp'] > 0)
            team2_remaining_hp_total += sum(p['current_hp'] for p in team2_sim if p['current_hp'] > 0)
            team2_pokemon_remaining_total += sum(1 for p in team2_sim if p['current_hp'] > 0)


    total_simulations = num_simulations
    team1_win_percentage = (team1_wins / total_simulations) * 100
    team2_win_percentage = (team2_wins / total_simulations) * 100
    draw_percentage = (draws / total_simulations) * 100

    # Determine predicted outcome
    if team1_wins > team2_wins and team1_wins > draws:
        predicted_outcome = 'Team 1 Wins'
    elif team2_wins > team1_wins and team2_wins > draws:
        predicted_outcome = 'Team 2 Wins'
    elif team1_wins == team2_wins:
        predicted_outcome = 'Likely Draw or Close Match' # More descriptive for ties
    else:
        predicted_outcome = 'Draw'


    # Calculate average remaining HP and Pokémon for winning teams (even in draws
    avg_team1_remaining_hp = team1_remaining_hp_total / (team1_wins + draws) if (team1_wins + draws) > 0 else 0
    avg_team2_remaining_hp = team2_remaining_hp_total / (team2_wins + draws) if (team2_wins + draws) > 0 else 0
    avg_team1_pokemon_remaining = team1_pokemon_remaining_total / (team1_wins + draws) if (team1_wins + draws) > 0 else 0
    avg_team2_pokemon_remaining = team2_pokemon_remaining_total / (team2_wins + draws) if (team2_wins + draws) > 0 else 0


    # Generate synergy reports for the *initial* teams
    team1_synergy_report_markdown = generate_team_synergy_report(team1, type_effectiveness_data, level)
    team2_synergy_report_markdown = generate_team_synergy_report(team2, type_effectiveness_data, level)

    # Calculate synergy scores for the initial teams
    team1_offensive_coverage = analyze_offensive_coverage(team1, type_effectiveness_data)
    team1_defensive_synergy = analyze_defensive_synergy(team1, type_effectiveness_data)
    team1_synergy_score = calculate_team_synergy_score(team1_offensive_coverage, team1_defensive_synergy, team1)

    team2_offensive_coverage = analyze_offensive_coverage(team2, type_effectiveness_data)
    team2_defensive_synergy = analyze_defensive_synergy(team2, type_effectiveness_data)
    team2_synergy_score = calculate_team_synergy_score(team2_offensive_coverage, team2_defensive_synergy, team2)


    return {
        'predicted_outcome': predicted_outcome,
        'team1_wins': team1_wins,
        'team2_wins': team2_wins,
        'draws': draws,
        'team1_win_percentage': team1_win_percentage,
        'team2_win_percentage': team2_win_percentage,
        'draw_percentage': draw_percentage,
        'avg_team1_remaining_hp': avg_team1_remaining_hp,
        'avg_team2_remaining_hp': avg_team2_remaining_hp,
        'avg_team1_pokemon_remaining': avg_team1_pokemon_remaining,
        'avg_team2_pokemon_remaining': avg_team2_pokemon_remaining,
        'team1_synergy_report': team1_synergy_report_markdown,
        'team2_synergy_report': team2_synergy_report_markdown,
        'team1_synergy_score': team1_synergy_score,
        'team2_synergy_score': team2_synergy_score
    }

#creation of synergy score
def calculate_team_synergy_score(offensive_coverage, defensive_synergy, team):
    """
    Calculates a numerical synergy score for a team based on offensive coverage,
    defensive synergy, and other potential factors like speed distribution.
    Higher score indicates better synergy.
    This is a refined scoring mechanism.
    """
    score = 0.0

    # Offensive Score
    # Reward having good offensive coverage (hitting types for super effective damage)
    # Penalize having poor coverage (hitting types for not very effective or no damage)
    offensive_score = 0.0
    for target_type, coverage_info in offensive_coverage.items():
        max_eff = coverage_info['max_effectiveness']
        if max_eff >= 4.0:
            offensive_score += 4.0 # Increased reward for 4x super effective
        elif max_eff >= 2.0:
            offensive_score += 2.5 # Increased reward for 2x super effective
        elif max_eff == 1.0:
            offensive_score += 0.5 # Small reward for neutral coverage
        elif max_eff <= 0.5 and max_eff > 0:
             offensive_score -= 1.5 # Increased penalty for not very effective
        elif max_eff == 0:
             offensive_score -= 3.0 # Higher penalty for no damage

    # Defensive Score
    # Reward having resistances and immunities
    # Penalize having weaknesses
    defensive_score = 0.0
    for attacking_type, synergy_info in defensive_synergy.items():
        min_mult = synergy_info['min_multiplier']
        if min_mult == 0.0:
            defensive_score += 4.0 # Increased reward for immunity
        elif min_mult <= 0.5 and min_mult > 0:
            defensive_score += 2.5 # Increased reward for resistance
        elif min_mult >= 2.0:
            defensive_score -= 3.0 # Increased penalty for weakness (2x or 4x)

    # Speed Synergy
    # Reward teams with a good distribution of speed tiers or high overall speed
    speed_score = 0.0
    speeds = [p['Effective Stats']['speed'] for p in team]
    if speeds:
        avg_speed = sum(speeds) / len(speeds)
        speed_score += avg_speed * 0.1 # Reward higher average speed

        # Reward having a mix of fast and slow Pokémon (might indicate speed control options)
        # This is a simple approach; more complex analysis could look at speed tiers.
        speed_variance = pd.Series(speeds).var() if len(speeds) > 1 else 0
        speed_score += speed_variance * 0.01 # Reward higher variance (up to a point)


    #Ability should be added here for dashboard
    ability_score = 0.0

    score = (offensive_score * 0.3) + (defensive_score * 0.5) + (speed_score * 0.2) # Adjusted weights


    return score

import ipywidgets as widgets
from IPython.display import display, clear_output, Markdown
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pprint

#importing natures

# Get a list of all pokemon names from the loaded data
if 'all_pokemon_data' in globals() and all_pokemon_data:
    all_pokemon_names = [p['Name'] for p in all_pokemon_data]
else:
    all_pokemon_names = []
    print("Warning: all_pokemon_data not found or empty. Pokémon dropdowns will be empty.")


# Define a list of common competitive natures (can be expanded)
common_natures = [
    "Adamant", "Modest", "Jolly", "Timid", "Brave", "Quiet", "Relaxed",
    "Sassy", "Bold", "Impish", "Careful", "Calm", "Gentle",
    "Naive", "Hasty", "Lax", "Naughty", "Lonely", "Hardy", "Docile", "Bashful",
    "Quirky", "Serious"
]

# Extract item names from the item_multipliers dictionary
if 'item_multipliers' in globals():
    held_item_options = ["None"] + sorted(list(item_multipliers.keys()))
else:
    held_item_options = ["None"]
    print("Warning: item_multipliers not found. Item dropdowns will only contain 'None'.")


# Define stat names for EV/IV inputs
stat_names = ['HP', 'Attack', 'Defense', 'Special-attack', 'Special-defense', 'Speed']

# Create nested lists of widgets for Team 1 and Team 2
team1_pokemon_widgets = []
for i in range(6):
    pokemon_dropdown = widgets.Dropdown(options=["Select Pokémon"] + sorted(all_pokemon_names), description=f'Team 1 Pokémon {i+1}:', disabled=False)
    move_dropdowns = [
        widgets.Dropdown(options=["Select Move"], description=f'Move {j+1}:', disabled=True) for j in range(4)
    ]
    nature_dropdown = widgets.Dropdown(options=["Select Nature"] + common_natures, description='Nature:', disabled=False)
    item_dropdown = widgets.Dropdown(options=held_item_options, description='Item:', disabled=False)

    # Create EV and IV input fields for each stat
    ev_inputs = [widgets.IntText(value=0, description=f'{stat} EV:', disabled=False, min=0, max=252) for stat in stat_names]
    iv_inputs = [widgets.IntText(value=31, description=f'{stat} IV:', disabled=False, min=0, max=31) for stat in stat_names]

    # Group EV and IV inputs
    ev_box = widgets.VBox(ev_inputs, layout=widgets.Layout(border='1px solid lightgrey', padding='10px', margin='5px'))
    iv_box = widgets.VBox(iv_inputs, layout=widgets.Layout(border='1px solid lightgrey', padding='10px', margin='5px'))

    # Create a box for stat inputs (EVs and IVs side by side)
    stat_inputs_box = widgets.HBox([ev_box, iv_box])


    pokemon_box = widgets.VBox([pokemon_dropdown] + move_dropdowns + [nature_dropdown, item_dropdown, widgets.Label("Stats (EVs/IVs):"), stat_inputs_box], layout=widgets.Layout(border='2px solid blue', padding='10px', margin='10px'))
    team1_pokemon_widgets.append({
        'pokemon_dropdown': pokemon_dropdown,
        'move_dropdowns': move_dropdowns,
        'nature_dropdown': nature_dropdown,
        'item_dropdown': item_dropdown,
        'ev_inputs': ev_inputs,
        'iv_inputs': iv_inputs,
        'box': pokemon_box
    })

team2_pokemon_widgets = []
for i in range(6):
    pokemon_dropdown = widgets.Dropdown(options=["Select Pokémon"] + sorted(all_pokemon_names), description=f'Team 2 Pokémon {i+1}:', disabled=False)
    move_dropdowns = [
        widgets.Dropdown(options=["Select Move"], description=f'Move {j+1}:', disabled=True) for j in range(4)
    ]
    nature_dropdown = widgets.Dropdown(options=["Select Nature"] + common_natures, description='Nature:', disabled=False)
    item_dropdown = widgets.Dropdown(options=held_item_options, description='Item:', disabled=False)

    # Create EV and IV input fields for each stat
    ev_inputs = [widgets.IntText(value=0, description=f'{stat} EV:', disabled=False, min=0, max=252) for stat in stat_names]
    iv_inputs = [widgets.IntText(value=31, description=f'{stat} IV:', disabled=False, min=0, max=31) for stat in stat_names]

     # Group EV and IV inputs
    ev_box = widgets.VBox(ev_inputs, layout=widgets.Layout(border='1px solid lightgrey', padding='10px', margin='5px'))
    iv_box = widgets.VBox(iv_inputs, layout=widgets.Layout(border='1px solid lightgrey', padding='10px', margin='5px'))

    # Create a box for stat inputs (EVs and IVs side by side)
    stat_inputs_box = widgets.HBox([ev_box, iv_box])


    pokemon_box = widgets.VBox([pokemon_dropdown] + move_dropdowns + [nature_dropdown, item_dropdown, widgets.Label("Stats (EVs/IVs):"), stat_inputs_box], layout=widgets.Layout(border='2px solid blue', padding='10px', margin='10px'))
    team2_pokemon_widgets.append({
        'pokemon_dropdown': pokemon_dropdown,
        'move_dropdowns': move_dropdowns,
        'nature_dropdown': nature_dropdown,
        'item_dropdown': item_dropdown,
        'ev_inputs': ev_inputs,
        'iv_inputs': iv_inputs,
        'box': pokemon_box
    })


# Function to update move dropdowns when a Pokémon is selected
def update_move_dropdowns(change, move_dropdowns):
    selected_pokemon_name = change['new']
    if selected_pokemon_name != "Select Pokémon":
        pokemon_data = next((p for p in all_pokemon_data if p["Name"] == selected_pokemon_name), None)
        if pokemon_data and 'Moves' in pokemon_data:
            available_moves = sorted(pokemon_data['Moves'])
            move_options = ["Select Move"] + available_moves
            for move_dropdown in move_dropdowns:
                move_dropdown.options = move_options
                move_dropdown.disabled = False
                move_dropdown.value = "Select Move" # Reset value
        else:
            for move_dropdown in move_dropdowns:
                move_dropdown.options = ["Select Move"]
                move_dropdown.disabled = True
                move_dropdown.value = "Select Move"
    else:
        for move_dropdown in move_dropdowns:
            move_dropdown.options = ["Select Move"]
            move_dropdown.disabled = True
            move_dropdown.value = "Select Move"


# Link the update function to each Pokémon dropdown
for team_widgets in [team1_pokemon_widgets, team2_pokemon_widgets]:
    for pokemon_widgets in team_widgets:
        pokemon_widgets['pokemon_dropdown'].observe(
            lambda change, move_dropdowns=pokemon_widgets['move_dropdowns']: update_move_dropdowns(change, move_dropdowns),
            names='value'
        )

# Create buttons and other UI elements
evaluate_button = widgets.Button(
    description='Simulate Battle & Predict',
    disabled=False,
    button_style='info', # 'primary', 'success', 'info', 'warning', 'danger' or ''
    tooltip='Simulate battles and predict outcome',
    icon='play'
)

randomize_button = widgets.Button(
    description='Randomize Teams & Predict',
    disabled=False,
    button_style='success',
    tooltip='Randomly select teams and simulate battles',
    icon='random'
)

general_report_button = widgets.Button(
    description='Generate Synergy Report (Team 1)',
    disabled=False,
    button_style='primary',
    tooltip='Generate a general synergy report for Team 1',
    icon='file-text'
)


level_dropdown = widgets.Dropdown(
    options=[50, 100],
    value=50,
    description='Level:',
    disabled=False,
)

num_simulations_slider = widgets.IntSlider(
    value=100,
    min=10,
    max=1000,
    step=10,
    description='Simulations:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)

output_area_battle = widgets.Output()


# Define the function to handle general synergy report button clicks
def on_general_report_button_clicked(b):
    with output_area_battle:
        clear_output(wait=True)

        # Get selected Pokémon names from dropdowns for Team 1
        team1_names = [dropdown.value for dropdown in [pw['pokemon_dropdown'] for pw in team1_pokemon_widgets] if dropdown.value != "Select Pokémon"]


        if not team1_names:
            print("Please select at least one Pokémon for Team 1 to generate a general report.")
            return

        # Limit the number of Pokémon to 6
        team1_names = team1_names[:6]

        # Get the selected level
        selected_level = level_dropdown.value

        # Fetch the base Pokémon data for the selected names
        team1_data = [get_pokemon_data(name) for name in team1_names]
        team1_data = [p for p in team1_data if p and 'Name' in p and 'Base Stats' in p and 'Types' in p] # Filter out invalid data


        team1_battle_ready = []
        stat_names = ['HP', 'Attack', 'Defense', 'Special-attack', 'Special-defense', 'Speed'] # Define stat_names here

        # Prepare Team 1 (using selected details from UI)
        for i, pokemon_data in enumerate(team1_data):
            pokemon_widget = next((pw for pw in team1_pokemon_widgets if pw['pokemon_dropdown'].value == pokemon_data['Name']), None)

            if pokemon_widget:
                 selected_moves = [md.value for md in pokemon_widget['move_dropdowns'] if md.value != "Select Move"]
                 selected_nature = pokemon_widget['nature_dropdown'].value
                 selected_item = pokemon_widget['item_dropdown'].value

                 # Get selected EVs and IVs from the widgets
                 selected_evs = {stat_names[j].lower(): ev_widget.value for j, ev_widget in enumerate(pokemon_widget['ev_inputs'])}
                 selected_ivs = {stat_names[j].lower(): iv_widget.value for j, iv_widget in enumerate(pokemon_widget['iv_inputs'])}


                 prepared_pokemon = prepare_pokemon_for_battle(
                     pokemon_data,
                     selected_level,
                     selected_moves,
                     selected_nature,
                     selected_item,
                     iv_dict=selected_ivs,
                     ev_dict=selected_evs
                 )
                 if prepared_pokemon:
                     team1_battle_ready.append(prepared_pokemon)
                 else:
                      print(f"Skipping {pokemon_data.get('Name', 'Unknown Pokemon')} due to preparation error.")
            else:
                 print(f"Warning: Could not find widget for selected Pokémon: {pokemon_data['Name']}")


        if not team1_battle_ready:
            print("Could not prepare team for synergy report. Ensure selected Pokémon names are valid.")
            return

        # Generate and display the general synergy report
        synergy_report_markdown = generate_team_synergy_report(team1_battle_ready, type_effectiveness_data, selected_level)
        display(Markdown(synergy_report_markdown))



# Define the function to handle evaluate button clicks
def on_evaluate_button_clicked(b):
    with output_area_battle:
        clear_output(wait=True)

        # Get selected Pokémon names from dropdowns for both teams
        team1_names = [dropdown.value for dropdown in [pw['pokemon_dropdown'] for pw in team1_pokemon_widgets] if dropdown.value != "Select Pokémon"]
        team2_names = [dropdown.value for dropdown in [pw['pokemon_dropdown'] for pw in team2_pokemon_widgets] if dropdown.value != "Select Pokémon"]


        if not team1_names or not team2_names:
            print("Please select at least one Pokémon for each team.")
            return

        # Limit the number of Pokémon to 6 per team
        team1_names = team1_names[:6]
        team2_names = team2_names[:6]

        # Get the selected level and number of simulations
        selected_level = level_dropdown.value
        num_sims = num_simulations_slider.value

        team1_battle_ready = []
        stat_names = ['HP', 'Attack', 'Defense', 'Special-attack', 'Special-defense', 'Speed'] # Define stat_names here

        # Prepare Team 1
        for i, pokemon_widgets in enumerate(team1_pokemon_widgets): # Iterate through widgets to get inputs
            selected_pokemon_name = pokemon_widgets['pokemon_dropdown'].value
            if selected_pokemon_name != "Select Pokémon":
                pokemon_data = next((p for p in all_pokemon_data if p["Name"] == selected_pokemon_name), None)
                if pokemon_data:
                    selected_moves = [md.value for md in pokemon_widgets['move_dropdowns'] if md.value != "Select Move"]
                    selected_nature = pokemon_widgets['nature_dropdown'].value
                    selected_item = pokemon_widgets['item_dropdown'].value

                    # Get selected EVs and IVs from the widgets
                    selected_evs = {stat_names[j].lower(): ev_widget.value for j, ev_widget in enumerate(pokemon_widgets['ev_inputs'])}
                    selected_ivs = {stat_names[j].lower(): iv_widget.value for j, iv_widget in enumerate(pokemon_widgets['iv_inputs'])}

                    prepared_pokemon = prepare_pokemon_for_battle(
                        pokemon_data,
                        selected_level,
                        selected_moves,
                        selected_nature,
                        selected_item,
                        iv_dict=selected_ivs,
                        ev_dict=selected_evs
                    )
                    if prepared_pokemon:
                        team1_battle_ready.append(prepared_pokemon)
                    else:
                         # Handle case where stat calculation failed (e.g. invalid EVs)
                         print(f"Skipping {pokemon_data.get('Name', 'Unknown Pokemon')} due to stat calculation error.")
                else:
                    print(f"Warning: Data not found for Pokémon: {selected_pokemon_name}")


        team2_battle_ready = []
        # Prepare Team 2 (using the same logic)
        for i, pokemon_widgets in enumerate(team2_pokemon_widgets): # Iterate through widgets
             selected_pokemon_name = pokemon_widgets['pokemon_dropdown'].value
             if selected_pokemon_name != "Select Pokémon":
                pokemon_data = next((p for p in all_pokemon_data if p["Name"] == selected_pokemon_name), None)
                if pokemon_data:
                     selected_moves = [md.value for md in pokemon_widgets['move_dropdowns'] if md.value != "Select Move"]
                     selected_nature = pokemon_widgets['nature_dropdown'].value
                     selected_item = pokemon_widgets['item_dropdown'].value

                     # Get selected EVs and IVs from the widgets
                     selected_evs = {stat_names[j].lower(): ev_widget.value for j, ev_widget in enumerate(pokemon_widgets['ev_inputs'])}
                     selected_ivs = {stat_names[j].lower(): iv_widget.value for j, iv_widget in enumerate(pokemon_widgets['iv_inputs'])}

                     prepared_pokemon = prepare_pokemon_for_battle(
                         pokemon_data,
                         selected_level,
                         selected_moves,
                         selected_nature,
                         selected_item,
                         iv_dict=selected_ivs,
                         ev_dict=selected_evs
                     )
                     if prepared_pokemon:
                         team2_battle_ready.append(prepared_pokemon)
                     else:
                         # Handle case where stat calculation failed (e.g. invalid EVs)
                         print(f"Skipping {pokemon_data.get('Name', 'Unknown Pokemon')} due to stat calculation error.")
                else:
                    print(f"Warning: Data not found for Pokémon: {selected_pokemon_name}")


        if not team1_battle_ready or not team2_battle_ready:
            print("Please select at least one valid Pokémon for each team to simulate a battle.")
            return

        print(f"Team 1: {[p['Name'].capitalize() for p in team1_battle_ready]}")
        print(f"Team 2: {[p['Name'].capitalize() for p in team2_battle_ready]}")


        # Battle Prediction

        prediction_result = predict_battle_outcome(team1_battle_ready, team2_battle_ready, selected_level, num_sims, type_effectiveness_data)


        # Display Battle Prediction Results
        print("\n# Battle Prediction Results")
        print(f"## Predicted Outcome: {prediction_result.get('predicted_outcome', 'N/A')}")
        print(f"Team 1 Wins: {prediction_result.get('team1_wins', 0)} ({prediction_result.get('team1_win_percentage', 0):.2f}%)")
        print(f"Team 2 Wins: {prediction_result.get('team2_wins', 0)} ({prediction_result.get('team2_win_percentage', 0):.2f}%)")
        print(f"Draws: {prediction_result.get('draws', 0)} ({prediction_result.get('draw_percentage', 0):.2f}%)")


        # Display Synergy Reports from Battle Outcome
        team1_synergy_report = prediction_result.get('team1_synergy_report')
        team2_synergy_report = prediction_result.get('team2_synergy_report')

        if team1_synergy_report:
            display(Markdown(team1_synergy_report)) # Display the markdown report for Team 1
        if team2_synergy_report:
            display(Markdown(team2_synergy_report)) # Display the markdown report for Team 2


# Define the function to handle randomization and evaluation
def on_randomize_button_clicked(b):
    with output_area_battle:
        clear_output(wait=True)
        if len(all_pokemon_data) < 12: # Need at least 12 for two full teams
            print("Not enough Pokémon data available to form two random teams.")
            return

        # Randomly select up to 6 unique Pokémon for each team
        random_team1_data_base = random.sample(all_pokemon_data, min(6, len(all_pokemon_data) // 2))
        remaining_pokemon = [p for p in all_pokemon_data if p not in random_team1_data_base]
        random_team2_data_base = random.sample(remaining_pokemon, min(6, len(remaining_pokemon)))

        # Get the selected level and number of simulations
        selected_level = level_dropdown.value
        num_sims = num_simulations_slider.value

        # Prepare random teams with randomized moves, nature, and item, EVs, and IVs
        random_team1_battle_ready = []
        stat_names = ['HP', 'Attack', 'Defense', 'Special-attack', 'Special-defense', 'Speed'] # Define stat_names here

        for pokemon_data in random_team1_data_base:
            available_moves = pokemon_data.get("Moves", [])
            # Select 4 random moves
            selected_moves = random.sample(available_moves, min(4, len(available_moves)))
            # Select a random nature
            selected_nature = random.choice(common_natures)
            # Select a random item (including "None")
            selected_item = random.choice(held_item_options)

            # Randomize EVs (e.g., distribute 508 EVs randomly)
            selected_evs = {stat.lower(): 0 for stat in stat_names}
            remaining_evs = 508
            while remaining_evs > 0:
                stat_to_boost = random.choice(stat_names).lower()
                # Ensure boost_amount does not exceed 252 or remaining EVs
                boost_amount = random.randint(0, min(remaining_evs, 252 - selected_evs[stat_to_boost]))
                selected_evs[stat_to_boost] += boost_amount
                remaining_evs -= boost_amount

            # Randomize IVs (e.g., between 0 and 31)
            selected_ivs = {stat.lower(): random.randint(0, 31) for stat in stat_names}


            prepared_pokemon = prepare_pokemon_for_battle(
                 pokemon_data,
                 selected_level,
                 selected_moves,
                 selected_nature,
                 selected_item,
                 iv_dict=selected_ivs,
                 ev_dict=selected_evs
            )
            if prepared_pokemon:
                random_team1_battle_ready.append(prepared_pokemon)
            else:
                 print(f"Skipping {pokemon_data.get('Name', 'Unknown Pokemon')} due to stat calculation error.")


        random_team2_battle_ready = []
        for pokemon_data in random_team2_data_base:
             available_moves = pokemon_data.get("Moves", [])
             selected_moves = random.sample(available_moves, min(4, len(available_moves)))
             selected_nature = random.choice(common_natures)
             selected_item = random.choice(held_item_options)

             # Randomize EVs
             selected_evs = {stat.lower(): 0 for stat in stat_names}
             remaining_evs = 508
             while remaining_evs > 0:
                 stat_to_boost = random.choice(stat_names).lower()
                 boost_amount = random.randint(0, min(remaining_evs, 252 - selected_evs[stat_to_boost]))
                 selected_evs[stat_to_boost] += boost_amount
                 remaining_evs -= boost_amount

             # Randomize IVs
             selected_ivs = {stat.lower(): random.randint(0, 31) for stat in stat_names}


             prepared_pokemon = prepare_pokemon_for_battle(
                 pokemon_data,
                 selected_level,
                 selected_moves,
                 selected_nature,
                 selected_item,
                 iv_dict=selected_ivs,
                 ev_dict=selected_evs
             )
             if prepared_pokemon:
                 random_team2_battle_ready.append(prepared_pokemon)
             else:
                 print(f"Skipping {pokemon_data.get('Name', 'Unknown Pokemon')} due to stat calculation error.")



        if not random_team1_battle_ready or not random_team2_battle_ready:
            print("Could not prepare random teams for battle simulation.")
            return

        print(f"Random Team 1: {[p['Name'].capitalize() for p in random_team1_battle_ready]}")
        print(f"Random Team 2: {[p['Name'].capitalize() for p in random_team2_battle_ready]}")


        # Perform Battle Prediction for random teams
        prediction_result = predict_battle_outcome(random_team1_battle_ready, random_team2_battle_ready, selected_level, num_sims, type_effectiveness_data)


        # Display Battle Prediction Results
        print("\n# Battle Prediction Results")
        print(f"## Predicted Outcome: {prediction_result.get('predicted_outcome', 'N/A')}")
        print(f"Team 1 Wins: {prediction_result.get('team1_wins', 0)} ({prediction_result.get('team1_win_percentage', 0):.2f}%)")
        print(f"Team 2 Wins: {prediction_result.get('team2_wins', 0)} ({prediction_result.get('team2_win_percentage', 0):.2f}%)")
        print(f"Draws: {prediction_result.get('draws', 0)} ({prediction_result.get('draw_percentage', 0):.2f}%)")


        # Display Synergy Reports from Battle Outcome
        team1_synergy_report = prediction_result.get('team1_synergy_report')
        team2_synergy_report = prediction_result.get('team2_synergy_report')

        if team1_synergy_report:
            display(Markdown(team1_synergy_report)) # Display the markdown report for Team 1
        if team2_synergy_report:
            display(Markdown(team2_synergy_report)) # Display the markdown report for Team 2


# Link the buttons to the functions
evaluate_button.on_click(on_evaluate_button_clicked)
randomize_button.on_click(on_randomize_button_clicked)
general_report_button.on_click(on_general_report_button_clicked)

# Display the UI
ui_team1_selection = widgets.VBox([pw['box'] for pw in team1_pokemon_widgets])
ui_team2_selection = widgets.VBox([pw['box'] for pw in team2_pokemon_widgets])

ui_controls = widgets.HBox([evaluate_button, randomize_button, general_report_button, level_dropdown, num_simulations_slider])

print("Select your teams (up to 6 Pokémon each, choose moves, natures, items, EVs, and IVs):")
display(widgets.HBox([ui_team1_selection, ui_team2_selection]), ui_controls, output_area_battle)

#define synergy scores
def calculate_team_synergy_score(offensive_coverage, defensive_synergy, team):
    score = 0.0

    # Offensive Score
    # Reward having good offensive coverage (hitting types for super effective damage)
    # Penalize having poor coverage (hitting types for not very effective or no damage)
    offensive_score = 0.0
    for target_type, coverage_info in offensive_coverage.items():
        max_eff = coverage_info['max_effectiveness']
        if max_eff >= 4.0:
            offensive_score += 4.0 # Increased reward for 4x super effective
        elif max_eff >= 2.0:
            offensive_score += 2.5 # Increased reward for 2x super effective
        elif max_eff == 1.0:
            offensive_score += 0.5 # Small reward for neutral coverage
        elif max_eff <= 0.5 and max_eff > 0:
             offensive_score -= 1.5 # Increased penalty for not very effective
        elif max_eff == 0:
             offensive_score -= 3.0 # Higher penalty for no damage

    #Defensive Score
    # Reward having resistances and immunities
    # Penalize having weaknesses
    defensive_score = 0.0
    for attacking_type, synergy_info in defensive_synergy.items():
        min_mult = synergy_info['min_multiplier']
        if min_mult == 0.0:
            defensive_score += 4.0 # Increased reward for immunity
        elif min_mult <= 0.5 and min_mult > 0:
            defensive_score += 2.5 # Increased reward for resistance
        elif min_mult >= 2.0:
            defensive_score -= 3.0 # Increased penalty for weakness (2x or 4x)

    # Speed Synergy (New Factor)
    # Reward teams with a good distribution of speed tiers or high overall speed
    speed_score = 0.0
    speeds = [p['Effective Stats']['speed'] for p in team]
    if speeds:
        avg_speed = sum(speeds) / len(speeds)
        speed_score += avg_speed * 0.1 # Reward higher average speed

        # Reward having a mix of fast and slow Pokémon (might indicate speed control options)
        speed_variance = pd.Series(speeds).var() if len(speeds) > 1 else 0
        speed_score += speed_variance * 0.01 # Reward higher variance (up to a point)
    score = (offensive_score * 0.3) + (defensive_score * 0.5) + (speed_score * 0.2) # Adjusted weights


    return score

# Define the number of random teams to generate and evaluate
num_random_teams_to_evaluate = 50 #50 so no crashes

best_team = None
best_synergy_score = -float('inf') # Initialize with a very low score

print(f"Generating and evaluating {num_random_teams_to_evaluate} random teams...")

# Check if filtered_pokemon_data is available and has enough Pokémon
if 'filtered_pokemon_data' in globals() and len(filtered_pokemon_data) >= 6:
    for i in range(num_random_teams_to_evaluate):
        # Randomly select 6 unique Pokémon to form a team
        try:
            current_team_data = random.sample(filtered_pokemon_data, 6)
        except ValueError as e:
            print(f"Error sampling pokemon for teams: {e}. Not enough suitable pokemon available.")
            break # Exit if unable to sample enough pokemon
        prepared_team = []
        default_ivs = {stat.lower(): 31 for stat in ['HP', 'Attack', 'Defense', 'Special-attack', 'Special-defense', 'Speed']}
        default_evs = {stat.lower(): 0 for stat in ['HP', 'Attack', 'Defense', 'Special-attack', 'Special-defense', 'Speed']}
        default_nature = "Hardy" # Neutral nature
        default_item = "None"

        for pokemon_data in current_team_data:
             # Select the first 4 moves available, if any
             selected_moves = pokemon_data.get('Moves', [])[:4]


             prepared_pokemon = prepare_pokemon_for_battle(
                 pokemon_data,
                 level=50, # Using level 50 for evaluation
                 selected_moves=selected_moves,
                 selected_nature=default_nature,
                 selected_item=default_item,
                 iv_dict=default_ivs,
                 ev_dict=default_evs
             )
             if prepared_pokemon:
                 prepared_team.append(prepared_pokemon)
             else:
                 # Skip this team if any Pokémon could not be prepared
                 prepared_team = []
                 break # Exit the inner loop if a Pokémon fails to prepare


        if prepared_team and len(prepared_team) == 6:
            # Calculate offensive and defensive synergy for the prepared team
            offensive_coverage = analyze_offensive_coverage(prepared_team, type_effectiveness_data)
            defensive_synergy = analyze_defensive_synergy(prepared_team, type_effectiveness_data)

            # Calculate the synergy score
            current_synergy_score = calculate_team_synergy_score(offensive_coverage, defensive_synergy, prepared_team)

            # Check if this team has a better score
            if current_synergy_score > best_synergy_score:
                best_synergy_score = current_synergy_score
                best_team = prepared_team # Store the prepared team data

        if (i + 1) % 1000 == 0:
            print(f"Evaluated {i + 1} teams...")

    print("\nFinished evaluating random teams.")

    # Present the best team found
    if best_team:
        print("\n## Best Team Found (Random Sampling):")
        print(f"Synergy Score: {best_synergy_score:.2f}")
        print("Team Members:")
        for pokemon in best_team:
            print(f"- {pokemon['Name'].capitalize()} (Types: {', '.join([t.capitalize() for t in pokemon['Types']])})")

        #Gnerate and display the detailed synergy report for the best team
        print("\n### Detailed Synergy Report for the Best Team:")
        best_team_report = generate_team_synergy_report(best_team, type_effectiveness_data, level=50)
        display(Markdown(best_team_report))


    else:
        print("\nCould not find a valid team to evaluate.")

else:
    print("Error: 'filtered_pokemon_data' is not available or does not contain enough Pokémon (at least 6) to form teams.")

# Import the scatter1D_by_stat.py from Olsen file
import sys
sys.path.insert(0, '/content/')
try:
    import scatter1D_by_stat
    print("Successfully imported scatter1D_by_stat.py")
except ImportError:
    print("Error: Could not import scatter1D_by_stat.py. Please ensure the file exists at /content/scatter1D_by_stat.py")

# Pull out my team
user_pokemon_names = ['venusaur', 'blastoise', 'charizard', 'gardevoir', 'goodra', 'aggron']

# Fetch the data for the specified Pokémon
user_pokemon_data = []
for name in user_pokemon_names:
    pokemon_data = next((p for p in all_pokemon_data if p["Name"].lower() == name.lower()), None)
    if pokemon_data:
        user_pokemon_data.append(pokemon_data)
    else:
        print(f"Warning: Data not found for Pokémon: {name}")

if not user_pokemon_data:
    print("Error: Could not retrieve data for any of the specified Pokémon.")
else:
    print("\nAttempting to generate scatter plot for the specified Pokémon...")
    pokemon_for_plotting = []
    for pokemon in user_pokemon_data:
        plotting_data = {
            'Name': pokemon.get('Name', 'Unknown'),
            'Type 1': pokemon.get('Types', [None])[0],
            'Type 2': pokemon.get('Types', [None, None])[1] if len(pokemon.get('Types', [])) > 1 else None,
            'HP': pokemon['Base Stats'].get('hp', 0),
            'Attack': pokemon['Base Stats'].get('attack', 0),
            'Defense': pokemon['Base Stats'].get('defense', 0),
            'Special-attack': pokemon['Base Stats'].get('special-attack', 0),
            'Special-defense': pokemon['Base Stats'].get('special-defense', 0),
            'Speed': pokemon['Base Stats'].get('speed', 0),
            'Total Base Stats': pokemon.get('Total Base Stats', 0)
        }
        pokemon_for_plotting.append(plotting_data)

    if pokemon_for_plotting:
        plotting_df = pd.DataFrame(pokemon_for_plotting)
        try:
            print("Generating scatter plot for Attack stat:")
            scatter1D_by_stat.plot_scatter_by_stat(plotting_df, 'Attack')
            print("\nGenerating scatter plot for Speed stat:")
            scatter1D_by_stat.plot_scatter_by_stat(plotting_df, 'Speed')
            print("\nGenerating scatter plot for Total Base Stats:")
            scatter1D_by_stat.plot_scatter_by_stat(plotting_df, 'Total Base Stats')


        except AttributeError:
            print("Error: The function 'plot_scatter_by_stat' was not found in scatter1D_by_stat.py or has a different signature.")
        except Exception as e:
            print(f"An error occurred while generating the plot: {e}")

# Evaluate the win rate of the best team found in the previous step (if a best team was found)
if 'best_team' in globals() and best_team:
    print("\nEvaluating the Win Rate of the Best Team")

    num_evaluation_battles = 100 # Number of battles to simulate for win rate evaluation
    best_team_wins = 0
    best_team_draws = 0
    evaluation_level = 50 # Use the same level as the synergy evaluation

    print(f"Simulating {num_evaluation_battles} battles with the best team against random teams...")

    for i in range(num_evaluation_battles):
        # Generate a random opponent team (excluding Pokémon already in the best team)
        if 'all_pokemon_data' not in globals() or not all_pokemon_data or len(all_pokemon_data) < len(best_team) + 6:
            print("Not enough Pokémon data available to form random opponent teams.")
            break

        best_team_names = [p['Name'] for p in best_team]
        remaining_pokemon_for_random = [p for p in all_pokemon_data if p['Name'] not in best_team_names]

        if len(remaining_pokemon_for_random) < 6:
             print("Not enough unique Pokémon data available to form random opponent teams.")
             break


        random_opponent_data_base = random.sample(remaining_pokemon_for_random, 6)

        # Prepare the random opponent team (random moves, nature, item, default EVs/IVs)
        opponent_team_prepared = []
        stat_names = ['HP', 'Attack', 'Defense', 'Special-attack', 'Special-defense', 'Speed']
        default_ivs = {stat.lower(): 31 for stat in stat_names}
        default_evs = {stat.lower(): 0 for stat in stat_names}
        common_natures = [
            "Adamant", "Modest", "Jolly", "Timid", "Brave", "Quiet", "Relaxed",
            "Sassy", "Bold", "Impish", "Careful", "Calm", "Gentle",
            "Naive", "Hasty", "Lax", "Naughty", "Lonely", "Hardy", "Docile", "Bashful",
            "Quirky", "Serious"
        ]
        held_item_options = ["None"] + sorted(list(item_multipliers.keys()))


        def prepare_random_pokemon(pokemon_data, level, default_ivs, default_evs, common_natures, held_item_options):
             if not pokemon_data: return None
             available_moves = pokemon_data.get("Moves", [])
             selected_moves = random.sample(available_moves, min(4, len(available_moves)))
             selected_nature = random.choice(common_natures)
             selected_item = random.choice(held_item_options)

             return prepare_pokemon_for_battle(
                 pokemon_data,
                 level=level,
                 selected_moves=selected_moves,
                 selected_nature=selected_nature,
                 selected_item=selected_item,
                 iv_dict=default_ivs,
                 ev_dict=default_evs
             )

        for pokemon_data in random_opponent_data_base:
            prepared = prepare_random_pokemon(pokemon_data, evaluation_level, default_ivs, default_evs, common_natures, held_item_options)
            if prepared:
                opponent_team_prepared.append(prepared)


        if opponent_team_prepared and len(opponent_team_prepared) == 6:
            best_team_sim = [p.copy() for p in best_team]
            opponent_team_sim = [p.copy() for p in opponent_team_prepared]


            # Use a small number of simulations per battle for speed in this evaluation
            prediction_result = predict_battle_outcome(best_team_sim, opponent_team_sim, evaluation_level, num_simulations=10, type_effectiveness_data=type_effectiveness_data)
            predicted_outcome = prediction_result['predicted_outcome']

            if predicted_outcome == 'Team 1 Wins':
                best_team_wins += 1
            elif predicted_outcome == 'Draw':
                 best_team_draws += 1

        else:
             print(f"Skipping evaluation battle {i+1} due to inability to prepare opponent team.")


        if (i + 1) % 20 == 0:
            print(f"Simulated {i + 1} evaluation battles")


    total_evaluated_battles = num_evaluation_battles
    win_rate = (best_team_wins / total_evaluated_battles) * 100 if total_evaluated_battles > 0 else 0
    draw_rate = (best_team_draws / total_evaluated_battles) * 100 if total_evaluated_battles > 0 else 0

    print("\nEvaluation Results")
    print(f"Best Team Win Rate: {win_rate:.2f}%")
    print(f"Best Team Draw Rate: {draw_rate:.2f}%")
    print(f"Best Team Loss Rate: {100 - win_rate - draw_rate:.2f}%")

elif 'best_team' not in globals():
    print("Please run the previous cell to find the best team before evaluating its win rate.")
else:
    print("No valid best team was found in the previous step to evaluate its win rate.")

"""**Reasoning**:
Review and refine the `calculate_team_synergy_score` function to incorporate more factors or adjust weights for a potentially better representation of team synergy.


"""

#Adding more factors
def calculate_team_synergy_score(offensive_coverage, defensive_synergy, team):
    """
    Calculates a numerical synergy score for a team based on offensive coverage,
    defensive synergy, and other potential factors like speed distribution.
    Higher score indicates better synergy.
    This is a refined scoring mechanism.
    """
    score = 0.0

    # Offensive Score
    # Reward having good offensive coverage (hitting types for super effective damage)
    # Penalize having poor coverage (hitting types for not very effective or no damage)
    offensive_score = 0.0
    for target_type, coverage_info in offensive_coverage.items():
        max_eff = coverage_info['max_effectiveness']
        if max_eff >= 4.0:
            offensive_score += 4.0 # Increased reward for 4x super effective
        elif max_eff >= 2.0:
            offensive_score += 2.5 # Increased reward for 2x super effective
        elif max_eff == 1.0:
            offensive_score += 0.5 # Small reward for neutral coverage
        elif max_eff <= 0.5 and max_eff > 0:
             offensive_score -= 1.5 # Increased penalty for not very effective
        elif max_eff == 0:
             offensive_score -= 3.0 # Higher penalty for no damage

    # Defensive Score
    # Reward having resistances and immunities
    # Penalize having weaknesses
    defensive_score = 0.0
    for attacking_type, synergy_info in defensive_synergy.items():
        min_mult = synergy_info['min_multiplier']
        if min_mult == 0.0:
            defensive_score += 4.0 # Increased reward for immunity
        elif min_mult <= 0.5 and min_mult > 0:
            defensive_score += 2.5 # Increased reward for resistance
        elif min_mult >= 2.0:
            defensive_score -= 3.0 # Increased penalty for weakness (2x or 4x)

    #Speed Synergy (New Factor)
    # Reward teams with a good distribution of speed tiers or high overall speed
    speed_score = 0.0
    speeds = [p['Effective Stats']['speed'] for p in team]
    if speeds:
        avg_speed = sum(speeds) / len(speeds)
        speed_score += avg_speed * 0.1 # Reward higher average speed

        # Reward having a mix of fast and slow Pokémon (might indicate speed control options)
        # This is a simple approach; more complex analysis could look at speed tiers.
        speed_variance = pd.Series(speeds).var() if len(speeds) > 1 else 0
        speed_score += speed_variance * 0.01 # Reward higher variance (up to a point)
    score = (offensive_score * 0.3) + (defensive_score * 0.5) + (speed_score * 0.2) # Adjusted weights


    return score

#Adjusting simple W/L ratio in battle outcome
def predict_battle_outcome(team1, team2, level, num_simulations, type_effectiveness_data):
    """
    Simulates multiple battles between two teams and predicts the outcome
    based on the simulation results.
    Also generates synergy reports for both teams.
    Returns a dictionary with prediction results and synergy reports.
    """
    team1_wins = 0
    team2_wins = 0
    draws = 0
    team1_remaining_hp_total = 0
    team2_remaining_hp_total = 0
    team1_pokemon_remaining_total = 0
    team2_pokemon_remaining_total = 0


    for _ in range(num_simulations):
        # Create fresh copies of the teams for each simulation
        team1_sim = [p.copy() for p in team1]
        team2_sim = [p.copy() for p in team2]

        winner, battle_log = simulate_battle(team1_sim, team2_sim, type_effectiveness_data)

        if winner == 'Team 1':
            team1_wins += 1
            # Calculate remaining HP and Pokémon for the winning team
            team1_remaining_hp_total += sum(p['current_hp'] for p in team1_sim if p['current_hp'] > 0)
            team1_pokemon_remaining_total += sum(1 for p in team1_sim if p['current_hp'] > 0)

        elif winner == 'Team 2':
            team2_wins += 1
            # Calculate remaining HP and Pokémon for the winning team
            team2_remaining_hp_total += sum(p['current_hp'] for p in team2_sim if p['current_hp'] > 0)
            team2_pokemon_remaining_total += sum(1 for p in team2_sim if p['current_hp'] > 0)
        else:
            draws += 1
            # For draws, you might sum remaining HP/Pokémon for both or handle differently
            team1_remaining_hp_total += sum(p['current_hp'] for p in team1_sim if p['current_hp'] > 0)
            team1_pokemon_remaining_total += sum(1 for p in team1_sim if p['current_hp'] > 0)
            team2_remaining_hp_total += sum(p['current_hp'] for p in team2_sim if p['current_hp'] > 0)
            team2_pokemon_remaining_total += sum(1 for p in team2_sim if p['current_hp'] > 0)


    total_simulations = num_simulations
    team1_win_percentage = (team1_wins / total_simulations) * 100
    team2_win_percentage = (team2_wins / total_simulations) * 100
    draw_percentage = (draws / total_simulations) * 100

    # Determine predicted outcome
    if team1_wins > team2_wins and team1_wins > draws:
        predicted_outcome = 'Team 1 Wins'
    elif team2_wins > team1_wins and team2_wins > draws:
        predicted_outcome = 'Team 2 Wins'
    elif team1_wins == team2_wins:
        predicted_outcome = 'Likely Draw or Close Match' # More descriptive for ties
    else:
        predicted_outcome = 'Draw'


    # Calculate average remaining HP and Pokémon for winning teams (or all teams in case of draws)
    avg_team1_remaining_hp = team1_remaining_hp_total / (team1_wins + draws) if (team1_wins + draws) > 0 else 0
    avg_team2_remaining_hp = team2_remaining_hp_total / (team2_wins + draws) if (team2_wins + draws) > 0 else 0
    avg_team1_pokemon_remaining = team1_pokemon_remaining_total / (team1_wins + draws) if (team1_wins + draws) > 0 else 0
    avg_team2_pokemon_remaining = team2_pokemon_remaining_total / (team2_wins + draws) if (team2_wins + draws) > 0 else 0


    # Generate synergy reports for the *initial* teams (not the simulation copies)
    team1_synergy_report_markdown = generate_team_synergy_report(team1, type_effectiveness_data, level)
    team2_synergy_report_markdown = generate_team_synergy_report(team2, type_effectiveness_data, level)

    # Calculate synergy scores for the initial teams
    team1_offensive_coverage = analyze_offensive_coverage(team1, type_effectiveness_data)
    team1_defensive_synergy = analyze_defensive_synergy(team1, type_effectiveness_data)
    team1_synergy_score = calculate_team_synergy_score(team1_offensive_coverage, team1_defensive_synergy, team1)

    team2_offensive_coverage = analyze_offensive_coverage(team2, type_effectiveness_data)
    team2_defensive_synergy = analyze_defensive_synergy(team2, type_effectiveness_data)
    team2_synergy_score = calculate_team_synergy_score(team2_offensive_coverage, team2_defensive_synergy, team2)


    return {
        'predicted_outcome': predicted_outcome,
        'team1_wins': team1_wins,
        'team2_wins': team2_wins,
        'draws': draws,
        'team1_win_percentage': team1_win_percentage,
        'team2_win_percentage': team2_win_percentage,
        'draw_percentage': draw_percentage,
        'avg_team1_remaining_hp': avg_team1_remaining_hp,
        'avg_team2_remaining_hp': avg_team2_remaining_hp,
        'avg_team1_pokemon_remaining': avg_team1_pokemon_remaining,
        'avg_team2_pokemon_remaining': avg_team2_pokemon_remaining,
        'team1_synergy_report': team1_synergy_report_markdown,
        'team2_synergy_report': team2_synergy_report_markdown,
        'team1_synergy_score': team1_synergy_score,
        'team2_synergy_score': team2_synergy_score
    }

#Call predition values
def predict_outcome_from_names(team1_names, team2_names, level):
    """
    Prepares two teams based on lists of Pokémon names and predicts the battle outcome.
    Uses default moves (first 4), neutral nature (Hardy), no item, 0 EVs, and 31 IVs.
    """
    team1_prepared = []
    team2_prepared = []
    default_ivs = {stat.lower(): 31 for stat in ['HP', 'Attack', 'Defense', 'Special-attack', 'Special-defense', 'Speed']}
    default_evs = {stat.lower(): 0 for stat in ['HP', 'Attack', 'Defense', 'Special-attack', 'Special-defense', 'Speed']}
    default_nature = "Hardy"
    default_item = "None"
    num_simulations = 100

    # Prepare Team 1
    for name in team1_names:
        pokemon_data = next((p for p in all_pokemon_data if p["Name"].lower() == name.lower()), None)
        if pokemon_data:
            selected_moves = pokemon_data.get('Moves', [])[:4] # First 4 moves
            prepared_pokemon = prepare_pokemon_for_battle(
                pokemon_data,
                level=level,
                selected_moves=selected_moves,
                selected_nature=default_nature,
                selected_item=default_item,
                iv_dict=default_ivs,
                ev_dict=default_evs
            )
            if prepared_pokemon:
                team1_prepared.append(prepared_pokemon)
            else:
                print(f"Warning: Could not prepare {name} for battle.")
        else:
            print(f"Warning: Pokémon data not found for '{name}'.")

    # Prepare Team 2
    for name in team2_names:
        pokemon_data = next((p for p in all_pokemon_data if p["Name"].lower() == name.lower()), None)
        if pokemon_data:
            selected_moves = pokemon_data.get('Moves', [])[:4] # First 4 moves
            prepared_pokemon = prepare_pokemon_for_battle(
                pokemon_data,
                level=level,
                selected_moves=selected_moves,
                selected_nature=default_nature,
                selected_item=default_item,
                iv_dict=default_ivs,
                ev_dict=default_evs
            )
            if prepared_pokemon:
                team2_prepared.append(prepared_pokemon)
            else:
                print(f"Warning: Could not prepare {name} for battle.")
        else:
            print(f"Warning: Pokémon data not found for '{name}'.")

    if not team1_prepared or not team2_prepared:
        print("Could not prepare at least one team for prediction.")
        return None

    # Call predict_battle_outcome
    prediction_result = predict_battle_outcome(team1_prepared, team2_prepared, level, num_simulations, type_effectiveness_data)

    # Print relevant results
    print("\n--- Prediction Results ---")
    print(f"Team 1: {[p['Name'].capitalize() for p in team1_prepared]}")
    print(f"Team 2: {[p['Name'].capitalize() for p in team2_prepared]}")
    print(f"Level: {level}, Simulations: {num_simulations}")
    print(f"Predicted Outcome: {prediction_result['predicted_outcome']}")
    print(f"Team 1 Wins: {prediction_result['team1_wins']} ({prediction_result['team1_win_percentage']:.2f}%)")
    print(f"Team 2 Wins: {prediction_result['team2_wins']} ({prediction_result['team2_win_percentage']:.2f}%)")
    print(f"Draws: {prediction_result['draws']} ({prediction_result['draw_percentage']:.2f}%)")
    print(f"Team 1 Synergy Score: {prediction_result['team1_synergy_score']:.2f}")
    print(f"Team 2 Synergy Score: {prediction_result['team2_synergy_score']:.2f}")

    # Optionally return the full result dictionary
    return prediction_result

# Example usage
team1_example = ['pikachu', 'charizard', 'blastoise', 'venusaur', 'snorlax', 'lapras']
team2_example = ['alakazam', 'machamp', 'gengar', 'gyarados', 'jolteon', 'vaporeon']
level_example = 50

predict_outcome_from_names(team1_example, team2_example, level_example)

# 1. Define the number of synthetic battles to simulate
num_synthetic_battles = 100

# 2. Define the level for Pokémon in synthetic battles
synthetic_battle_level = 50

#print statements
print(f"Number of synthetic battles to simulate: {num_synthetic_battles}")
print(f"Pokémon level for synthetic battles: {synthetic_battle_level}")
print(f"Using filtered_pokemon_data ({len(filtered_pokemon_data)} Pokémon) which excludes Total Base Stats > 600.")

# No further filtering is required based on the current instructions for this subtask.

# Generate synthetic battle data
synthetic_battles_data = []

# Check if filtered_pokemon_data is available and has enough Pokémon
if 'filtered_pokemon_data' in globals() and len(filtered_pokemon_data) >= 12: # Need at least 12 for two full teams
    print(f"Generating {num_synthetic_battles} synthetic battles...")
    for i in range(num_synthetic_battles):
        # Randomly select up to 6 unique Pokémon for each team from filtered data
        try:
            random_team1_data_base = random.sample(filtered_pokemon_data, min(6, len(filtered_pokemon_data) // 2))
            remaining_pokemon = [p for p in filtered_pokemon_data if p not in random_team1_data_base]
            random_team2_data_base = random.sample(remaining_pokemon, min(6, len(remaining_pokemon)))
        except ValueError as e:
            print(f"Error sampling pokemon for teams: {e}. Not enough suitable pokemon available.")
            break # Exit if unable to sample enough pokemon


        # Prepare random teams with randomized moves, nature, item, EVs, and IVs
        team1_prepared = []
        team2_prepared = []
        stat_names = ['HP', 'Attack', 'Defense', 'Special-attack', 'Special-defense', 'Speed']
        stat_names_lower = [s.lower() for s in stat_names]

        def prepare_random_pokemon(pokemon_data, level):
            available_moves = pokemon_data.get("Moves", [])
            selected_moves = random.sample(available_moves, min(4, len(available_moves))) # Select 4 random moves
            selected_nature = random.choice(common_natures) # Select a random nature
            selected_item = random.choice(held_item_options) # Select a random item (including "None")

            # Randomize EVs (distribute 508 EVs randomly, max 252 per stat)
            selected_evs = {stat: 0 for stat in stat_names_lower}
            remaining_evs = 508
            while remaining_evs > 0:
                stat_to_boost = random.choice(stat_names_lower)
                boost_amount = random.randint(0, min(remaining_evs, 252 - selected_evs[stat_to_boost]))
                selected_evs[stat_to_boost] += boost_amount
                remaining_evs -= boost_amount

            # Randomize IVs (between 0 and 31)
            selected_ivs = {stat: random.randint(0, 31) for stat in stat_names_lower}

            return prepare_pokemon_for_battle(
                pokemon_data,
                level=level,
                selected_moves=selected_moves,
                selected_nature=selected_nature,
                selected_item=selected_item,
                iv_dict=selected_ivs,
                ev_dict=selected_evs
            )

        for pokemon_data in random_team1_data_base:
            prepared = prepare_random_pokemon(pokemon_data, synthetic_battle_level)
            if prepared:
                team1_prepared.append(prepared)

        for pokemon_data in random_team2_data_base:
             prepared = prepare_random_pokemon(pokemon_data, synthetic_battle_level)
             if prepared:
                 team2_prepared.append(prepared)


        if team1_prepared and team2_prepared:
             # Simulate the battle
             # Only run a small number of simulations per battle for dataset generation speed
             prediction_result = predict_battle_outcome(team1_prepared, team2_prepared, synthetic_battle_level, num_simulations=5, type_effectiveness_data=type_effectiveness_data)

             # Store battle data and outcome
             synthetic_battles_data.append({
                 'Team1_Pokemon': [p['Name'] for p in team1_prepared],
                 'Team1_Moves': [[m['name'] for m in p['Moves']] for p in team1_prepared], # Store moves as list of lists
                 'Team1_Natures': [p['Nature'] for p in team1_prepared],
                 'Team1_Items': [p['Item'] for p in team1_prepared],
                 'Team1_EVs': [[p['EVs'].get(s, 0) for s in stat_names_lower] for p in team1_prepared], # Store EVs as list of lists
                 'Team1_IVs': [[p['IVs'].get(s, 0) for s in stat_names_lower] for p in team1_prepared], # Store IVs as list of lists
                 'Team2_Pokemon': [p['Name'] for p in team2_prepared],
                 'Team2_Moves': [[m['name'] for m in p['Moves']] for p in team2_prepared],
                 'Team2_Natures': [p['Nature'] for p in team2_prepared],
                 'Team2_Items': [p['Item'] for p in team2_prepared],
                 'Team2_EVs': [[p['EVs'].get(s, 0) for s in stat_names_lower] for p in team2_prepared],
                 'Team2_IVs': [[p['IVs'].get(s, 0) for s in stat_names_lower] for p in team2_prepared],
                 'Level': synthetic_battle_level,
                 'Outcome': prediction_result['predicted_outcome'] # Use the predicted outcome as the 'actual' outcome for synthetic data
             })
        else:
             print(f"Skipping battle {i+1} due to inability to prepare teams.")


        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1} battles...")

    print(f"\nFinished generating {len(synthetic_battles_data)} synthetic battles.")

    # Convert to DataFrame
    synthetic_battle_df = pd.DataFrame(synthetic_battles_data)

else:
     print("Error: 'filtered_pokemon_data' is not available or does not contain enough Pokémon (at least 12) to form teams.")
     synthetic_battle_df = pd.DataFrame() # Create empty DataFrame if data generation failed

# Display the first few rows of the generated data
if not synthetic_battle_df.empty:
    display(synthetic_battle_df.head())



import random
import itertools

#best 10 teams randomly generated
# Define the number of random teams to generate and evaluate for finding the best synergy
num_teams_for_synergy_search = 10

best_teams = []  # Store the top 10 teams and their scores
top_n = 10

print(f"Generating and evaluating {num_teams_for_synergy_search} random teams to find the top {top_n} by synergy score (under 600 total stats)...")

# Check if filtered_pokemon_data is available and has enough Pokémon
if 'filtered_pokemon_data' in globals() and len(filtered_pokemon_data) >= 6:
    for i in range(num_teams_for_synergy_search):
        # Randomly select 6 unique Pokémon to form a team
        try:
            current_team_data = random.sample(filtered_pokemon_data, 6)
        except ValueError as e:
            print(f"Error sampling pokemon for teams: {e}. Not enough suitable pokemon available.")
            break # Exit if unable to sample enough pokemon
        prepared_team = []
        default_ivs = {stat.lower(): 31 for stat in ['HP', 'Attack', 'Defense', 'Special-attack', 'Special-defense', 'Speed']}
        default_evs = {stat.lower(): 0 for stat in ['HP', 'Attack', 'Defense', 'Special-attack', 'Special-defense', 'Speed']}
        default_nature = "Hardy" # Neutral nature
        default_item = "None"
        level_for_synergy = 50 # Using level 50 for synergy evaluation


        for pokemon_data in current_team_data:
             # Select the first 4 moves available, if any
             available_moves = pokemon_data.get('Moves', [])
             selected_moves = random.sample(available_moves, min(4, len(available_moves)))


             prepared_pokemon = prepare_pokemon_for_battle(
                 pokemon_data,
                 level=level_for_synergy,
                 selected_moves=selected_moves,
                 selected_nature=default_nature,
                 selected_item=default_item,
                 iv_dict=default_ivs,
                 ev_dict=default_evs
             )
             if prepared_pokemon:
                 prepared_team.append(prepared_pokemon)
             else:
                 # Skip this team if any Pokémon could not be prepared
                 prepared_team = []
                 break # Exit the inner loop if a Pokémon fails to prepare


        if prepared_team and len(prepared_team) == 6:
            # Calculate offensive and defensive synergy for the prepared team
            offensive_coverage = analyze_offensive_coverage(prepared_team, type_effectiveness_data)
            defensive_synergy = analyze_defensive_synergy(prepared_team, type_effectiveness_data)

            # Calculate the synergy score
            current_synergy_score = calculate_team_synergy_score(offensive_coverage, defensive_synergy, prepared_team)

            # Maintain the top N list
            if len(best_teams) < top_n:
                best_teams.append({'team': prepared_team, 'score': current_synergy_score})
                best_teams.sort(key=lambda x: x['score'], reverse=True) # Keep sorted
            elif current_synergy_score > best_teams[-1]['score']:
                best_teams.pop() # Remove the lowest score
                best_teams.append({'team': prepared_team, 'score': current_synergy_score})
                best_teams.sort(key=lambda x: x['score'], reverse=True) # Keep sorted


        if (i + 1) % 1000 == 0:
            print(f"Evaluated {i + 1} teams...")

    print(f"\nFinished evaluating {num_teams_for_synergy_search} random teams.")

    # Present the top N teams found
    if best_teams:
        print(f"\n## Top {top_n} Teams by Synergy Score (Pokémon with <= 600 Total Stats):")
        for rank, team_info in enumerate(best_teams):
            team = team_info['team']
            score = team_info['score']
            team_names = [p['Name'].capitalize() for p in team]
            print(f"\nRank {rank + 1} - Synergy Score: {score:.2f}")
            print(f"Team: {', '.join(team_names)}")


    else:
        print("\nCould not find any valid teams to evaluate.")

else:
    print("Error: 'filtered_pokemon_data' is not available or does not contain enough Pokémon (at least 6) to form teams.")

# Define the path to save the synthetic battle dataset
synthetic_battle_dataset_path = '/tmp/synthetic_battle_dataset.csv'

# Check if the synthetic_battle_df DataFrame exists and is not empty
if 'synthetic_battle_df' in globals() and not synthetic_battle_df.empty:
    try:
        # Save the DataFrame to a CSV file
        synthetic_battle_df.to_csv(synthetic_battle_dataset_path, index=False)
        print(f"Synthetic battle dataset saved to: {synthetic_battle_dataset_path}")
    except Exception as e:
        print(f"Error saving synthetic battle dataset: {e}")
else:
    print("No synthetic battle data to save. Please run the data generation step first.")

def predict_outcome_from_names(team1_names, team2_names, level):
    """
    Prepares two teams based on lists of Pokémon names and predicts the battle outcome.
    Uses default moves (first 4), neutral nature (Hardy), no item, 0 EVs, and 31 IVs.
    """
    team1_prepared = []
    team2_prepared = []
    default_ivs = {stat.lower(): 31 for stat in ['HP', 'Attack', 'Defense', 'Special-attack', 'Special-defense', 'Speed']}
    default_evs = {stat.lower(): 0 for stat in ['HP', 'Attack', 'Defense', 'Special-attack', 'Special-defense', 'Speed']}
    default_nature = "Hardy"
    default_item = "None"
    num_simulations = 100 # Reasonable number of simulations

    # Prepare Team 1
    for name in team1_names:
        pokemon_data = next((p for p in all_pokemon_data if p["Name"].lower() == name.lower()), None)
        if pokemon_data:
            selected_moves = pokemon_data.get('Moves', [])[:4] # First 4 moves
            prepared_pokemon = prepare_pokemon_for_battle(
                pokemon_data,
                level=level,
                selected_moves=selected_moves,
                selected_nature=default_nature,
                selected_item=default_item,
                iv_dict=default_ivs,
                ev_dict=default_evs
            )
            if prepared_pokemon:
                team1_prepared.append(prepared_pokemon)
            else:
                print(f"Warning: Could not prepare {name} for battle.")
        else:
            print(f"Warning: Pokémon data not found for '{name}'.")

    # Prepare Team 2
    for name in team2_names:
        pokemon_data = next((p for p in all_pokemon_data if p["Name"].lower() == name.lower()), None)
        if pokemon_data:
            selected_moves = pokemon_data.get('Moves', [])[:4] # First 4 moves
            prepared_pokemon = prepare_pokemon_for_battle(
                pokemon_data,
                level=level,
                selected_moves=selected_moves,
                selected_nature=default_nature,
                selected_item=default_item,
                iv_dict=default_ivs,
                ev_dict=default_evs
            )
            if prepared_pokemon:
                team2_prepared.append(prepared_pokemon)
            else:
                print(f"Warning: Could not prepare {name} for battle.")
        else:
            print(f"Warning: Pokémon data not found for '{name}'.")

    if not team1_prepared or not team2_prepared:
        print("Could not prepare at least one team for prediction.")
        return None

    # Call predict_battle_outcome
    prediction_result = predict_battle_outcome(team1_prepared, team2_prepared, level, num_simulations, type_effectiveness_data)

    # Print relevant results
    print("\n--- Prediction Results ---")
    print(f"Team 1: {[p['Name'].capitalize() for p in team1_prepared]}")
    print(f"Team 2: {[p['Name'].capitalize() for p in team2_prepared]}")
    print(f"Level: {level}, Simulations: {num_simulations}")
    print(f"Predicted Outcome: {prediction_result['predicted_outcome']}")
    print(f"Team 1 Wins: {prediction_result['team1_wins']} ({prediction_result['team1_win_percentage']:.2f}%)")
    print(f"Team 2 Wins: {prediction_result['team2_wins']} ({prediction_result['team2_win_percentage']:.2f}%)")
    print(f"Draws: {prediction_result['draws']} ({prediction_result['draw_percentage']:.2f}%)")
    print(f"Team 1 Synergy Score: {prediction_result['team1_synergy_score']:.2f}")
    print(f"Team 2 Synergy Score: {prediction_result['team2_synergy_score']:.2f}")

    # Optionally return the full result dictionary
    return prediction_result

# Example usage
team1_example = ['pikachu', 'charizard', 'blastoise', 'venusaur', 'snorlax', 'lapras']
team2_example = ['alakazam', 'machamp', 'gengar', 'gyarados', 'jolteon', 'vaporeon']
level_example = 50

predict_outcome_from_names(team1_example, team2_example, level_example)

!pip install dash dash-core-components dash-html-components plotly

# Initialize the Dash app
app = dash.Dash(__name__)

# Get a list of all pokemon names from the loaded data
if 'all_pokemon_data' in globals() and all_pokemon_data:
    all_pokemon_names = sorted([p['Name'] for p in all_pokemon_data])
else:
    all_pokemon_names = []
    print("Warning: all_pokemon_data not found or empty. Pokémon dropdowns will be empty.")

# Extract item names from the item_multipliers dictionary
if 'item_multipliers' in globals():
    held_item_options = ["None"] + sorted(list(item_multipliers.keys()))
else:
    held_item_options = ["None"]
    print("Warning: item_multipliers not found. Item dropdowns will only contain 'None'.")


# Define stat names for EV/IV inputs
stat_names = ['HP', 'Attack', 'Defense', 'Special-attack', 'Special-defense', 'Speed']


# Function to create a single Pokémon configuration block
def create_pokemon_config_block(team_number, pokemon_index):
    return html.Div([
        html.H5(f'Pokémon {pokemon_index + 1}'),
        dcc.Dropdown(
            id=f'team{team_number}-pokemon-{pokemon_index}',
            options=[{'label': name.capitalize(), 'value': name} for name in ["Select Pokémon"] + all_pokemon_names],
            value="Select Pokémon"
        ),
        dcc.Dropdown(
            id=f'team{team_number}-move1-{pokemon_index}',
            options=[{'label': 'Select Move', 'value': 'Select Move'}],
            value='Select Move',
            placeholder='Select Move',
            disabled=True
        ),
        dcc.Dropdown(
            id=f'team{team_number}-move2-{pokemon_index}',
            options=[{'label': 'Select Move', 'value': 'Select Move'}],
            value='Select Move',
            placeholder='Select Move',
            disabled=True
        ),
        dcc.Dropdown(
            id=f'team{team_number}-move3-{pokemon_index}',
            options=[{'label': 'Select Move', 'value': 'Select Move'}],
            value='Select Move',
            placeholder='Select Move',
            disabled=True
        ),
        dcc.Dropdown(
            id=f'team{team_number}-move4-{pokemon_index}',
            options=[{'label': 'Select Move', 'value': 'Select Move'}],
            value='Select Move',
            placeholder='Select Move',
            disabled=True
        ),
        dcc.Dropdown(
            id=f'team{team_number}-nature-{pokemon_index}',
            options=[{'label': name.capitalize(), 'value': name} for name in ["Select Nature"] + common_natures],
            value="Select Nature",
            placeholder='Select Nature'
        ),
        dcc.Dropdown(
            id=f'team{team_number}-item-{pokemon_index}',
            options=[{'label': name.replace('-', ' ').title(), 'value': name} for name in held_item_options],
            value="None",
            placeholder='Select Item'
        ),
        html.Hr() # Separator
    ])


# Create configuration blocks for each team (6 Pokémon each)
team1_configs = [create_pokemon_config_block(1, i) for i in range(6)]
team2_configs = [create_pokemon_config_block(2, i) for i in range(6)]

# App layout
app.layout = html.Div(style={'backgroundColor': '#f0f0f0', 'padding': '20px'}, children=[ # Apply background and padding here
    html.H1("Pokémon Battle Simulator & Team Analyzer", style={'color': '#333'}), # Apply title color here

    dcc.Tabs(id="tabs-example-graph", value='tab-1', style={'backgroundColor': '#e0e0f0', 'padding': '10px'}, children=[ # Apply tab area background and padding here
        dcc.Tab(label='Team 1 Configuration', value='tab-1', children=[
            html.Div(team1_configs, style={'padding': '10px'})
        ]),
        dcc.Tab(label='Team 2 Configuration', value='tab-2', children=[
             html.Div(team2_configs, style={'padding': '10px'})
        ]),
         dcc.Tab(label='Battle & Report', value='tab-3', children=[
            html.Div([
                html.H2("Battle Simulation & Report"),
                html.Label("Pokémon Level:"),
                 dcc.Dropdown(
                    id='battle-level',
                    options=[{'label': str(level), 'value': level} for level in [50, 100]],
                    value=50,
                    style={'width': '100px'}
                ),
                html.Label("Number of Simulations:"),
                dcc.Input(
                    id='num-simulations',
                    type='number',
                    value=100,
                    min=10,
                    step=10,
                    style={'width': '100px'}
                ),
                 html.Button('Simulate Battle & Generate Report (UI Teams)', id='simulate-button', n_clicks=0),
                 html.Button('Randomize Team 2 & Simulate', id='randomize-team2-simulate-button', n_clicks=0, style={'marginLeft': '10px'}),
                 html.Button('Randomize Both Teams & Simulate', id='randomize-both-simulate-button', n_clicks=0, style={'marginLeft': '10px'}), # New button

                html.Div(id='battle-output') # Area to display battle results and reports
            ], style={'padding': '10px'})
        ]),
    ]),
    html.Div(id='tabs-content-example-graph') # Content area for tabs (currently not used for content)
])


# Callback to update move dropdowns based on selected Pokémon
def create_move_dropdown_callback(team_number, pokemon_index):
    @app.callback(
        [Output(f'team{team_number}-move1-{pokemon_index}', 'options'),
         Output(f'team{team_number}-move1-{pokemon_index}', 'disabled'),
         Output(f'team{team_number}-move1-{pokemon_index}', 'value'), # Reset value
         Output(f'team{team_number}-move2-{pokemon_index}', 'options'),
         Output(f'team{team_number}-move2-{pokemon_index}', 'disabled'),
         Output(f'team{team_number}-move2-{pokemon_index}', 'value'), # Reset value
         Output(f'team{team_number}-move3-{pokemon_index}', 'options'),
         Output(f'team{team_number}-move3-{pokemon_index}', 'disabled'),
         Output(f'team{team_number}-move3-{pokemon_index}', 'value'), # Reset value
         Output(f'team{team_number}-move4-{pokemon_index}', 'options'),
         Output(f'team{team_number}-move4-{pokemon_index}', 'disabled'),
         Output(f'team{team_number}-move4-{pokemon_index}', 'value')], # Reset value
        [Input(f'team{team_number}-pokemon-{pokemon_index}', 'value')]
    )
    def update_moves(selected_pokemon_name):
        if selected_pokemon_name and selected_pokemon_name != "Select Pokémon" and all_pokemon_data:
            pokemon_data = next((p for p in all_pokemon_data if p["Name"].lower() == selected_pokemon_name.lower()), None)
            if pokemon_data and 'Moves' in pokemon_data:
                available_moves = sorted(pokemon_data['Moves'])
                move_options = [{'label': 'Select Move', 'value': 'Select Move'}] + [{'label': move.capitalize(), 'value': move} for move in available_moves]
                return [move_options, False, 'Select Move'] * 4
        # Reset to default if no valid pokemon selected
        default_options = [{'label': 'Select Move', 'value': 'Select Move'}]
        return [default_options, True, 'Select Move'] * 4


# Create callbacks for all Pokémon move dropdowns
for team_number in [1, 2]:
    for pokemon_index in range(6):
        create_move_dropdown_callback(team_number, pokemon_index)


# Callback to handle all battle simulation buttons
@app.callback(
    Output('battle-output', 'children'),
    [Input('simulate-button', 'n_clicks'),
     Input('randomize-team2-simulate-button', 'n_clicks'),
     Input('randomize-both-simulate-button', 'n_clicks')], # New input for the new button
    [State('battle-level', 'value'),
     State('num-simulations', 'value')] +
    [State(f'team{team_number}-pokemon-{pokemon_index}', 'value') for team_number in [1, 2] for pokemon_index in range(6)] +
    [State(f'team{team_number}-move{move_index+1}-{pokemon_index}', 'value') for team_number in [1, 2] for pokemon_index in range(6) for move_index in range(4)] +
    [State(f'team{team_number}-nature-{pokemon_index}', 'value') for team_number in [1, 2] for pokemon_index in range(6)] +
    [State(f'team{team_number}-item-{pokemon_index}', 'value') for team_number in [1, 2] for pokemon_index in range(6)]
)
def update_battle_output(simulate_clicks, randomize_team2_clicks, randomize_both_clicks, level, num_simulations, *args):
    ctx = dash.callback_context

    if not ctx.triggered:
        return "Select teams and simulate a battle."

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Extract selected team configurations from State args
    team1_configs_raw = args[:6]
    team2_configs_raw = args[6:12]
    team1_moves_raw = args[12:36] # 6 pokemon * 4 moves
    team2_moves_raw = args[36:60]
    team1_natures_raw = args[60:66]
    team2_natures_raw = args[66:72]
    team1_items_raw = args[72:78]
    team2_items_raw = args[78:84]

    stat_names_lower = ['hp', 'attack', 'defense', 'special-attack', 'special-defense', 'speed'] # Assume default EVs/IVs for now
    default_ivs = {stat.lower(): 31 for stat in stat_names}
    default_evs = {stat.lower(): 0 for stat in stat_names}
    default_nature = "Hardy"
    default_item = "None"

    # Helper function to prepare a team from a list of pokemon data and configurations
    def prepare_team(pokemon_data_list, moves_raw, natures_raw, items_raw, level, default_ivs, default_evs, default_nature, default_item):
        prepared_team = []
        for i, pokemon_data in enumerate(pokemon_data_list):
            if pokemon_data:
                selected_moves = [moves_raw[i * 4 + j] for j in range(4) if moves_raw[i * 4 + j] != "Select Move"]
                selected_nature = natures_raw[i] if i < len(natures_raw) and natures_raw[i] != "Select Nature" else default_nature
                selected_item = items_raw[i] if i < len(items_raw) and items_raw[i] != "None" else default_item

                prepared_pokemon = prepare_pokemon_for_battle(
                    pokemon_data,
                    level=level,
                    selected_moves=selected_moves,
                    selected_nature=selected_nature,
                    selected_item=selected_item,
                    iv_dict=default_ivs,
                    ev_dict=default_evs
                )
                if prepared_pokemon:
                    prepared_team.append(prepared_pokemon)
        return prepared_team

    def save_synergy_report_to_csv(synergy_report_data, team_name, filename='/tmp/battle_synergy_report.csv'):
        """Extracts synergy data and appends it to a CSV file."""
        if not synergy_report_data:
            print(f"No synergy data to save for {team_name}.")
            return

        offensive_coverage = analyze_offensive_coverage(synergy_report_data['team'], type_effectiveness_data)
        defensive_synergy = analyze_defensive_synergy(synergy_report_data['team'], type_effectiveness_data)
        synergy_score = calculate_team_synergy_score(offensive_coverage, defensive_synergy, synergy_report_data['team'])


        data_to_save = {
            'Team': [team_name],
            'Synergy Score': [synergy_score]
        }

        # Add offensive coverage data
        for target_type, coverage_info in offensive_coverage.items():
            data_to_save[f'Offensive_vs_{target_type.capitalize()}'] = [f"{coverage_info['max_effectiveness']:.1f}x"]
            data_to_save[f'Attacking_Types_vs_{target_type.capitalize()}'] = [", ".join([t.capitalize() for t in coverage_info['attacking_types']]) or "None"]

        # Add defensive synergy data
        for attacking_type, synergy_info in defensive_synergy.items():
            data_to_save[f'Defensive_vs_{attacking_type.capitalize()}'] = [f"{synergy_info['min_multiplier']:.1f}x"]
            data_to_save[f'Resisting_Pokemon_vs_{attacking_type.capitalize()}'] = [", ".join([p.capitalize() for p in synergy_info['resisting_pokemon']]) or "None"]


        df_to_save = pd.DataFrame(data_to_save)

        try:
            # Append to the CSV file
            header = not pd.io.common.file_exists(filename) # Write header only if file doesn't exist
            df_to_save.to_csv(filename, mode='a', index=False, header=header)
            print(f"Synergy data for {team_name} appended to {filename}")
        except Exception as e:
            print(f"Error saving synergy data to CSV: {e}")


    if button_id == 'simulate-button' and simulate_clicks > 0:
        # Simulate with UI Teams
        team1_data_base = [next((p for p in all_pokemon_data if p["Name"].lower() == name.lower()), None) for name in team1_configs_raw]
        team1_prepared = prepare_team(team1_data_base, team1_moves_raw, team1_natures_raw, team1_items_raw, level, default_ivs, default_evs, default_nature, default_item)

        team2_data_base = [next((p for p in all_pokemon_data if p["Name"].lower() == name.lower()), None) for name in team2_configs_raw]
        team2_prepared = prepare_team(team2_data_base, team2_moves_raw, team2_natures_raw, team2_items_raw, level, default_ivs, default_evs, default_nature, default_item)

        if not team1_prepared or not team2_prepared:
            return "Please select at least one Pokémon for each team."

        # Perform battle prediction
        prediction_result = predict_battle_outcome(team1_prepared, team2_prepared, level, num_simulations, type_effectiveness_data)

        # Save synergy reports to CSV
        save_synergy_report_to_csv({'team': team1_prepared}, 'Team 1 (UI)')
        save_synergy_report_to_csv({'team': team2_prepared}, 'Team 2 (UI)')


        # Display results
        output_elements = [
            html.H3("Battle Prediction Results (Team 1 vs Team 2 from UI)"),
            html.P(f"Predicted Outcome: {prediction_result.get('predicted_outcome', 'N/A')}"),
            html.P(f"Team 1 Wins: {prediction_result.get('team1_wins', 0)} ({prediction_result.get('team1_win_percentage', 0):.2f}%)"),
            html.P(f"Team 2 Wins: {prediction_result.get('team2_wins', 0)} ({prediction_result.get('team2_win_percentage', 0):.2f}%)"),
            html.P(f"Draws: {prediction_result.get('draws', 0)} ({prediction_result.get('draw_percentage', 0):.2f}%)"),
            html.P(f"Team 1 Synergy Score: {prediction_result.get('team1_synergy_score', 0):.2f}"),
            html.P(f"Team 2 Synergy Score: {prediction_result.get('team2_synergy_score', 0):.2f}"),

            html.H3("Synergy Report - Team 1"),
            dcc.Markdown(prediction_result.get('team1_synergy_report', 'No report generated.')),

            html.H3("Synergy Report - Team 2"),
            dcc.Markdown(prediction_result.get('team2_synergy_report', 'No report generated.')),
        ]
        return output_elements


    elif button_id == 'randomize-team2-simulate-button' and randomize_team2_clicks > 0:
         # Team 1 is from UI, Team 2 is randomized
         team1_data_base = [next((p for p in all_pokemon_data if p["Name"].lower() == name.lower()), None) for name in team1_configs_raw]
         team1_prepared = prepare_team(team1_data_base, team1_moves_raw, team1_natures_raw, team1_items_raw, level, default_ivs, default_evs, default_nature, default_item)

         if not team1_prepared:
              return "Please select at least one Pokémon for Team 1 to battle against a random team."

         if 'all_pokemon_data' not in globals() or not all_pokemon_data or len(all_pokemon_data) < len(team1_prepared) + 6:
             return "Not enough Pokémon data available to form a random Team 2."

         # Randomly select up to 6 unique Pokémon for Team 2 from all_pokemon_data, excluding those already in Team 1
         team1_names_list = [p['Name'] for p in team1_prepared]
         remaining_pokemon_for_random = [p for p in all_pokemon_data if p['Name'] not in team1_names_list]

         if len(remaining_pokemon_for_random) < 6:
              return "Not enough unique Pokémon data available to form a random Team 2."

         random_team2_data_base = random.sample(remaining_pokemon_for_random, 6)

         # Prepare random Team 2 (random moves, nature, item, default EVs/IVs)
         def prepare_random_pokemon(pokemon_data, level, default_ivs, default_evs, common_natures, held_item_options):
             if not pokemon_data: return None
             available_moves = pokemon_data.get("Moves", [])
             selected_moves = random.sample(available_moves, min(4, len(available_moves)))
             selected_nature = random.choice(common_natures)
             selected_item = random.choice(held_item_options)

             return prepare_pokemon_for_battle(
                 pokemon_data,
                 level=level,
                 selected_moves=selected_moves,
                 selected_nature=selected_nature,
                 selected_item=selected_item,
                 iv_dict=default_ivs,
                 ev_dict=default_evs
             )

         team2_prepared = [prepare_random_pokemon(p_data, level, default_ivs, default_evs, common_natures, held_item_options) for p_data in random_team2_data_base]
         team2_prepared = [p for p in team2_prepared if p] # Filter out any None values


         if not team2_prepared or len(team2_prepared) < 1: # Ensure at least one prepared pokemon
              return "Could not prepare random Team 2 for battle simulation."

         # Perform battle prediction (Team 1 from UI vs Random Team 2)
         prediction_result = predict_battle_outcome(team1_prepared, team2_prepared, level, num_simulations, type_effectiveness_data)

         # Save synergy reports to CSV
         save_synergy_report_to_csv({'team': team1_prepared}, 'Team 1 (UI)')
         save_synergy_report_to_csv({'team': team2_prepared}, 'Team 2 (Random)')


         # Display results for Team 1 vs Random Team 2
         output_elements = [
             html.H3("Battle Prediction Results (Team 1 from UI vs Random Team 2)"),
             html.P(f"Team 1: {[p['Name'].capitalize() for p in team1_prepared]}"),
             html.P(f"Random Team 2: {[p['Name'].capitalize() for p in team2_prepared]}"),
             html.P(f"Predicted Outcome: {prediction_result.get('predicted_outcome', 'N/A')}"),
             html.P(f"Team 1 Wins: {prediction_result.get('team1_wins', 0)} ({prediction_result.get('team1_win_percentage', 0):.2f}%)"),
             html.P(f"Team 2 Wins: {prediction_result.get('team2_wins', 0)} ({prediction_result.get('team2_win_percentage', 0):.2f}%)"),
             html.P(f"Draws: {prediction_result.get('draws', 0)} ({prediction_result.get('draw_percentage', 0):.2f}%)"),
             html.P(f"Team 1 Synergy Score: {prediction_result.get('team1_synergy_score', 0):.2f}"),
             html.P(f"Team 2 Synergy Score: {prediction_result.get('team2_synergy_score', 0):.2f}"),

             html.H3("Synergy Report - Team 1"),
             dcc.Markdown(prediction_result.get('team1_synergy_report', 'No report generated.')),

             html.H3("Synergy Report - Team 2 (Randomized)"),
             dcc.Markdown(prediction_result.get('team2_synergy_report', 'No report generated.')),
         ]
         return output_elements

    elif button_id == 'randomize-both-simulate-button' and randomize_both_clicks > 0:
        # Randomize BOTH teams and simulate
        if 'all_pokemon_data' not in globals() or len(all_pokemon_data) < 12:
             return "Not enough Pokémon data available to form two random teams."

        # Randomly select up to 6 unique Pokémon for each team
        random_team1_data_base = random.sample(all_pokemon_data, min(6, len(all_pokemon_data) // 2))
        remaining_pokemon = [p for p in all_pokemon_data if p not in random_team1_data_base]
        random_team2_data_base = random.sample(remaining_pokemon, min(6, len(remaining_pokemon)))


        # Prepare both random teams (random moves, nature, item, default EVs/IVs)
        def prepare_random_pokemon(pokemon_data, level, default_ivs, default_evs, common_natures, held_item_options):
             if not pokemon_data: return None
             available_moves = pokemon_data.get("Moves", [])
             selected_moves = random.sample(available_moves, min(4, len(available_moves)))
             selected_nature = random.choice(common_natures)
             selected_item = random.choice(held_item_options)

             return prepare_pokemon_for_battle(
                 pokemon_data,
                 level=level,
                 selected_moves=selected_moves,
                 selected_nature=selected_nature,
                 selected_item=selected_item,
                 iv_dict=default_ivs,
                 ev_dict=default_evs
             )

        team1_prepared = [prepare_random_pokemon(p_data, level, default_ivs, default_evs, common_natures, held_item_options) for p_data in random_team1_data_base]
        team1_prepared = [p for p in team1_prepared if p] # Filter out any None values

        team2_prepared = [prepare_random_pokemon(p_data, level, default_ivs, default_evs, common_natures, held_item_options) for p_data in random_team2_data_base]
        team2_prepared = [p for p in team2_prepared if p] # Filter out any None values


        if not team1_prepared or not team2_prepared or len(team1_prepared) < 1 or len(team2_prepared) < 1:
             return "Could not prepare two random teams for battle simulation."


        # Perform battle prediction (Random Team 1 vs Random Team 2)
        prediction_result = predict_battle_outcome(team1_prepared, team2_prepared, level, num_simulations, type_effectiveness_data)

        # Save synergy reports to CSV
        save_synergy_report_to_csv({'team': team1_prepared}, 'Team 1 (Random)')
        save_synergy_report_to_csv({'team': team2_prepared}, 'Team 2 (Random)')


        # Display results for Random Team 1 vs Random Team 2
        output_elements = [
            html.H3("Battle Prediction Results (Random Team 1 vs Random Team 2)"),
            html.P(f"Random Team 1: {[p['Name'].capitalize() for p in team1_prepared]}"),
            html.P(f"Random Team 2: {[p['Name'].capitalize() for p in team2_prepared]}"),
            html.P(f"Predicted Outcome: {prediction_result.get('predicted_outcome', 'N/A')}"),
            html.P(f"Team 1 Wins: {prediction_result.get('team1_wins', 0)} ({prediction_result.get('team1_win_percentage', 0):.2f}%)"),
            html.P(f"Team 2 Wins: {prediction_result.get('team2_wins', 0)} ({prediction_result.get('team2_win_percentage', 0):.2f}%)"),
            html.P(f"Draws: {prediction_result.get('draws', 0)} ({prediction_result.get('draw_percentage', 0):.2f}%)"),
             html.P(f"Team 1 Synergy Score: {prediction_result.get('team1_synergy_score', 0):.2f}"),
            html.P(f"Team 2 Synergy Score: {prediction_result.get('team2_synergy_score', 0):.2f}"),

            html.H3("Synergy Report - Team 1 (Randomized)"),
            dcc.Markdown(prediction_result.get('team1_synergy_report', 'No report generated.')),

            html.H3("Synergy Report - Team 2 (Randomized)"),
            dcc.Markdown(prediction_result.get('team2_synergy_report', 'No report generated.')),
        ]
        return output_elements


    return "Select teams and simulate a battle."

#To run the app,
#use `app.run_server(mode='inline')` in a separate cell.
#This cell only defines the app structure and callbacks.

app.run(mode='inline')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define the list of Pokémon names
user_pokemon_names = ['venusaur', 'blastoise', 'charizard', 'gardevoir', 'goodra', 'aggron']

# Fetch the data for the specified Pokémon (re-fetching to ensure we have the latest)
user_pokemon_data = []
for name in user_pokemon_names:
    pokemon_data = next((p for p in all_pokemon_data if p["Name"].lower() == name.lower()), None)
    if pokemon_data:
        user_pokemon_data.append(pokemon_data)
    else:
        print(f"Warning: Data not found for Pokémon: {name}")

if not user_pokemon_data:
    print("Error: Could not retrieve data for any of the specified Pokémon to create the plot.")
else:
    combined_plotting_data = []
    stats_to_plot = ['HP', 'Attack', 'Defense', 'Special-attack', 'Special-defense', 'Speed']

    for pokemon in user_pokemon_data:
        pokemon_name = pokemon.get('Name', 'Unknown').capitalize()
        for stat_name_full in stats_to_plot:
            # Get the stat value, handling potential API name differences ('special-attack' vs 'Special-attack')
            stat_value = pokemon['Base Stats'].get(stat_name_full.lower(), 0)
            combined_plotting_data.append({
                'Pokemon': pokemon_name,
                'Stat Type': stat_name_full,
                'Stat Value': stat_value
            })

    if combined_plotting_data:
        combined_plotting_df = pd.DataFrame(combined_plotting_data)

        # Create a single 1D scatter plot
        plt.figure(figsize=(12, 6)) # Adjust figure size for better readability
        sns.scatterplot(data=combined_plotting_df, x='Stat Value', y='Pokemon', hue='Stat Type', s=100, zorder=10) # zorder to ensure points are above lines if any

        plt.title('Combined Scatter Plot of Base Stats for Selected Pokémon')
        plt.xlabel('Stat Value')
        plt.ylabel('Pokémon')
        plt.grid(axis='x', linestyle='--', alpha=0.7) # Add vertical grid lines
        plt.legend(title='Stat Type', bbox_to_anchor=(1.05, 1), loc='upper left') # Move legend outside plot
        plt.tight_layout() # Adjust layout to prevent labels overlapping
        plt.show()

    else:
        print("Could not prepare plotting data for the specified Pokémon.")
