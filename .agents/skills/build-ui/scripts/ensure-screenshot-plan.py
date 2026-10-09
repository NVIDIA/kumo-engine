#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PRIMARY_RECIPES_PATH = SCRIPT_DIR.parent.parent.parent.parent / "ui" / "src" / "screenshot" / "catalog.json"
ROUTE_MAP_PATH = SCRIPT_DIR.parent.parent.parent.parent / "ui" / "src" / "screenshot" / "route-map.json"


def get_recipes_path():
    if PRIMARY_RECIPES_PATH.exists():
        return PRIMARY_RECIPES_PATH
    raise FileNotFoundError(
        f"Screenshot catalog not found: {PRIMARY_RECIPES_PATH}")


def load_recipes():
    recipes_path = get_recipes_path()
    data = json.loads(recipes_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{recipes_path} must contain a list")

    recipes = []
    for recipe in data:
        if not isinstance(recipe, dict):
            continue
        patterns = recipe.get("patterns") or []
        canonical_routes = set(recipe.get("canonicalRoutes") or [])
        legacy_routes = set(recipe.get("legacyRoutes") or [])
        if not recipe.get("key") or not patterns or not canonical_routes:
            continue
        recipes.append({
            "key": recipe["key"],
            "patterns": patterns,
            "canonical_routes": canonical_routes,
            "legacy_routes": legacy_routes,
            "visual_coverage": recipe.get("visualCoverage") or [],
            "required_feature_flags": recipe.get("requiredFeatureFlags") or {},
            "plan": recipe.get("plan") or {},
        })
    return recipes


def load_route_map():
    if not ROUTE_MAP_PATH.exists():
        return []

    data = json.loads(ROUTE_MAP_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{ROUTE_MAP_PATH} must contain a list")

    mappings = []
    for mapping in data:
        if not isinstance(mapping, dict):
            continue
        patterns = mapping.get("patterns") or []
        recipe_keys = mapping.get("recipeKeys") or []
        routes = mapping.get("routes") or []
        if not mapping.get("key") or not patterns or not recipe_keys:
            continue
        mappings.append({
            "key": mapping["key"],
            "description": mapping.get("description") or "",
            "patterns": patterns,
            "recipe_keys": recipe_keys,
            "routes": routes,
        })
    return mappings


def matching_files_for_recipe(changed_files, recipe):
    compiled = [re.compile(pattern) for pattern in recipe["patterns"]]
    return [
        path for path in changed_files if any(
            regex.search(path) for regex in compiled)
    ]


def matched_files_for_mapping(changed_files, mapping):
    compiled = [re.compile(pattern) for pattern in mapping["patterns"]]
    return [
        path for path in changed_files if any(
            regex.search(path) for regex in compiled)
    ]


def matched_surfaces(changed_files, recipes, route_map=None):
    matches_by_key = {}
    for recipe in recipes:
        matched_files = matching_files_for_recipe(changed_files, recipe)
        if matched_files:
            recipe_with_matches = dict(recipe)
            recipe_with_matches["matched_files"] = matched_files
            recipe_with_matches["match_sources"] = ["catalog-patterns"]
            matches_by_key[recipe["key"]] = recipe_with_matches

    recipes_by_key = {recipe["key"]: recipe for recipe in recipes}
    for mapping in route_map or []:
        matched_files = matched_files_for_mapping(changed_files, mapping)
        if not matched_files:
            continue
        for recipe_key in mapping["recipe_keys"]:
            recipe = recipes_by_key.get(recipe_key)
            if not recipe:
                continue
            existing = matches_by_key.get(recipe_key)
            if existing is None:
                existing = dict(recipe)
                existing["matched_files"] = []
                existing["match_sources"] = []
                matches_by_key[recipe_key] = existing
            existing["matched_files"] = sorted(
                set(existing.get("matched_files", [])) | set(matched_files))
            existing["match_sources"] = sorted(
                set(existing.get("match_sources", [])) | {mapping["key"]})
            existing.setdefault("matched_route_map", []).append({
                "key":
                mapping["key"],
                "description":
                mapping["description"],
                "routes":
                mapping["routes"],
                "matchedFiles":
                matched_files,
            })

    return sorted(
        matches_by_key.values(),
        key=lambda surface: (-len(surface["canonical_routes"]), surface["key"]),
    )


def read_existing_plan(plan_path):
    if not plan_path.exists():
        return None
    try:
        return json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_override(override_path):
    if not override_path:
        return {}
    path = Path(override_path)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"invalid screenshot override JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("screenshot override must be a JSON object")
    return data


def collect_routes(plan):
    routes = set()
    if not isinstance(plan, dict):
        return routes
    for route in plan.get("routes", []) or []:
        if isinstance(route, str):
            routes.add(route.strip())
    for scenario in plan.get("scenarios", []) or []:
        if isinstance(scenario, dict) and isinstance(scenario.get("route"),
                                                     str):
            routes.add(scenario["route"].strip())
    return routes


def route_path(route):
    return route.split("?", 1)[0]


def route_matches_allowed(route, allowed_route):
    current = route_path(route)
    allowed = route_path(allowed_route)
    return current == allowed or (allowed != "/"
                                  and current.startswith(allowed))


def normalize_covers(value):
    if isinstance(value, str) and value.strip():
        return {value.strip()}
    if isinstance(value, list):
        return {
            item.strip()
            for item in value if isinstance(item, str) and item.strip()
        }
    return set()


def collect_plan_covers(plan):
    covers = set()
    if not isinstance(plan, dict):
        return covers
    for scenario in plan.get("scenarios", []) or []:
        if not isinstance(scenario, dict):
            continue
        for action in scenario.get("setup", []) or []:
            if isinstance(action, dict):
                covers.update(normalize_covers(action.get("covers")))
        for assertion in scenario.get("assertions", []) or []:
            if isinstance(assertion, dict):
                covers.update(normalize_covers(assertion.get("covers")))
    return covers


def collect_item_covers(items):
    covers = set()
    for item in items or []:
        if isinstance(item, dict):
            covers.update(normalize_covers(item.get("covers")))
    return covers


def coverage_for_ids(surface, coverage_ids):
    return [
        item for item in surface.get("visual_coverage", [])
        if isinstance(item, dict) and item.get("id") in coverage_ids
    ]


def scenario_visual_coverage(surface, scenario):
    coverage_ids = set()
    if isinstance(scenario, dict):
        coverage_ids.update(collect_item_covers(scenario.get("setup")))
        coverage_ids.update(collect_item_covers(scenario.get("assertions")))
    matched = coverage_for_ids(surface, coverage_ids)
    return matched or surface.get("visual_coverage", [])


def combined_surface(surfaces):
    coverage_by_id = {}
    required_feature_flags = {}
    for surface in surfaces:
        for coverage in surface.get("visual_coverage", []):
            if isinstance(coverage, dict) and isinstance(
                    coverage.get("id"), str):
                coverage_by_id.setdefault(coverage["id"], coverage)
        for flag, expected in surface.get("required_feature_flags", {}).items():
            if isinstance(flag, str) and isinstance(expected, bool):
                required_feature_flags[flag] = expected

    return {
        "key":
        "multi" if len(surfaces) > 1 else surfaces[0]["key"],
        "canonical_routes":
        set().union(*(surface["canonical_routes"] for surface in surfaces)),
        "legacy_routes":
        set().union(*(surface["legacy_routes"] for surface in surfaces)),
        "visual_coverage":
        list(coverage_by_id.values()),
        "required_feature_flags":
        required_feature_flags,
    }


def verify_plan(plan, surface=None):
    routes = collect_routes(plan)
    messages = []
    if not isinstance(plan, dict):
        return {
            "status": "failed",
            "messages": ["plan is missing or invalid JSON object"],
        }
    if not routes:
        messages.append("plan has no routes or scenario routes")

    if surface is not None:
        allowed_routes = {
            route_path(route)
            for route in (surface["canonical_routes"]
                          | surface["legacy_routes"])
        }
        if routes and not any(
                route_matches_allowed(route, allowed_route) for route in routes
                for allowed_route in allowed_routes):
            messages.append("plan routes do not match selected recipe routes")
        expected_coverage = {
            item.get("id")
            for item in surface.get("visual_coverage", [])
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        covered = collect_plan_covers(plan)
        missing_coverage = sorted(expected_coverage - covered)
        unknown_coverage = sorted(covered - expected_coverage)
        if missing_coverage:
            messages.append(
                f"missing coverage ids: {', '.join(missing_coverage)}")
        if unknown_coverage:
            messages.append(
                f"unknown coverage ids: {', '.join(unknown_coverage)}")

    return {
        "status": "passed" if not messages else "failed",
        "messages": messages,
    }


def plan_routes_match_surface(plan, surface):
    routes = collect_routes(plan)
    if not routes:
        return False
    allowed_routes = surface["canonical_routes"] | surface["legacy_routes"]
    return any(
        route_matches_allowed(route, allowed_route) for route in routes
        for allowed_route in allowed_routes)


def write_plan(plan_path, payload):
    plan_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def plan_for_surface(surface):
    payload = deepcopy(surface["plan"])
    payload.setdefault("_recipeKey", surface["key"])
    payload.setdefault("_visualCoverage", surface["visual_coverage"])
    if surface.get("required_feature_flags"):
        payload.setdefault("_requiredFeatureFlags",
                           surface["required_feature_flags"])
    if isinstance(payload.get("scenarios"), list):
        for scenario in payload["scenarios"]:
            if isinstance(scenario, dict):
                scenario.setdefault("visualCoverage",
                                    scenario_visual_coverage(surface, scenario))
                if surface.get("required_feature_flags"):
                    scenario.setdefault("requiredFeatureFlags",
                                        surface["required_feature_flags"])
    return payload


def surface_subsumes(candidate, surface):
    if candidate["key"] == surface["key"]:
        return False

    candidate_files = set(candidate.get("matched_files", []))
    surface_files = set(surface.get("matched_files", []))
    if not surface_files or not candidate_files.issuperset(surface_files):
        return False

    if not candidate["canonical_routes"].issuperset(
            surface["canonical_routes"]):
        return False

    if (candidate_files == surface_files
            and candidate["canonical_routes"] == surface["canonical_routes"]):
        return candidate["key"] < surface["key"]

    return True


def prune_subsumed_surfaces(surfaces):
    return [
        surface for surface in surfaces if not any(
            surface_subsumes(candidate, surface) for candidate in surfaces)
    ]


def plan_for_surfaces(surfaces):
    if len(surfaces) == 1:
        return plan_for_surface(surfaces[0])

    scenarios = []
    seen = set()
    coverage_by_id = {}
    recipe_keys = [surface["key"] for surface in surfaces]

    for surface in surfaces:
        surface_plan = plan_for_surface(surface)
        for coverage in surface.get("visual_coverage", []):
            if isinstance(coverage, dict) and isinstance(
                    coverage.get("id"), str):
                coverage_by_id.setdefault(coverage["id"], coverage)

        for scenario in surface_plan.get("scenarios", []) or []:
            if not isinstance(scenario, dict):
                continue
            scenario_payload = deepcopy(scenario)
            scenario_payload.setdefault(
                "visualCoverage",
                scenario_visual_coverage(surface, scenario_payload))
            scenario_key = json.dumps(
                {
                    "name":
                    scenario_payload.get("name"),
                    "route":
                    scenario_payload.get("route"),
                    "requiredFeatureFlags":
                    scenario_payload.get("requiredFeatureFlags"),
                    "setup":
                    scenario_payload.get("setup"),
                    "assertions":
                    scenario_payload.get("assertions"),
                    "readySelector":
                    scenario_payload.get("readySelector"),
                    "readySelectors":
                    scenario_payload.get("readySelectors"),
                },
                sort_keys=True,
            )
            if scenario_key in seen:
                continue
            seen.add(scenario_key)
            scenarios.append(scenario_payload)

    return {
        "scenarios":
        scenarios,
        "_recipeKey":
        "multi",
        "_recipeKeys":
        recipe_keys,
        "_visualCoverage":
        list(coverage_by_id.values()),
        "_rationale":
        "Merged from multiple catalog recipes because the diff matched multiple screenshot surfaces.",
    }


def scenario_key(scenario_payload):
    return json.dumps(
        {
            "name": scenario_payload.get("name"),
            "route": scenario_payload.get("route"),
            "requiredFeatureFlags":
            scenario_payload.get("requiredFeatureFlags"),
            "setup": scenario_payload.get("setup"),
            "assertions": scenario_payload.get("assertions"),
            "readySelector": scenario_payload.get("readySelector"),
            "readySelectors": scenario_payload.get("readySelectors"),
        },
        sort_keys=True,
    )


def scenario_matches_surface(scenario, surface):
    route = scenario.get("route")
    if isinstance(route, str) and route.strip():
        allowed_routes = surface["canonical_routes"] | surface["legacy_routes"]
        if any(
                route_matches_allowed(route, allowed_route)
                for allowed_route in allowed_routes):
            return True

    expected_coverage = {
        item.get("id")
        for item in surface.get("visual_coverage", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    return bool(expected_coverage
                & collect_plan_covers({"scenarios": [scenario]}))


def surface_coverage_ids(surface):
    return {
        item.get("id")
        for item in surface.get("visual_coverage", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }


def scenario_has_complete_surface_coverage(scenario, surfaces, covered_ids):
    coverage_matched_surfaces = [
        surface for surface in surfaces
        if covered_ids & surface_coverage_ids(surface)
    ]
    if not coverage_matched_surfaces:
        return False

    return any(
        surface_coverage_ids(surface) <= covered_ids
        for surface in coverage_matched_surfaces)


def surfaces_matching_coverage(surfaces, covered_ids):
    return [
        surface for surface in surfaces
        if covered_ids & surface_coverage_ids(surface)
    ]


def known_coverage_by_id(surfaces):
    coverage_by_id = {}
    for surface in surfaces:
        for coverage in surface.get("visual_coverage", []):
            if isinstance(coverage, dict) and isinstance(
                    coverage.get("id"), str):
                coverage_by_id.setdefault(coverage["id"], coverage)
    return coverage_by_id


def catalog_scenario_keys(surfaces):
    scenarios = plan_for_surfaces(surfaces).get("scenarios", []) or []
    return {
        scenario_key(scenario)
        for scenario in scenarios if isinstance(scenario, dict)
    }


def existing_primary_scenarios(existing, surfaces, catalog_keys=None):
    if not isinstance(existing, dict):
        return []
    scenarios = existing.get("scenarios")
    if not isinstance(scenarios, list):
        return []

    primary_scenarios = []
    catalog_primary_surface_keys = set()
    coverage_by_id = known_coverage_by_id(surfaces)
    known_coverage_ids = set(coverage_by_id)
    catalog_keys = catalog_keys or set()
    focused_existing_plan = len(
        [scenario for scenario in scenarios if isinstance(scenario, dict)]) <= 3

    for scenario in scenarios:
        if not isinstance(scenario, dict):
            continue
        if scenario_key(scenario) in catalog_keys and not focused_existing_plan:
            continue

        covered_ids = collect_plan_covers({"scenarios": [scenario]})
        if not covered_ids or not covered_ids <= known_coverage_ids:
            continue

        if not scenario_has_complete_surface_coverage(scenario, surfaces,
                                                      covered_ids):
            catalog_primary_surface_keys.update(
                surface["key"] for surface in surfaces_matching_coverage(
                    surfaces, covered_ids))
            continue

        if not any(
                scenario_matches_surface(scenario, surface)
                for surface in surfaces):
            continue

        scenario_payload = deepcopy(scenario)
        scenario_payload["presentation"] = "primary"
        scenario_payload["visualCoverage"] = [
            coverage_by_id[coverage_id] for coverage_id in sorted(covered_ids)
        ]
        primary_scenarios.append(scenario_payload)
    return primary_scenarios, catalog_primary_surface_keys


def scenario_matches_surface_key_by_coverage(scenario, surfaces, surface_keys):
    if not surface_keys:
        return False

    covered_ids = collect_plan_covers({"scenarios": [scenario]})
    return any(
        surface["key"] in surface_keys and bool(covered_ids
                                                & surface_coverage_ids(surface))
        for surface in surfaces)


def plan_for_surfaces_with_primary(existing, surfaces):
    payload = plan_for_surfaces(surfaces)
    primary_scenarios, catalog_primary_surface_keys = (
        existing_primary_scenarios(existing, surfaces,
                                   catalog_scenario_keys(surfaces)))
    if not primary_scenarios:
        if not catalog_primary_surface_keys:
            return payload

    scenarios = []
    seen = set()
    for scenario in primary_scenarios:
        key = scenario_key(scenario)
        if key in seen:
            continue
        seen.add(key)
        scenarios.append(scenario)

    for scenario in payload.get("scenarios", []) or []:
        if not isinstance(scenario, dict):
            continue
        scenario_payload = deepcopy(scenario)
        if scenario_matches_surface_key_by_coverage(
                scenario_payload, surfaces, catalog_primary_surface_keys):
            scenario_payload.setdefault("presentation", "primary")
        else:
            scenario_payload.setdefault("presentation", "regression")
        key = scenario_key(scenario_payload)
        if key in seen:
            continue
        seen.add(key)
        scenarios.append(scenario_payload)

    payload["scenarios"] = scenarios
    payload[
        "_rationale"] = "Preserved agent-authored ticket evidence as primary and appended catalog regression coverage for other matched surfaces."
    return payload


def canonical_plan_equal(left, right):
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def normalize_string_list(value):
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list):
        return [
            item.strip() for item in value
            if isinstance(item, str) and item.strip()
        ]
    return []


def select_surface_from_override(override, recipes):
    if not override:
        return None

    recipe_key = override.get("recipeKey")
    route = override.get("route")
    coverage_ids = set(normalize_string_list(override.get("coverageIds")))

    candidates = recipes
    if isinstance(recipe_key, str) and recipe_key.strip():
        candidates = [
            recipe for recipe in candidates
            if recipe["key"] == recipe_key.strip()
        ]

    if isinstance(route, str) and route.strip():
        candidates = [
            recipe for recipe in candidates if any(
                route_matches_allowed(route.strip(), recipe_route)
                for recipe_route in (recipe["canonical_routes"]
                                     | recipe["legacy_routes"]))
        ]

    if coverage_ids:
        candidates = [
            recipe for recipe in candidates if coverage_ids <= {
                coverage.get("id")
                for coverage in recipe.get("visual_coverage", [])
                if isinstance(coverage, dict)
            }
        ]

    if not candidates:
        return None

    selected = dict(candidates[0])
    selected["matched_files"] = []
    selected["match_sources"] = ["explicit-override"]
    return selected


def apply_override_to_plan(plan, override):
    if not override:
        return plan

    payload = deepcopy(plan)
    metadata = {}
    route = override.get("route")
    fixtures = normalize_string_list(override.get("fixtures"))
    coverage_ids = normalize_string_list(override.get("coverageIds"))
    if isinstance(route, str) and route.strip():
        metadata["route"] = route.strip()
    if fixtures:
        metadata["fixtures"] = fixtures
    if coverage_ids:
        metadata["coverageIds"] = coverage_ids
    if metadata:
        payload["_screenshotOverride"] = metadata

    scenarios = payload.get("scenarios")
    if isinstance(scenarios, list):
        for scenario in scenarios:
            if isinstance(scenario, dict):
                if fixtures:
                    existing_fixtures = normalize_string_list(
                        scenario.get("fixtures"))
                    scenario["fixtures"] = list(
                        dict.fromkeys(existing_fixtures + fixtures))
                if isinstance(route, str) and route.strip():
                    scenario["route"] = route.strip()
        return payload

    if isinstance(route, str) and route.strip() and isinstance(
            payload.get("routes"), list):
        payload["routes"] = [route.strip()]

    return payload


def select_surface_for_existing_plan(surfaces, existing):
    exact_matches = []
    compatible_matches = []
    for surface in surfaces:
        expected_plan = plan_for_surface(surface)
        if existing == expected_plan:
            exact_matches.append(surface)
            continue
        if plan_routes_match_surface(existing, surface):
            compatible_matches.append(surface)

    if exact_matches:
        return exact_matches[0], "exact"
    if compatible_matches:
        return compatible_matches[0], "compatible"
    return surfaces[0], "incompatible"


def write_resolution_report(report_path, report):
    if not report_path:
        return
    Path(report_path).write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )


def base_report(plan_path, changed_files, surfaces, existing):
    return {
        "generatedAt":
        datetime.now(timezone.utc).isoformat(),
        "planPath":
        str(plan_path),
        "changedFiles":
        changed_files,
        "existingPlanPresent":
        existing is not None,
        "matchedRecipes": [{
            "key":
            surface["key"],
            "matchedFiles":
            surface.get("matched_files", []),
            "matchSources":
            surface.get("match_sources", []),
            "matchedRouteMap":
            surface.get("matched_route_map", []),
            "canonicalRoutes":
            sorted(surface["canonical_routes"]),
            "legacyRoutes":
            sorted(surface["legacy_routes"]),
            "visualCoverage":
            surface["visual_coverage"],
        } for surface in surfaces],
        "selectedRecipe":
        None,
        "selectedRecipes": [],
        "screenshotOverride":
        None,
        "action":
        "skipped",
        "reason":
        "no matching recipe",
        "verification": {
            "status": "skipped",
            "messages": [],
        },
    }


def main():
    if len(sys.argv) not in (3, 4, 5):
        print(
            "usage: ensure-screenshot-plan.py <plan-path> <changed-files-path> [resolution-report-path] [screenshot-override-path]",
            file=sys.stderr,
        )
        return 2

    plan_path = Path(sys.argv[1])
    changed_files_path = Path(sys.argv[2])
    report_path = Path(sys.argv[3]) if len(sys.argv) >= 4 else None
    override_path = Path(sys.argv[4]) if len(sys.argv) == 5 else None
    recipes = load_recipes()
    route_map = load_route_map()
    override = read_override(override_path)
    changed_files = [
        line.strip()
        for line in changed_files_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    surfaces = matched_surfaces(changed_files, recipes, route_map)
    surfaces = prune_subsumed_surfaces(surfaces)
    existing = read_existing_plan(plan_path)
    report = base_report(plan_path, changed_files, surfaces, existing)
    report["screenshotOverride"] = override or None

    override_surface = select_surface_from_override(override, recipes)
    if override and override_surface is None:
        report["action"] = "failed-override"
        report[
            "reason"] = "explicit screenshot override did not match a catalog recipe"
        report["verification"] = {
            "status":
            "failed",
            "messages": [
                "screenshot override must match a catalog recipe by recipe, route, and coverage ids"
            ],
        }
        write_resolution_report(report_path, report)
        print(f"failed-override:{plan_path}")
        return 1

    if override_surface is not None:
        selected_plan = apply_override_to_plan(
            plan_for_surface(override_surface), override)
        write_plan(plan_path, selected_plan)
        report["selectedRecipe"] = override_surface["key"]
        report["selectedRecipes"] = [override_surface["key"]]
        report["action"] = "wrote-override"
        report[
            "reason"] = "explicit screenshot override selected catalog recipe"
        report["verification"] = verify_plan(selected_plan, override_surface)
        write_resolution_report(report_path, report)
        print(f"wrote-override:{plan_path}:{override_surface['key']}")
        return 0
    if not surfaces:
        if existing is not None:
            report["verification"] = verify_plan(existing)
        write_resolution_report(report_path, report)
        print(f"skipped:{plan_path}")
        return 0

    routes = collect_routes(existing)

    if existing is None or not routes:
        selected_plan = plan_for_surfaces(surfaces)
        write_plan(plan_path, selected_plan)
        report["selectedRecipe"] = (surfaces[0]["key"]
                                    if len(surfaces) == 1 else "multi")
        report["selectedRecipes"] = [surface["key"] for surface in surfaces]
        report["action"] = "wrote"
        report["reason"] = "missing or empty existing plan"
        report["verification"] = verify_plan(selected_plan,
                                             combined_surface(surfaces))
        write_resolution_report(report_path, report)
        print(
            f"wrote:{plan_path}:{','.join(surface['key'] for surface in surfaces)}"
        )
        return 0

    allowed_routes = set().union(*(surface["canonical_routes"]
                                   for surface in surfaces))
    legacy_routes = set().union(*(surface["legacy_routes"]
                                  for surface in surfaces))

    if any(
            route_matches_allowed(route, legacy_route) for route in routes
            for legacy_route in legacy_routes):
        selected_plan = plan_for_surfaces(surfaces)
        write_plan(plan_path, selected_plan)
        report["selectedRecipe"] = (surfaces[0]["key"]
                                    if len(surfaces) == 1 else "multi")
        report["selectedRecipes"] = [surface["key"] for surface in surfaces]
        report["action"] = "rewrote-legacy"
        report["reason"] = "existing plan used a legacy route"
        report["verification"] = verify_plan(selected_plan,
                                             combined_surface(surfaces))
        write_resolution_report(report_path, report)
        print(
            f"rewrote-legacy:{plan_path}:{','.join(surface['key'] for surface in surfaces)}"
        )
        return 0

    if len(surfaces) == 1:
        surface = surfaces[0]
        expected_plan = plan_for_surface(surface)
        if existing != expected_plan:
            write_plan(plan_path, expected_plan)
            reason = "mismatch" if not (routes & allowed_routes) else "recipe"
            report["selectedRecipe"] = surface["key"]
            report["selectedRecipes"] = [surface["key"]]
            report["action"] = f"rewrote-{reason}"
            report["reason"] = "single matching recipe overrides agent plan"
            report["verification"] = verify_plan(expected_plan, surface)
            write_resolution_report(report_path, report)
            print(f"rewrote-{reason}:{plan_path}:{surface['key']}")
            return 0

    expected_plan = plan_for_surfaces(surfaces)
    report["selectedRecipe"] = (surfaces[0]["key"]
                                if len(surfaces) == 1 else "multi")
    report["selectedRecipes"] = [surface["key"] for surface in surfaces]

    if canonical_plan_equal(existing, expected_plan):
        report["action"] = "kept"
        report["reason"] = "existing plan exactly matches selected recipes"
        report["verification"] = verify_plan(existing,
                                             combined_surface(surfaces))
        write_resolution_report(report_path, report)
        print(
            f"kept:{plan_path}:{','.join(surface['key'] for surface in surfaces)}"
        )
        return 0

    selected_plan = plan_for_surfaces_with_primary(existing, surfaces)
    if any(
            isinstance(scenario, dict)
            and scenario.get("presentation") == "primary"
            for scenario in selected_plan.get("scenarios", []) or []):
        write_plan(plan_path, selected_plan)
        report["action"] = "merged-primary"
        report[
            "reason"] = "preserved compatible agent-authored scenarios as primary evidence and appended catalog regression coverage"
        report["verification"] = verify_plan(selected_plan,
                                             combined_surface(surfaces))
        write_resolution_report(report_path, report)
        print(
            f"merged-primary:{plan_path}:{','.join(surface['key'] for surface in surfaces)}"
        )
        return 0

    write_plan(plan_path, expected_plan)
    report["action"] = "rewrote-multi"
    report[
        "reason"] = "multiple matching recipes require a merged catalog-backed plan"
    report["verification"] = verify_plan(expected_plan,
                                         combined_surface(surfaces))
    write_resolution_report(report_path, report)
    print(
        f"rewrote-multi:{plan_path}:{','.join(surface['key'] for surface in surfaces)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
