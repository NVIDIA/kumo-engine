# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect
from pathlib import Path

import pytest
from temporalio import activity

# Import worker modules to load every production connector activity binding.
import pool.activities  # noqa: F401
import rest.backends.data_backend.worker  # noqa: F401
import rest.backends.ml_backend.worker  # noqa: F401
from rest.backends.data_backend import connector_activity_registry
from rest.backends.data_backend.activities import connector_authorization
from statedb.rest.connector_authorization import (
    CONNECTOR_ACTIVITY_AUTHORIZATION_ACTIONS,
    CONNECTOR_DATABRICKS_BATCH_PREDICTION_OUTPUT_ACTIONS,
    CONNECTOR_GRAPH_ACTIONS,
    CONNECTOR_ISSUABLE_ACTIONS,
    CONNECTOR_REFERENCE_ACTIONS,
    CONNECTOR_TABLE_SOURCE_ACTIONS,
    ConnectorCapabilityAction,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_WORKER_SOURCES = (
    _REPOSITORY_ROOT / 'rest/backends/data_backend/worker.py',
    _REPOSITORY_ROOT / 'rest/backends/ml_backend/worker.py',
    _REPOSITORY_ROOT / 'pool/worker_main.py',
)
_AUDIT_ROOTS = (
    _REPOSITORY_ROOT / 'rest',
    _REPOSITORY_ROOT / 'pool',
    _REPOSITORY_ROOT / 'kumo/diskgraph/diskgraph',
)


def _decorator_name(decorator: ast.expr) -> str:
    if isinstance(decorator, ast.Call):
        decorator = decorator.func
    if isinstance(decorator, ast.Attribute):
        return decorator.attr
    if isinstance(decorator, ast.Name):
        return decorator.id
    return ''


def _assigned_names(path: Path, assignment: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    [value] = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == assignment
            for target in node.targets)
    ]
    return {node.id for node in ast.walk(value) if isinstance(node, ast.Name)}


def _enclosing_function(
    tree: ast.AST,
    target: ast.AST,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node
    current = parent.get(target)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
        current = parent.get(current)
    return None


def _is_direct_broker_resolve(call: ast.Call) -> bool:
    func = call.func
    if not isinstance(func, ast.Attribute) or func.attr not in {
            'resolve', 'resolve_async'
    }:
        return False
    try:
        receiver = ast.unparse(func.value)
    except Exception:
        return False
    return 'connector_credential_broker' in receiver


def _connector_reachable_activity_violations(
    sources: dict[str, str], ) -> tuple[list[str], set[str]]:
    """Return activities transitively reaching a credential-resolution root.

    Imports, same-module functions, and ``self`` methods are resolved exactly.
    An attribute with no statically known receiver is resolved only when its
    function name is unique in the audited source set.
    """
    trees = {
        source_name: ast.parse(source, filename=source_name)
        for source_name, source in sources.items()
    }
    functions = []
    by_module_name = {}
    by_module_class_name = {}
    by_name = {}
    parent = {}
    imports = {}
    resolution_root_keys = set()

    def module_name(source_name):
        return source_name.removesuffix('.py').replace('/', '.')

    for source_name, source in sources.items():
        del source
        tree = trees[source_name]
        module = module_name(source_name)
        module_imports = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parent[child] = node
            if isinstance(node, ast.ImportFrom) and node.module:
                imported_module = node.module
                if node.level:
                    package = module.split('.')[:-node.level]
                    imported_module = '.'.join([*package, imported_module])
                for alias in node.names:
                    module_imports[alias.asname
                                   or alias.name] = (imported_module,
                                                     alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    module_imports[alias.asname
                                   or alias.name.split('.')[0]] = (alias.name,
                                                                   None)
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorators = {_decorator_name(item) for item in node.decorator_list}
            enclosing = parent.get(node)
            classes = []
            while enclosing is not None:
                if isinstance(enclosing, ast.ClassDef):
                    classes.append(enclosing.name)
                enclosing = parent.get(enclosing)
            class_name = '.'.join(reversed(classes))
            key = (source_name, class_name, node.name, node.lineno)
            record = (key, source_name, module, class_name, node, decorators)
            functions.append(record)
            by_module_name.setdefault((module, node.name), []).append(key)
            by_module_class_name.setdefault((module, class_name, node.name),
                                            []).append(key)
            by_name.setdefault(node.name, []).append(key)
            if 'connector_resolution_helper' in decorators:
                resolution_root_keys.add(key)
            if any(
                    isinstance(call, ast.Call)
                    and _is_direct_broker_resolve(call)
                    for call in ast.walk(node)):
                resolution_root_keys.add(key)
        imports[source_name] = module_imports

    def imported_target(source_name, imported_name, attribute=None):
        target = imports[source_name].get(imported_name)
        if target is None:
            return set()
        target_module, symbol = target
        if attribute is None:
            if symbol is None:
                return set()
            return set(by_module_name.get((target_module, symbol), []))
        if symbol is None:
            return set(by_module_name.get((target_module, attribute), []))
        return set(
            by_module_class_name.get((target_module, symbol, attribute), []))

    references = {}
    for key, source_name, module, class_name, node, _decorators in functions:
        targets = set()
        for statement in node.body:
            for child in ast.walk(statement):
                if isinstance(child, ast.Name) and isinstance(
                        child.ctx, ast.Load):
                    imported = imported_target(source_name, child.id)
                    if imported:
                        targets.update(imported)
                    else:
                        targets.update(
                            by_module_name.get((module, child.id), []))
                elif isinstance(child, ast.Attribute):
                    attribute_targets = set()
                    if isinstance(child.value, ast.Name):
                        receiver = child.value.id
                        if receiver in {'self', 'cls'} and class_name:
                            attribute_targets.update(
                                by_module_class_name.get(
                                    (module, class_name, child.attr), []))
                        else:
                            attribute_targets.update(
                                imported_target(source_name, receiver,
                                                child.attr))
                    if (not attribute_targets
                            and len(by_name.get(child.attr, [])) == 1):
                        attribute_targets.update(by_name[child.attr])
                    targets.update(attribute_targets)
        references[key] = targets

    reachable = set(resolution_root_keys)
    changed = True
    while changed:
        changed = False
        for key, _source, _module, _class, _node, _decorators in functions:
            if key in reachable:
                continue
            if references[key].intersection(reachable):
                reachable.add(key)
                changed = True

    violations = []
    for key, source_name, _module, _class, node, decorators in functions:
        is_activity = bool(
            decorators.intersection({
                'defn',
                'process_pool_activity_defn',
                'connector_authorized_activity',
            }))
        if (is_activity and key in reachable
                and 'connector_authorized_activity' not in decorators):
            violations.append(f'{source_name}:{node.lineno}:{node.name}')
    root_names = {key[2] for key in resolution_root_keys}
    return sorted(violations), root_names


def test_every_closed_connector_action_has_one_registered_activity_and_policy():
    registered = connector_activity_registry.registered_connector_activities()

    assert set(registered) == set(ConnectorCapabilityAction)
    assert set(registered) == set(CONNECTOR_ISSUABLE_ACTIONS)
    for action, activity_func in registered.items():
        definition = activity._Definition.must_from_callable(activity_func)
        assert definition.name == action.value


def test_connector_actions_are_registered_on_a_production_worker():
    sources = '\n'.join(
        path.read_text(encoding='utf-8') for path in _WORKER_SOURCES)
    missing = []
    for action, activity_func in (connector_activity_registry.
                                  registered_connector_activities().items()):
        module = inspect.getmodule(activity_func)
        assert module is not None
        if module.__name__ == 'pool.activities':
            # Pool activities are registered by inspected discovery.
            assert '_discover_activities' in sources
            continue
        if activity_func.__name__ not in sources:
            missing.append(action.value)
    assert missing == []


def test_custom_prediction_staging_is_registered_on_data_worker():
    """Keep staging on the queue used by the prediction parent workflow."""
    activity = 'stage_custom_prediction_table'
    assert activity in _assigned_names(_WORKER_SOURCES[0],
                                       'data_backend_activities')
    assert activity not in _assigned_names(_WORKER_SOURCES[1],
                                           'ml_backend_activities')


def test_root_issuance_and_child_delegation_cover_table_version_activity():
    action = ConnectorCapabilityAction.GET_CONNECTOR_TABLE_DATA_VERSION
    assert action in CONNECTOR_TABLE_SOURCE_ACTIONS
    assert action in CONNECTOR_GRAPH_ACTIONS
    assert action in CONNECTOR_REFERENCE_ACTIONS

    for workflow_type in connector_authorization._KNOWN_CONNECTOR_WORKFLOWS:
        child_actions = connector_authorization._actions_for_child_workflow(
            workflow_type)
        assert child_actions <= CONNECTOR_REFERENCE_ACTIONS
        assert not child_actions.intersection(
            CONNECTOR_ACTIVITY_AUTHORIZATION_ACTIONS)
        assert child_actions <= set(
            connector_activity_registry.registered_connector_activities())

    assert action in connector_authorization._actions_for_child_workflow(
        'GraphRefreshWorkflow')
    assert action in connector_authorization._actions_for_child_workflow(
        'TableRefreshWorkflow')


@pytest.mark.parametrize(
    ('workflow_type', 'pool_action'),
    [
        ('DiskgraphIngestWorkflow', ConnectorCapabilityAction.ENSURE_SNAPSHOT),
        ('GenerateTrainingTableWorkflow',
         ConnectorCapabilityAction.POOL_GENERATE_TARGET_TABLE),
        ('GeneratePredictionTableWorkflow',
         ConnectorCapabilityAction.POOL_GENERATE_TARGET_TABLE),
        ('CustomTrainTableWorkflow',
         ConnectorCapabilityAction.BUILD_TTNG_CUSTOM_TRAIN_TABLE_BUNDLE),
        ('MaterializeTrainingTaskWorkflow',
         ConnectorCapabilityAction.POOL_MATERIALIZE_TRAINING_TASK),
        ('MaterializePredictionTaskWorkflow',
         ConnectorCapabilityAction.POOL_MATERIALIZE_PREDICTION_TASK),
    ],
)
def test_pool_child_workflows_receive_their_downstream_connector_action(
    workflow_type,
    pool_action,
):
    actions = connector_authorization._actions_for_child_workflow(workflow_type)
    assert pool_action in actions

    if workflow_type == 'DiskgraphIngestWorkflow':
        # Snapshot ingest only resolves connectors while downloading source
        # files inside ensure_snapshot; do not broaden this child to generic
        # table or graph authority.
        assert actions == {ConnectorCapabilityAction.ENSURE_SNAPSHOT}


def test_job_children_receive_only_their_connector_boundaries() -> None:
    training_actions = connector_authorization._actions_for_child_workflow(
        'TrainingJobWorkflow')
    assert training_actions == {
        ConnectorCapabilityAction.POSTPROCESS_TRAINING_JOB_OUTPUT,
    }

    batch_prediction_actions = (
        connector_authorization._actions_for_child_workflow(
            'BatchPredictionJobWorkflow'))
    assert batch_prediction_actions == set(
        CONNECTOR_DATABRICKS_BATCH_PREDICTION_OUTPUT_ACTIONS)
    assert ConnectorCapabilityAction.START_BATCH_PREDICTION_STATS in (
        batch_prediction_actions)
    assert ConnectorCapabilityAction.DELEGATE_CHILD in batch_prediction_actions


def test_every_direct_production_broker_resolve_is_declared():
    violations = []
    for path in (_REPOSITORY_ROOT / 'rest').rglob('*.py'):
        if path.name in {
                'connector_broker.py', 'connector_activity_registry.py'
        }:
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node,
                              ast.Call) or not _is_direct_broker_resolve(node):
                continue
            function = _enclosing_function(tree, node)
            decorators = ({
                _decorator_name(item)
                for item in function.decorator_list
            } if function is not None else set())
            if not decorators.intersection({
                    'connector_authorized_activity',
                    'connector_resolution_helper',
            }):
                violations.append(
                    f'{path.relative_to(_REPOSITORY_ROOT)}:{node.lineno}')
    assert violations == []


def test_every_transitive_connector_activity_has_a_closed_action():
    sources = {
        str(path.relative_to(_REPOSITORY_ROOT)):
        path.read_text(encoding='utf-8')
        for root in _AUDIT_ROOTS
        for path in root.rglob('*.py')
    }
    violations, roots = _connector_reachable_activity_violations(sources)

    assert {
        'get_connector_from_spec',
        'get_connector_from_spec_async',
        'hydrate_graph_config_for_activity',
        '_resolve_output_connectors',
    } <= roots
    assert violations == []


def test_multi_workflow_connector_inputs_are_reissued_between_boundaries():
    """Inventory request paths that probe, then start with connector input."""
    expected = {
        ('rest/driver/data_snapshot_api_driver.py', 'create_graph_snapshot'),
        ('rest/driver/table_driver.py',
         '_trigger_async_refresh_for_table_entry'),
        ('rest/driver/table_logic.py', '_do_async_refresh'),
    }
    found = set()
    probe_calls = {
        'get_table_data_version_in_activity',
        'refresh_graph_data_version_in_activity',
    }
    start_calls = {'start_refresh_graph', 'start_refresh_table'}
    for path in (_REPOSITORY_ROOT / 'rest').rglob('*.py'):
        source_name = str(path.relative_to(_REPOSITORY_ROOT))
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=source_name)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls = sorted((child.lineno, child.func.attr)
                           for child in ast.walk(node)
                           if isinstance(child, ast.Call)
                           and isinstance(child.func, ast.Attribute))
            probes = [line for line, name in calls if name in probe_calls]
            starts = [line for line, name in calls if name in start_calls]
            if not probes or not starts:
                continue
            key = (source_name, node.name)
            found.add(key)
            reissues = [
                line for line, name in calls
                if name == 'fresh_execution_refs_for_workflow'
            ]
            assert any(probe < reissue < start for probe in probes
                       for reissue in reissues for start in starts), key
    assert found == expected

    # Batched snapshot refresh deliberately separates capability preparation
    # from the all-or-rollback start fanout.  Preserve the same security
    # ordering across that explicit helper boundary.
    snapshot_path = (_REPOSITORY_ROOT / 'rest/driver/'
                     'data_snapshot_api_driver.py')
    snapshot_tree = ast.parse(snapshot_path.read_text(encoding='utf-8'))
    functions = {
        node.name: node
        for node in ast.walk(snapshot_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    def called_attributes(function_name):
        return {
            child.func.attr
            for child in ast.walk(functions[function_name])
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
        }

    prepare_calls = called_attributes(
        '_prepare_state_services_table_refresh_with_connector')
    start_calls = called_attributes(
        '_trigger_many_state_services_table_refresh')
    assert 'get_table_data_version_in_activity' in prepare_calls
    assert 'fresh_execution_refs_for_workflow' in prepare_calls
    assert '_prepare_state_services_table_refresh_with_connector' in start_calls
    assert 'start_refresh_table' in start_calls
    assert '_rollback_table_refresh_batch' in start_calls


def test_transitive_audit_rejects_an_unmapped_indirect_activity():
    violations, _ = _connector_reachable_activity_violations({
        'artificial.py':
        '''
@connector_resolution_helper
def resolve_credentials():
    pass

def intermediate_helper():
    resolve_credentials()

@activity.defn
def silently_unmapped_activity():
    intermediate_helper()
'''
    })

    assert violations == [
        'artificial.py:10:silently_unmapped_activity',
    ]
