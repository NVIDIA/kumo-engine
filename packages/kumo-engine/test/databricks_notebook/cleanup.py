# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remove only resources admitted as absent for this notebook attempt."""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlencode


class AttemptResources:
    def __init__(self, client, config, spec, root, *, lifecycle=False,
                 resume=False):
        self.client, self.config, self.spec = client, config, spec
        self.root = Path(root)
        attempt = spec['attempt']
        if not re.fullmatch('[a-f0-9]{12}', attempt):
            raise ValueError('Invalid cleanup attempt')
        kind = 'lifecycle' if lifecycle else 'graph'
        folder = ('/Shared/kumo_test_catalogue/KUMO_TEST/lifecycle'
                  if lifecycle else config['workspace_dir'])
        if spec['notebook_path'] != folder + '/kumo-test-' + kind + '-' + attempt:
            raise ValueError(
                'Cleanup requires the exact generated notebook path')
        output = config['volume'] + ('/lifecycle'
                                     if lifecycle else '') + '/runs/' + attempt
        if spec['output'] != output + '/outputs':
            raise ValueError('Cleanup requires the exact attempt output prefix')
        self.prefixes = [output]
        if lifecycle:
            self.prefixes.append(config['volume'].removesuffix('/notebook-ui') +
                                 '/lifecycle/' + attempt)
        self.experiment = (spec['experiment'] + '-' +
                           attempt) if lifecycle else None
        if self.experiment and self.experiment != folder + '/KUMO_TEST_MLOPS-' + attempt:
            raise ValueError('Cleanup requires the exact attempt experiment')
        self.model = ('kumo_test_catalogue.kumo_test.KUMO_TEST_MLOPS-' +
                      attempt if lifecycle else None)
        scope = dict(notebook=spec['notebook_path'], prefixes=self.prefixes,
                     experiment=self.experiment, model=self.model,
                     user=config.get('expected_user'))
        scope_hash = hashlib.sha256(json.dumps(
            scope, sort_keys=True).encode()).hexdigest()
        if resume:
            self.ledger = json.loads((self.root / 'resources.json').read_text())
            if self.ledger.get('scope_sha256') != scope_hash or self.ledger.get(
                    'attempt') != attempt:
                raise ValueError('Recovery ledger scope differs')
        else:
            self.ledger = dict(attempt=attempt, notebook_created=False,
                               scope_sha256=scope_hash, admitted=False,
                               started_ms=int(time.time() * 1000))
            self.save()

    def save(self):
        temporary = self.root / 'resources.json.tmp'
        temporary.write_text(json.dumps(self.ledger, indent=2) + '\n')
        temporary.replace(self.root / 'resources.json')

    def missing(self, path):
        try:
            self.client.api(path)
        except RuntimeError as error:
            if str(error) == 'Databricks API HTTP 404':
                return True
            raise
        return False

    def __enter__(self):
        for prefix in self.prefixes:
            if not self.missing('/api/2.0/fs/directories' +
                                quote(prefix, safe='/')):
                raise ValueError(
                    'Attempt storage already exists; no cleanup ownership acquired'
                )
        if self.model and not self.missing('/api/2.1/unity-catalog/models/' +
                                           quote(self.model, safe='')):
            raise ValueError(
                'Attempt model already exists; no cleanup ownership acquired')
        self.ledger['admitted'] = True
        self.save()
        return self

    def prepare_experiment(self):
        if not self.experiment:
            return
        endpoint = '/api/2.0/mlflow/experiments/get-by-name?' + urlencode(
            dict(experiment_name=self.experiment))
        if not self.missing(endpoint):
            raise ValueError(
                'Attempt experiment already exists; refusing ownership')
        self.ledger['experiment_pending'] = True
        self.save()
        response = self.client.api(
            '/api/2.0/mlflow/experiments/create', 'POST',
            dict(
                name=self.experiment, tags=[
                    dict(key='kumo.test.attempt', value=self.spec['attempt'])
                ]))
        self.ledger['experiment_id'] = str(response['experiment_id'])
        self.ledger['experiment_pending'] = False
        self.save()

    def submitting(self):
        self.ledger['submission_pending'] = True
        self.save()

    def submitted(self):
        self.ledger['submission_pending'] = False
        self.save()

    def importing(self):
        self.ledger['import_pending'] = True
        self.save()

    def notebook_created(self):
        self.ledger['import_pending'] = False
        self.ledger['notebook_created'] = True
        self.save()

    def clear_directory(self, prefix, budget):
        if budget[0] <= 0:
            raise ValueError('Cleanup exceeded the file budget')
        budget[0] -= 1
        endpoint = '/api/2.0/fs/directories' + quote(prefix, safe='/')
        entries, tokens, token = [], set(), None
        while True:
            try:
                page = self.client.api(endpoint + '?' + urlencode(
                    dict(page_size=1000,
                         **({
                             'page_token': token
                         } if token else {}))))
            except RuntimeError as error:
                if str(error) == 'Databricks API HTTP 404':
                    return
                raise
            entries.extend(page.get('contents', []))
            if len(entries) > budget[0]:
                raise ValueError('Cleanup exceeded the file budget')
            token = page.get('next_page_token')
            if not token:
                break
            if token in tokens:
                raise ValueError('Cleanup pagination repeated')
            tokens.add(token)
        owned = []
        for entry in entries:
            directory = entry.get('is_directory', False)
            path = entry['path'].removesuffix('/') if directory else entry['path']
            if ('..' in PurePosixPath(path).parts or '//' in path
                    or path.rsplit('/', 1)[0] != prefix):
                raise ValueError(
                    'Directory listing escaped the admitted attempt prefix')
            owned.append((path, directory))
        for path, directory in owned:
            if directory:
                self.clear_directory(path, budget)
            else:
                self.client.api('/api/2.0/fs/files' + quote(path, safe='/'),
                                'DELETE')
                budget[0] -= 1
        self.client.api(endpoint, 'DELETE')

    def model_identity(self, model):
        if (not isinstance(model.get('id'), str) or not model['id']
                or str(model.get('full_name', '')).casefold() != self.model.casefold()
                or model.get('owner') != self.config['expected_user']
                or type(model.get('created_at')) is not int
                or model['created_at'] < self.ledger['started_ms'] - 2000):
            raise ValueError('Model ownership differs from the admitted attempt')
        return model['id'], model['created_at']

    def model_versions(self, endpoint, model):
        versions, tokens, token = {}, set(), None
        for _ in range(10):
            page = self.client.api(endpoint + '/versions?' + urlencode(
                dict(max_results=3, **({'page_token': token} if token else {}))))
            for version in page.get('model_versions', []):
                number = str(version.get('version'))
                parent = '.'.join(str(version.get(key, '')) for key in
                                  ('catalog_name', 'schema_name', 'model_name'))
                if (number not in ('1', '2') or number in versions
                        or parent.casefold() != model['full_name'].casefold()
                        or version.get('created_by') != self.config['expected_user']
                        or type(version.get('created_at')) is not int
                        or version['created_at'] < model['created_at']
                        or not isinstance(version.get('id'), str) or not version['id']
                        or version.get('status') not in ('READY', 'FAILED_REGISTRATION')):
                    raise ValueError('Model version differs from the owned test attempt')
                versions[number] = (version['id'], version['created_at'],
                                    version['created_by'], parent.casefold())
            token = page.get('next_page_token')
            if token is not None and not isinstance(token, str):
                raise ValueError('Model version pagination is ambiguous')
            if not token:
                return versions
            if token in tokens:
                raise ValueError('Model version pagination is ambiguous')
            tokens.add(token)
        raise ValueError('Model version pagination exceeded the cleanup budget')

    def remove_model(self, model):
        identity = self.model_identity(model)
        endpoint = '/api/2.1/unity-catalog/models/' + quote(model['full_name'], safe='')
        remaining = self.model_versions(endpoint, model)

        def unchanged():
            if self.model_identity(self.client.api(endpoint)) != identity:
                raise ValueError('Model identity changed during cleanup')
            if self.model_versions(endpoint, model) != remaining:
                raise ValueError('Model versions changed during cleanup')

        for number in sorted(remaining):
            unchanged()
            self.client.api(endpoint + '/versions/' + number, 'DELETE')
            del remaining[number]
            unchanged()
        unchanged()
        self.client.api(endpoint, 'DELETE')
        if not self.missing(endpoint):
            raise ValueError('Model deletion was not confirmed')

    def __exit__(self, *failure):
        if not self.ledger.get('admitted'):
            return
        errors = []
        result_path = self.root / 'cleanup.json'
        result_path.write_text(
            json.dumps(dict(status='failed', failed_resources=['job_state'])) +
            '\n')
        submitted = self.root / 'submitted-run.json'
        if self.ledger.get('submission_pending') and not submitted.is_file():
            if failure and failure[0] is not None:
                return False
            raise RuntimeError(
                'Submission result unknown; resources preserved for inspection')
        if submitted.is_file():
            run_id = json.loads(submitted.read_text())['run_id']
            deadline = time.monotonic() + 180
            while True:
                run = self.client.api('/api/2.2/jobs/runs/get?run_id=' +
                                      str(run_id))
                if run['state']['life_cycle_state'] in ('TERMINATED', 'SKIPPED',
                                                        'INTERNAL_ERROR'):
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        'Cleanup cannot delete resources while the notebook job is running'
                    )
                time.sleep(5)
        if self.ledger.get('import_pending'):
            errors.append('notebook_import_unknown')
        if self.ledger.get('experiment_pending'):
            errors.append('experiment')
        if self.ledger.get('experiment_id'):
            try:
                experiment = self.client.api(
                    '/api/2.0/mlflow/experiments/get?' +
                    urlencode(dict(experiment_id=self.ledger['experiment_id']))
                )['experiment']
                tags = {
                    tag['key']: tag['value']
                    for tag in experiment.get('tags', [])
                }
                if (str(experiment.get('experiment_id'))
                        != self.ledger['experiment_id']
                        or experiment.get('name') != self.experiment
                        or experiment.get('creation_time', 0)
                        < self.ledger['started_ms'] - 2000 or
                        tags.get('kumo.test.attempt') != self.spec['attempt']):
                    raise ValueError(
                        'Experiment ownership differs from the admitted attempt'
                    )
                self.client.api(
                    '/api/2.0/mlflow/experiments/delete', 'POST',
                    dict(experiment_id=self.ledger['experiment_id']))
            except Exception:
                errors.append('experiment')
        if self.model:
            try:
                endpoint = '/api/2.1/unity-catalog/models/' + quote(
                    self.model, safe='')
                if not self.missing(endpoint):
                    model = self.client.api(endpoint)
                    self.remove_model(model)
            except Exception:
                errors.append('model')
        if self.ledger['notebook_created']:
            try:
                self.client.api(
                    '/api/2.0/workspace/delete', 'POST',
                    dict(path=self.spec['notebook_path'], recursive=False))
            except Exception:
                errors.append('notebook')
        for prefix in self.prefixes:
            try:
                self.clear_directory(prefix, [10000])
            except Exception:
                errors.append('storage')
        result = dict(status='failed' if errors else 'passed',
                      failed_resources=errors)
        (self.root /
         'cleanup.json').write_text(json.dumps(result, indent=2) + '\n')
        if errors and not (failure and failure[0] is not None):
            raise RuntimeError(
                'Per-attempt cleanup failed; inspect cleanup.json')


def recover(root, client, config):
    root = Path(root)
    if not (root / 'resources.json').is_file():
        return
    result = root / 'cleanup.json'
    if result.is_file() and json.loads(
            result.read_text()).get('status') == 'passed':
        return
    spec = json.loads((root / 'scenario.json').read_text())
    lifecycle = '/kumo-test-lifecycle-' in spec['notebook_path']
    resources = AttemptResources(client, config, spec, root,
                                 lifecycle=lifecycle, resume=True)
    if not resources.ledger.get('admitted'):
        return
    submitted = root / 'submitted-run.json'
    if submitted.is_file():
        run_id = json.loads(submitted.read_text())['run_id']
        run = client.api('/api/2.2/jobs/runs/get?run_id=' + str(run_id))
        if (run.get('run_id') != run_id or not config.get('expected_user')
                or run.get('creator_user_name') != config['expected_user']
                or run.get('run_name')
                != 'kumo-test-notebook-ui-' + spec['attempt']
                or len(run.get('tasks', [])) != 1
                or run['tasks'][0].get('notebook_task', {}).get('notebook_path')
                != spec['notebook_path']):
            raise ValueError('Recovery cannot prove submitted run ownership')
        if run['state']['life_cycle_state'] not in ('TERMINATED', 'SKIPPED',
                                                    'INTERNAL_ERROR'):
            client.api('/api/2.2/jobs/runs/cancel', 'POST', dict(run_id=run_id))
    resources.__exit__()


def main():
    from client import Client, configuration

    parser = argparse.ArgumentParser(
        description='Recover an owned notebook attempt')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    config = configuration()
    recover(args.output_dir, Client(config), config)


if __name__ == '__main__':
    main()
