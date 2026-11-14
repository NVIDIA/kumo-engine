# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Use only the suite's existing private Studio space and JupyterLab app."""
import json
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from botocore.exceptions import ClientError
from config import STUDIO_IMAGE


class Studio:
    def __init__(self, client, domain_id):
        if not re.fullmatch(r'd-[a-z0-9]+', domain_id):
            raise ValueError('Expected the dedicated Studio domain ID')
        self.sm, self.config = client.sm, client.config
        self.domain_id = domain_id
        self.created = False
        self.s3 = client.s3
        self.locked = False
        self.lease_key = (self.config['root'] + '/studio-leases/' + domain_id
                          + '/' + self.config['space'] + '.json')
        self.invocation = uuid4().hex
        self.target = dict(DomainId=domain_id, SpaceName=self.config['space'],
                           AppType='JupyterLab', AppName='default')

    def arn(self, kind, name):
        return (f'arn:aws:sagemaker:{self.config["region"]}:'
                f'{self.config["account"]}:{kind}/{name}')

    def owned(self, arn, *, invocation=None):
        tags = {
            t['Key']: t['Value']
            for t in self.sm.list_tags(ResourceArn=arn)['Tags']
        }
        if tags.get('KumoTestSuite') != 'kumo-engine-sagemaker-studio':
            raise ValueError('Studio resource is not owned by this test suite')
        if invocation is not None and tags.get(
                'KumoTestInvocation') != invocation:
            raise ValueError('Studio app does not belong to this invocation')

    def app(self, *, require_owner=True):
        try:
            result = self.sm.describe_app(**self.target)
        except ClientError as error:
            if error.response['Error']['Code'] == 'ResourceNotFound':
                return None
            raise
        expected = self.arn(
            'app',
            self.domain_id + '/' + self.config['space'] + '/JupyterLab/default')
        if (result.get('AppArn') != expected
                or result.get('DomainId') != self.domain_id
                or result.get('SpaceName') != self.config['space']
                or result.get('AppType') != 'JupyterLab'
                or result.get('AppName') != 'default'):
            raise ValueError('Studio returned another app')
        if result['Status'] != 'Deleted':
            if require_owner:
                self.owned(result['AppArn'])
            instance = result.get('ResourceSpec', {}).get('InstanceType')
            if instance not in (None, self.config['instance']):
                raise ValueError('Studio app uses another instance type')
        return result

    def preflight(self):
        cfg = self.config
        domain = self.sm.describe_domain(DomainId=self.domain_id)
        domain_arn = self.arn('domain', self.domain_id)
        if (domain.get('DomainArn') != domain_arn
                or domain.get('DomainName') != cfg['domain_name']
                or domain.get('Status') != 'InService'
                or domain.get('AuthMode') != 'IAM'):
            raise ValueError('Studio domain is not the configured test domain')
        self.owned(domain_arn)
        profile = self.sm.describe_user_profile(
            DomainId=self.domain_id, UserProfileName=cfg['user_profile'])
        profile_arn = self.arn('user-profile',
                               self.domain_id + '/' + cfg['user_profile'])
        if (profile.get('UserProfileArn') != profile_arn
                or profile.get('Status') != 'InService'
                or profile.get('UserSettings', {}).get('ExecutionRole')
                != cfg['execution_role']):
            raise ValueError(
                'Studio profile does not use the test execution role')
        self.owned(profile_arn)
        space = self.sm.describe_space(DomainId=self.domain_id,
                                       SpaceName=cfg['space'])
        space_arn = self.arn('space', self.domain_id + '/' + cfg['space'])
        settings = space.get('SpaceSettings', {})
        if (space.get('SpaceArn') != space_arn
                or space.get('Status') != 'InService' or space.get(
                    'OwnershipSettings', {}).get('OwnerUserProfileName')
                != cfg['user_profile'] or space.get(
                    'SpaceSharingSettings', {}).get('SharingType') != 'Private'
                or settings.get('AppType') != 'JupyterLab'
                or settings.get('JupyterLabAppSettings', {}).get(
                    'DefaultResourceSpec',
                    {}).get('InstanceType') != cfg['instance']
                or settings.get('SpaceStorageSettings', {}).get(
                    'EbsStorageSettings',
                    {}).get('EbsVolumeSizeInGb') != cfg['volume_gib']):
            raise ValueError(
                'Studio space is not the dedicated private CPU test space')
        self.owned(space_arn)
        url = urlsplit(space.get('Url', ''))
        if (url.scheme != 'https' or url.username or url.password or url.port
                or not re.fullmatch(r'[a-z0-9-]+\.studio\.us-west-2\.sagemaker\.aws', url.hostname or '')
                or url.path != '/jupyterlab/default' or url.query or url.fragment):
            raise ValueError('Studio space did not return its canonical regional JupyterLab URL')
        self.space_url = space['Url']
        app = self.app()
        if app and app['Status'] not in ('InService', 'Pending', 'Deleted'):
            raise ValueError('Studio test app is not ready for safe reuse: ' +
                             app['Status'])
        return dict(domain_id=self.domain_id, profile=cfg['user_profile'],
                    space=cfg['space'], instance=cfg['instance'],
                    volume_gib=cfg['volume_gib'],
                    app_status=app['Status'] if app else 'absent')

    def lease(self):
        try:
            response = self.s3.get_object(Bucket=self.config['bucket'], Key=self.lease_key)
        except ClientError as error:
            code = error.response['Error']['Code']
            if code in ('403', 'AccessDenied'):
                listed = self.s3.list_objects_v2(Bucket=self.config['bucket'],
                                                 Prefix=self.lease_key, MaxKeys=1)
                if any(item['Key'] == self.lease_key for item in listed.get('Contents', [])):
                    raise
            elif code not in ('NoSuchKey', '404'):
                raise
            return None, None
        body = response['Body']
        try:
            return json.loads(body.read()), response['ETag']
        finally:
            body.close()

    def acquire(self):
        current, etag = self.lease()
        if current and current.get('expires', float('inf')) >= time.time():
            raise ValueError('Dedicated Studio test space is in use by another invocation')
        value = dict(invocation=self.invocation, expires=time.time() + 3600)
        try:
            self.s3.put_object(Bucket=self.config['bucket'], Key=self.lease_key,
                               Body=json.dumps(value).encode(),
                               **({'IfMatch': etag} if current else {'IfNoneMatch': '*'}))
        except ClientError as error:
            if error.response['Error']['Code'] in ('PreconditionFailed', '412'):
                raise ValueError('Dedicated Studio test space was acquired concurrently') from None
            raise
        self.locked = True

    def require_lease(self):
        value, etag = self.lease()
        if (not value or value.get('invocation') != self.invocation
                or value.get('expires', 0) < time.time()):
            raise ValueError('Studio lease ownership changed; no app will be stopped')
        return etag

    def start(self, timeout=1200):
        if timeout > 1200:
            raise ValueError('Studio startup must fit the exclusive lease deadline')
        self.preflight()
        self.acquire()
        current = self.app()
        if not current or current['Status'] == 'Deleted':
            response = self.sm.create_app(
                **self.target,
                ResourceSpec=dict(InstanceType=self.config['instance'],
                                  SageMakerImageArn=STUDIO_IMAGE), Tags=[
                    dict(Key='KumoTestSuite',
                         Value='kumo-engine-sagemaker-studio'),
                    dict(Key='KumoTestInvocation', Value=self.invocation)
                ])
            self.created = True
            if response['AppArn'] != self.arn(
                    'app', self.domain_id + '/' + self.config['space'] +
                    '/JupyterLab/default'):
                raise ValueError(
                    'Created Studio app has an unexpected identity')
        deadline = time.monotonic() + timeout
        while True:
            app = self.app(require_owner=not self.created)
            if app and app['Status'] == 'InService':
                self.owned(app['AppArn'],
                           invocation=self.invocation if self.created else None)
                if app.get('ResourceSpec',
                           {}).get('InstanceType') != self.config['instance']:
                    raise ValueError(
                        'Ready Studio app does not use the test instance type')
                if app.get('ResourceSpec', {}).get('SageMakerImageArn') != STUDIO_IMAGE:
                    raise ValueError('Studio app does not use the configured CPU image')
                return dict(app_arn=app['AppArn'], created=self.created,
                            status='InService', resource_spec=app['ResourceSpec'])
            if app and app['Status'] not in ('Pending', ):
                raise ValueError('Studio test app startup failed: ' +
                                 app['Status'])
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    'Studio test app did not start within its deadline')
            time.sleep(10)

    def url(self):
        return self.sm.create_presigned_domain_url(
            DomainId=self.domain_id,
            UserProfileName=self.config['user_profile'],
            SpaceName=self.config['space'], LandingUri='app:JupyterLab:/',
            ExpiresInSeconds=300,
            SessionExpirationDurationInSeconds=1800)['AuthorizedUrl']

    def close(self, timeout=300):
        if not self.locked:
            return dict(status='no lease acquired', created=self.created)
        self.require_lease()
        app = self.app()
        if app and app['Status'] != 'Deleted':
            self.require_lease()
            self.sm.delete_app(**self.target)
        deadline = time.monotonic() + timeout
        while True:
            self.require_lease()
            app = self.app()
            if not app or app['Status'] == 'Deleted':
                etag = self.require_lease()
                self.s3.delete_object(Bucket=self.config['bucket'], Key=self.lease_key,
                                      IfMatch=etag)
                self.locked = False
                return dict(status='deleted', created=self.created)
            if time.monotonic() >= deadline:
                raise TimeoutError('Studio test app did not finish stopping')
            time.sleep(10)


def prepare_studio_input(root, executed):
    """Add the non-code edit fixture when replaying an older producer notebook."""
    from evidence import save, save_probe_cell

    original = root / 'executed.ipynb'
    fixture = save_probe_cell()
    for cell in executed['cells']:
        if cell.get('id') == fixture['id']:
            if (cell['cell_type'] != 'markdown'
                    or ''.join(cell['source']) != ''.join(fixture['source'])):
                raise ValueError(
                    'Reserved Studio fixture cell differs from the producer')
            return original
    prepared = dict(executed, cells=[fixture, *executed['cells']])
    target = root / 'studio-input.ipynb'
    save(target, prepared)
    return target


def verify_studio(client, spec, directory):
    from config import DOMAIN_ID
    from evidence import digest, receipt

    root = Path(directory)
    executed = json.loads((root / 'executed.ipynb').read_text())
    submitted = json.loads((root / spec['notebook_name']).read_text())
    record = receipt(executed, submitted, spec)
    upload = prepare_studio_input(root, executed)
    receipt(json.loads(upload.read_text()), submitted, spec)
    studio = Studio(client, DOMAIN_ID)
    report = dict(
        status='failed', producer_notebook_sha256=digest(
            (root / 'executed.ipynb').read_bytes()),
        upload_notebook_sha256=digest(upload.read_bytes()))
    try:
        report['app'] = studio.start()
        request = dict(output=str(root / 'studio'), domain_id=DOMAIN_ID,
                       space_url=studio.space_url, url=studio.url(),
                       invocation=studio.invocation, notebook=str(upload),
                       notebook_sha256=report['upload_notebook_sha256'],
                       receipt=record)
        if spec['scenario'] in ('graph', 'all'):
            request['graph'] = json.loads(
                (root / 'outputs/graph-result.json').read_text())
        result = subprocess.run(
            ['node',
             str(Path(__file__).with_name('studio_browser.cjs'))],
            input=json.dumps(request), text=True, capture_output=True,
            timeout=600)
        report['browser'] = json.loads(
            (root / 'studio/result.json').read_text())
        if result.returncode or report['browser'].get('status') != 'passed':
            raise RuntimeError(
                'Native Studio notebook verification failed; see studio/result.json'
            )
        report['status'] = 'passed'
    finally:
        try:
            report['cleanup'] = studio.close()
        except Exception:
            report['status'] = 'failed'
            report['cleanup'] = dict(status='failed')
            raise
        finally:
            (root /
             'studio-app.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
