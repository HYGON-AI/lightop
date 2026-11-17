# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only coverage of the repository skill's offline and evidence-cache contract."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / '.agents/skills/lightop-kernel-agent-loop/assets/lightop-kernel-knowledge/scripts'
DIFF = b'diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-a=1\n+a=2\n'
BASE = 'a' * 40
HEAD = 'b' * 40


@unittest.skipUnless(SCRIPTS.is_dir(), 'Repository skill is absent from this wheel-only checkout')
@unittest.skipUnless(importlib.util.find_spec('yaml') is not None, 'Requires the optional PyYAML skill dependency')
class KernelSkillToolsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp = Path(temporary.name)
        old_path = list(sys.path)
        sys.path.insert(0, str(SCRIPTS))
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), old_path))
        spec = importlib.util.spec_from_file_location('lightop_skill_fetch_test', SCRIPTS / 'fetch_evidence.py')
        self.consumer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.consumer)
        root = self.tmp / 'corpus'
        (root / 'sources').mkdir(parents=True)
        for name, value in [('knowledge_root', lambda: root), ('cache_root', lambda: self.tmp / 'cache')]:
            patcher = mock.patch.object(self.consumer, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        from _kb import Page
        self.page = Page(root / 'sources/p.md', 'sources/p.md', {
            'id': 'pr-example-1', 'url': 'https://github.com/example/project/pull/1',
            'head_ref_oid': HEAD, 'artifact_dir': 'evidence/pull-bundles/example/gh-1',
        }, '')

    def network_forbidden(self, *args):
        self.fail('Offline lookup contacted the network')

    def test_cold_lookup_is_offline(self):
        with mock.patch.object(self.consumer, 'request', self.network_forbidden):
            with self.assertRaisesRegex(FileNotFoundError, '--fetch'):
                self.consumer.load_or_fetch(self.page)

    def test_fetch_then_offline_reuse_and_corruption(self):
        calls = []

        def request(url, accept):
            calls.append((url, accept))
            if accept == 'application/vnd.github.diff':
                return DIFF
            return json.dumps({'base': {'sha': BASE}, 'head': {'sha': HEAD}}).encode()

        with mock.patch.object(self.consumer, 'request', request):
            path, meta = self.consumer.load_or_fetch(self.page, fetch=True)
        self.assertEqual(path.read_bytes(), DIFF)
        self.assertEqual(meta['diff_sha256'], hashlib.sha256(DIFF).hexdigest())
        self.assertEqual(meta['revision']['head_sha'], HEAD)
        self.assertIs(meta['revision']['index_head_matches'], True)
        self.assertEqual(len(calls), 3)
        with mock.patch.object(self.consumer, 'request', self.network_forbidden):
            self.assertEqual(self.consumer.load_or_fetch(self.page, fetch=True)[0], path)
        path.write_bytes(DIFF + b'corrupted')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.consumer.load_or_fetch(self.page)

    def test_upstream_revision_mismatch_is_not_cached(self):
        with mock.patch.object(self.consumer, 'request', return_value=json.dumps({'base': {'sha': BASE}, 'head': {'sha': 'c' * 40}}).encode()):
            with self.assertRaisesRegex(ValueError, 'differs from the index'):
                self.consumer.load_or_fetch(self.page, fetch=True)
        self.assertFalse(list(self.consumer.cache_root().rglob('review.diff')))

    def test_changed_pr_is_not_cached(self):
        responses = [json.dumps({'base': {'sha': BASE}, 'head': {'sha': HEAD}}).encode(), DIFF,
                     json.dumps({'base': {'sha': 'c' * 40}, 'head': {'sha': HEAD}}).encode()]
        with mock.patch.object(self.consumer, 'request', side_effect=responses):
            with self.assertRaisesRegex(ValueError, 'changed during collection'):
                self.consumer.load_or_fetch(self.page, fetch=True)
        self.assertFalse(list(self.consumer.cache_root().rglob('review.diff')))

    def test_unrecorded_head_is_reported_unknown(self):
        del self.page.meta['head_ref_oid']
        def request(_, accept):
            return DIFF if accept == 'application/vnd.github.diff' else json.dumps({'base': {'sha': BASE}, 'head': {'sha': HEAD}}).encode()
        with mock.patch.object(self.consumer, 'request', request):
            _, meta = self.consumer.load_or_fetch(self.page, fetch=True)
        self.assertIsNone(meta['revision']['index_head_matches'])
        self.assertIs(meta['revision']['historical_base_verified'], False)

    def test_full_corpus_and_path_boundary(self):
        directory = self.consumer.knowledge_root() / self.page.meta['artifact_dir']
        directory.mkdir(parents=True)
        (directory / 'review.diff').write_bytes(DIFF)
        with mock.patch.object(self.consumer, 'request', self.network_forbidden):
            path, meta = self.consumer.load_or_fetch(self.page)
        self.assertEqual(path.read_bytes(), DIFF)
        self.assertEqual(meta['location'], 'full-corpus')
        self.page.meta['artifact_dir'] = '../outside'
        with self.assertRaisesRegex(ValueError, 'escapes'):
            self.consumer.load_or_fetch(self.page)

    def test_non_github_source_requires_authorized_external_fetch(self):
        self.page.meta['url'] = 'https://example.org/team/project/-/merge_requests/1'
        with mock.patch.object(self.consumer, 'request', self.network_forbidden):
            with self.assertRaisesRegex(ValueError, 'authorized tool'):
                self.consumer.load_or_fetch(self.page, fetch=True)

    def test_bundled_index_query_and_page(self):
        env = dict(os.environ, XDG_CACHE_HOME=str(self.tmp), PYTHONDONTWRITEBYTECODE='1')
        env.pop('KERNEL_KNOWLEDGE_ROOT', None)
        result = subprocess.run([sys.executable, str(SCRIPTS / 'query.py'), 'lightop', '--compact', '--limit', '3'], env=env, capture_output=True, text=True, timeout=40)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.strip())
        self.assertFalse((self.tmp / 'lightop-kernel-agent').exists())
        page = subprocess.run([sys.executable, str(SCRIPTS / 'get_page.py'), 'ref-hygon-lightop', '--frontmatter-only'], env=env, capture_output=True, text=True, timeout=40)
        self.assertEqual(page.returncode, 0, page.stderr)
        self.assertIn('ref-hygon-lightop', page.stdout)

    def test_hcu_lookup_preserves_hygon_results(self):
        env = dict(os.environ, XDG_CACHE_HOME=str(self.tmp), PYTHONDONTWRITEBYTECODE='1')
        env.pop('KERNEL_KNOWLEDGE_ROOT', None)
        outputs = []
        for name in ('hcu', 'hygon'):
            result = subprocess.run([sys.executable, str(SCRIPTS / 'query.py'), name,
                                     '--architecture', name, '--paths-only', '--limit', '4000'],
                                    env=env, capture_output=True, text=True, timeout=40)
            self.assertEqual(result.returncode, 0, result.stderr)
            outputs.append(result.stdout)
        self.assertTrue(outputs[0].strip())
        self.assertEqual(outputs[0], outputs[1])
        page = subprocess.run([sys.executable, str(SCRIPTS / 'get_page.py'),
                               'ref-hygon-lightop', '--frontmatter-only'],
                              env=env, capture_output=True, text=True, timeout=40)
        self.assertEqual(page.returncode, 0, page.stderr)
        self.assertIn('HCU LightOp operator library', page.stdout)

    def test_page_id_and_relative_path_match(self):
        env = dict(os.environ, XDG_CACHE_HOME=str(self.tmp), PYTHONDONTWRITEBYTECODE='1')
        env.pop('KERNEL_KNOWLEDGE_ROOT', None)
        command = [sys.executable, str(SCRIPTS / 'get_page.py')]
        by_id = subprocess.run(command + ['ref-hygon-lightop'], env=env, capture_output=True, text=True, timeout=20)
        self.assertEqual(by_id.returncode, 0, by_id.stderr)
        paths = subprocess.run([sys.executable, str(SCRIPTS / 'query.py'), '--repo', 'HYGON-AI/lightop', '--paths-only', '--limit', '1'], env=env, capture_output=True, text=True, timeout=20)
        self.assertEqual(paths.returncode, 0, paths.stderr)
        from _kb import id_map
        by_path = subprocess.run(command + [id_map()['ref-hygon-lightop'].relpath], env=env, capture_output=True, text=True, timeout=20)
        self.assertEqual(by_path.returncode, 0, by_path.stderr)
        self.assertEqual(by_id.stdout, by_path.stdout)

    def test_invalid_index_is_rejected(self):
        from _kb import load_index
        path = self.tmp / 'assets/index.jsonl'
        path.parent.mkdir()
        path.write_text(json.dumps({'kind': 'catalog', 'schema_version': 1, 'page_count': 1}) + '\n' +
                        json.dumps({'kind': 'page', 'path': '../outside', 'meta': {}, 'body': ''}) + '\n')
        with self.assertRaisesRegex(ValueError, 'Invalid or duplicate index path'):
            load_index(path)

    def test_legacy_full_corpus_override(self):
        root = self.tmp / 'legacy'
        path = root / 'sources/prs/example.md'
        path.parent.mkdir(parents=True)
        text = '---\nid: legacy-example\ntitle: Legacy full corpus\n---\nOriginal page body.\n'
        path.write_text(text)
        env = dict(os.environ, KERNEL_KNOWLEDGE_ROOT=str(root), PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run([sys.executable, str(SCRIPTS / 'get_page.py'), 'legacy-example'], env=env, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, text)

    def test_profiler_digest_on_cpu(self):
        log = self.tmp / 'hipprof.log'
        log.write_text('hipLaunchKernel 100 us\nmy_test_kernel 20 us\n')
        output = self.tmp / 'digest.md'
        script = ROOT / '.agents/skills/lightop-kernel-agent-loop/assets/hcu-profiler-report/scripts/hcu_profile_digest.py'
        run = subprocess.run([sys.executable, str(script), '--hipprof', str(log), '--output', str(output)], capture_output=True, text=True, timeout=20)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertTrue(output.is_file())
        self.assertIn('my_test_kernel', output.read_text())
        self.assertIn('HCU Profile Digest', output.read_text())


if __name__ == '__main__':
    unittest.main()
