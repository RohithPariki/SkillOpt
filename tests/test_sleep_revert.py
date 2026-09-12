"""Tests for reverting adopted skills."""
import hashlib
import json
import os
import shutil
import tempfile
import unittest

from skillopt_sleep.staging import (
    SkillProposal,
    adopt,
    adopt_skills,
    revert,
    revert_skills,
    write_staging,
    StagingError,
)
from skillopt_sleep.types import SleepReport


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

class TestRevert(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.project = os.path.join(self.tmp, "project")
        self.staging = os.path.join(self.tmp, "staging")
        os.makedirs(self.project)
        os.makedirs(self.staging)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_revert_skills(self):
        skill_live = os.path.join(self.project, "skills", "hello", "SKILL.md")
        _write(skill_live, "before")
        before_sha = _sha("before")

        proposal = SkillProposal(
            skill_name="hello",
            proposed_skill="after",
            live_skill_path=skill_live,
        )
        report = SleepReport(night=1, project=self.project)
        self.staging = write_staging(
            self.tmp,
            report=report,
            proposed_skill=None, proposed_memory=None,
            live_skill_path=None, live_memory_path=None,
            skill_proposals=[proposal],
            report_md="",
        )

        receipts = adopt_skills(self.staging)
        self.assertEqual(len(receipts), 1)
        
        with open(skill_live, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "after")
            
        reverted = revert_skills(self.staging)
        self.assertEqual(len(reverted), 1)
        
        with open(skill_live, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "before")
            
    def test_revert_legacy(self):
        skill_live = os.path.join(self.project, "SKILL.md")
        _write(skill_live, "legacy before")
        before_sha = _sha("legacy before")

        report = SleepReport(night=1, project=self.project, accepted=True)
        self.staging = write_staging(
            self.tmp,
            report=report,
            proposed_skill="legacy after",
            proposed_memory=None,
            live_skill_path=skill_live,
            live_memory_path=None,
            skill_proposals=[],
            report_md="",
        )

        updated = adopt(self.staging)
        self.assertEqual(len(updated), 1)
        
        with open(skill_live, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "legacy after")
            
        reverted = revert(self.staging)
        self.assertEqual(len(reverted), 1)
        
        with open(skill_live, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "legacy before")
