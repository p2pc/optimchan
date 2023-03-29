from unittest import TestCase

from utils import seed_everything, seed_random

class TestSeed(TestCase):
    def test_case_00(self):
        seed_everything(2023)

    def test_case_01(self):
        seed_random(2024)
