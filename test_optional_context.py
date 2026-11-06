import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import data_retrieval as d


class OptionalContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.airfare = self.root / 'airfare_cpi_monthly.csv'
        self.airfare.write_text('old-primary-data\n')
        self.fuel = self.root / 'jet_fuel_weekly.csv'
        self.fuel.write_text('old-context-data\n')

    def run_cli(self, extra, fuel_failure=True, airfare_failure=False):
        def fuel(*args):
            log = args[-1]
            log.append(dict(d.log_entry('eia_jet_fuel', 'https://www.eia.gov/test'), error='test outage'))
            if fuel_failure:
                raise OSError('test outage')
            return [{'date': '2026-11-01', 'usd_per_gallon': 2.5}]
        with patch.object(sys, 'argv', ['data_retrieval.py', '--output-dir', str(self.root)] + extra), \
             patch('data_retrieval.retrieve_fuel', side_effect=fuel), \
             patch('data_retrieval.retrieve_airfare', side_effect=ValueError('invalid primary') if airfare_failure else None,
                   return_value=[{'date': '2026-09-01', 'airfare_cpi_index': '300'}]):
            return d.main()

    def test_optional_fuel_outage_preserves_context_and_updates_primary(self):
        self.assertEqual(self.run_cli(['--sources', 'airfare', 'eia']), 0)
        self.assertIn('300', self.airfare.read_text())
        self.assertEqual(self.fuel.read_text(), 'old-context-data\n')
        self.assertEqual(pd.read_csv(self.root / 'retrieval_log.csv').iloc[0]['status'], 'optional_context_failed')

    def test_strict_or_eia_only_failure_is_blocking(self):
        for options in (['--sources', 'airfare', 'eia', '--require-eia'], ['--sources', 'eia']):
            self.assertEqual(self.run_cli(options), 1)
            self.assertEqual(self.airfare.read_text(), 'old-primary-data\n')
            self.assertEqual(self.fuel.read_text(), 'old-context-data\n')

    def test_required_signal_failure_still_preserves_existing_data(self):
        self.assertEqual(self.run_cli(['--sources', 'airfare', 'eia'], airfare_failure=True), 1)
        self.assertEqual(self.airfare.read_text(), 'old-primary-data\n')
        self.assertEqual(self.fuel.read_text(), 'old-context-data\n')

    def test_successful_fuel_and_primary_are_both_updated(self):
        self.assertEqual(self.run_cli(['--sources', 'airfare', 'eia'], fuel_failure=False), 0)
        self.assertIn('300', self.airfare.read_text())
        self.assertIn('2.5', self.fuel.read_text())

    def test_default_intake_uses_only_the_three_primary_sources(self):
        with patch.object(sys, 'argv', ['data_retrieval.py', '--output-dir', str(self.root)]), \
             patch('data_retrieval.extract_dataset', return_value=([{'quarter': '2026Q3'}], [{'metric': 'revenue'}])) as delta, \
             patch('data_retrieval.retrieve_tsa', return_value=[{'date': '2026-11-01', 'passengers': '100'}]) as tsa, \
             patch('data_retrieval.retrieve_airfare', return_value=[{'date': '2026-09-01', 'airfare_cpi_index': '300'}]) as airfare, \
             patch('data_retrieval.retrieve_fuel') as fuel:
            self.assertEqual(d.main(), 0)
        delta.assert_called_once()
        tsa.assert_called_once()
        airfare.assert_called_once()
        fuel.assert_not_called()


if __name__ == '__main__':
    unittest.main()
