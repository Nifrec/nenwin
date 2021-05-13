"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
March 2021

Copyright (C) 2021 Lulof Pirée, 

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Testcases for FilenameGenerator.
"""

import unittest

from nenwin.backprop.filename_gen import FilenameGenerator


class FilenameGeneratorTestCase(unittest.TestCase):

    def setUp(self) -> None:
        base = "base_"
        directory = r"/dir"
        extension = ".meh"
        self.timestamp = 0

        self.name_gen = FilenameGenerator(directory, base, extension)

    def test_gen_name_checkpoint(self):

        expected = r"/dir/base_Thu Jan  1 01-00-00 1970_checkpoint.meh"
        result = self.name_gen.gen_filename(True, self.timestamp)

        self.assertEqual(expected, result)

    def test_gen_name_no_checkpoint(self):

        expected = r"/dir/base_Thu Jan  1 01-00-00 1970.meh"
        result = self.name_gen.gen_filename(False, self.timestamp)

        self.assertEqual(expected, result)

if __name__ == "__main__":
    unittest.main()
