"""
Nenwin-project (NEural Networks WIthout Neurons) for
the AI Honors Academy track 2020-2021 at the TU Eindhoven.

Author: Lulof Pirée
May 2021

Copyright (C) 2021 Lulof Pirée

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


Class to generate filenames with a timestamp.
"""
import time
import os


class FilenameGenerator:
    """
    Class to generate filenames with a timestamp.
    """

    def __init__(self, directory: str, base: str, extension: str):
        """
        Arguments:
            * directory: path to directory containing the target file.
            * base: prefix of the name of the file itself.
            * extension: suffix of the file, 
                usually an extension such as ".pt".
        """
        self.__directory = directory
        self.__base = base
        self.__extension = extension

    def gen_filename(self,
                     is_checkpoint: bool = False,
                     timestamp: float=time.time()) -> str:
        """
        Arguments:
            * is_checkpoint: flag whether "_checkpoint" should be
                added to the filename (after the timestamp,
                before the extension).
            * time: epochtime of the timestamp that is part of the
                filename.
        """
        if is_checkpoint:
            checkpoint = "_checkpoint"
        else:
            checkpoint = ""

        name = self.__base + time.asctime(time.localtime(timestamp)) \
            + checkpoint + self.__extension
        name = name.replace(":", "-")
        name = name.replace(" ", "_")
        output = os.path.join(self.__directory, name)
        return output
