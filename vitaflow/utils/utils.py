# Copyright 2018 The vitaFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Miscellaneous Utility functions.
"""

def dict_lookup(dict_, keys, default=None):
    """Looks up :attr:`keys` in the dict, returns the corresponding values.

    The :attr:`default` is used for keys not present in the dict.

    Args:
        dict_ (dict): A dictionary for lookup.
        keys: A numpy array or a (possibly nested) list of keys.
        default (optional): Value to be returned when a key is not in
            :attr:`dict_`. Error is raised if :attr:`default` is not given and
            key is not in the dict.

    Returns:
        A numpy array of values with the same structure as :attr:`keys`.

    Raises:
        TypeError: If key is not in :attr:`dict_` and :attr:`default` is `None`.
    """
    return np.vectorize(lambda x: dict_.get(x, default))(keys)