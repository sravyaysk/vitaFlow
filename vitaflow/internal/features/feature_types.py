# Copyright 2018 The vitFlow Authors. All Rights Reserved.
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

"""

class SingleFeature():
    
    def __init__(self):
        self.FEATURE_NAME = None
        self.LABEL_NAME = None

class TwoFeatures:
    
    def __init__(self):
        self.NUM_FEATURES = 2
        self.FEATURE_TYPE = None
        self.FEATURE_1_NAME = None
        self.FEATURE_2_NAME = None

        self.OUT_TAGS = "classes"
        self.OUT_CONFIDENCE = "confidence"

    def __eq__(self, other):
        """Override the default Equals behavior"""
        return self.FEATURE_TYPE == other.FEATURE_NAME and self.NUM_FEATURES == other.NUM_FEATURES


class ImageFeature(SingleFeature):
    
    def __init__(self):
        SingleFeature.__init__(self)
        self.FEATURE_NAME = "image"
        self.LABEL_NAME = "label"

class GANFeature(TwoFeatures):
    
    def __init__(self):
        super(GANFeature, self).__init__()
        self.FEATURE_TYPE = "image+noise"
        self.FEATURE_1_NAME = "image"
        self.FEATURE_2_NAME = "noise"

class ITextFeature(TwoFeatures):
    
    def __init__(self):
        super(ITextFeature, self).__init__()
        self.FEATURE_TYPE = "text+char_ids"
        self.FEATURE_1_NAME = "text"
        self.FEATURE_2_NAME = "char_ids"


class IPostionalFeature():
    
    def __init__(self):
        self.FEATURE_NAME = "text+char_ids+positional_info"
        self.NUM_FEATURES = 3
        self.FEATURE_1_NAME = "text"
        self.FEATURE_2_NAME = "char_ids"
        self.FEATURE_3_NAME = "position"
