#!/usr/bin/env python
"""Reader Module

Author: Hunter Mellema
Summary: Provides a reader object for processing measurements from a file

"""
import pickle as pck
from filtering.measurements import *
from filtering.stations import *

class ReaderR3(object):
    """ Reader for processing Range and Range Rate measurments from a text file
    """
    def __init__(self, stns):
        self.stns = {stn.stn_id:stn for stn in stns}
        self.msr_list = []

    def process(self, txt_file, km=True):
        """
        """
        with open(txt_file) as f:
            for line in f:
                line = line.strip()
                time, stn_id, ran, range_rate = line.split("    ")
                msr = [float(ran), float(range_rate)]
                if km:
                    msr = [i / 1000 for i in msr]

                self.msr_list.append(R3Msr(float(time), msr, self.stns[stn_id],
                                           self.stns[stn_id].cov))
