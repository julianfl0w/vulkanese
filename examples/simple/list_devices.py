import os
import sys
import numpy as np
import pkg_resources

here = os.path.dirname(os.path.abspath(__file__))

if "vulkanese" not in [pkg.key for pkg in pkg_resources.working_set]:
    sys.path = [os.path.join(here, "..", "..", "..", "vulkanese", "vulkanese")] + sys.path

from vulkanese import *

# get vulkanese instance
instance_inst = Instance(verbose=False)
print(json.dumps(instance_inst.getDeviceList(), indent=2))
instance_inst.release()
#
