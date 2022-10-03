import os
import sys
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))

localtest = True
if localtest == True:
    vkpath = os.path.join(here, "..", "..", "vulkanese")
    sys.path = [vkpath] + sys.path
    from vulkanese import *
else:
    from vulkanese import *

# get vulkanese instance
instance_inst = Instance(verbose=True)
print(json.dumps(instance_inst.getDeviceList(), indent=2))
instance_inst.release()
#
