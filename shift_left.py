# from vulkanese import vulkanese as ve
import vulkanese as ve

# begin GPU test
instance = ve.instance.Instance(verbose=False)
device = instance.getDevice(0)

ve.math.arith.test(device=device)
ve.math.machine_learning.resnet.test(device=device)

instance.release()
