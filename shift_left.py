# from vulkanese import vulkanese as ve
import vulkanese as ve

# begin GPU test
instance = ve.instance.Instance(verbose=False)

for i in range(len(instance.getDeviceList())):
    device = instance.getDevice(i)

    ve.math.arith.test(device=device)
    ve.examples.simple_graph.test(device=device)
    ve.math.machine_learning.resnet.test(device=device)

instance.release()
