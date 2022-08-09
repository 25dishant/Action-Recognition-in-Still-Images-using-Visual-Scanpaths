from os import name
from threading import main_thread
import mxnet as mx
from mxnet import ndarray as nd

def Positional_Encoding(input, d_model):
    input = input.reshape((-1,1))
    # print("input: ",input)
    dim = mx.nd.arange(d_model//2)
    # print("dim: ",dim)
    sin = mx.nd.sin(input / 10000 ** (2 * dim / d_model))
    # print("sin: ",sin)
    cos = mx.nd.cos(input / 10000 ** (2 * dim / d_model))
    # print("cos: ",cos)

    out = nd.zeros((input.shape[0],d_model))
    # print("out: ",out)
    out[:, ::2] = sin
    # print("out: ",out)
    out[:, 1::2] = cos
    # print("out: ",out)

    return out


if __name__ == "__main__":
    input = nd.array([1,2,3,4,5])


    # print(input)
    ans = Positional_Encoding(input,10)
    print(ans)