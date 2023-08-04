# The code is adapted from Programming Assignment 2 of CSC 292


import time
import math
from collections import defaultdict
import heapq
import io
import numpy as np
import sys
import argparse
from scipy.fftpack import dct, idct
from PIL import Image

np.set_printoptions(threshold=sys.maxsize)

timers = defaultdict(int)

izigzagmat = np.array(
        [[0,   1,  5,  6, 14, 15, 27, 28],
         [2,   4,  7, 13, 16, 26, 29, 42],
         [3,   8, 12, 17, 25, 30, 41, 43],
         [9,  11, 18, 24, 31, 40, 44, 53],
         [10, 19, 23, 32, 39, 45, 52, 54],
         [20, 22, 33, 38, 46, 51, 55, 60],
         [21, 34, 37, 47, 50, 56, 59, 61],
         [35, 36, 48, 49, 57, 58, 62, 63]])

zigzagarr = np.array([0,  1,  8, 16,  9,  2,  3, 10, 17, 24, 32, 25, 18, 11,  4,  5, 12,
                     19, 26, 33, 40, 48, 41, 34, 27, 20, 13,  6,  7, 14, 21, 28, 35, 42,
                     49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52,
                     45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63])


base_mat = np.array(
    [[16,  11,  10,  16,  24,  40,  51,  61],
       [12,  12,  14,  19,  26,  58,  60,  55],
       [14,  13,  16,  24,  40,  57,  69,  56],
       [14,  17,  22,  29,  51,  87,  80,  62],
       [18,  22,  37,  56,  68, 109, 103,  77],
       [24,  35,  55,  64,  81, 104, 113,  92],
       [49,  64,  78,  87, 103, 121, 120, 101],
       [72,  92,  95,  98, 112, 100, 103,  99]], dtype = np.uint64)


color_spaces = ["1", "CMYK", "F", "HSV", "I", "L", "LAB", "P", "RGB", "RGBA", "RGBX", "YCbCr"]

def parseArg():
    parser = argparse.ArgumentParser()
    parser.add_argument('im', type = str, help='Input image path')
    parser.add_argument('-m', type = str, choices = color_spaces, required = False,
                        default = 'RGB', help = 'Set color space, default as RGB')
    parser.add_argument('-d', action = 'store_true', help = 'Display the output image')
    parser.add_argument('-huff', action = 'store_true', help = 'Use Huffman Coding when compressing (pretty slow)')
    parser.add_argument('-c', type = int, choices = range(1, 101), required = False,
                        default = 0, help = 'Compress the input image at given quality, default as 75')
    parser.add_argument('-o', type = str, required = False, help = 'Output input or compressed image to the path')
    return parser.parse_args()

# Input 1 to 100 and return a quantization matrix
# Adapted from stackoverflow.com/questions/29215879/how-can-i-generalize-the-quantization-matrix-in-jpeg-compression
def generate_q_matrix(Q):
    if Q < 50:
        S = 5000/Q
    else:
        S = 200 - 2*Q
    Ts = ((S * base_mat + 50) // 100)
    # A magical number 7 to make sure the overflow will not be that serious
    return np.clip(Ts, 7, 255).astype(np.uint8)


def dft8x8(block):
    return np.fft.fft2(block)


def idft8x8(block):
    return np.fft.ifft2(block).real


def dct8x8(block):
    start_time = time.time()
    re = dct(dct(block.T, norm='ortho').T, norm='ortho')
    timers['dct8x8'] += time.time() - start_time
    return re


def idct8x8(block):
    start_time = time.time()
    re = idct(idct(block.T, norm='ortho').T, norm='ortho')
    timers['idct8x8'] += time.time() - start_time
    return re


def quantize8x8(block, qmat):
    start_time = time.time()
    re = np.round(block / qmat)
    timers['quantize8x8'] += time.time() - start_time
    return re


def iquantize8x8(block, qmat):
    start_time = time.time()
    re = block * qmat
    timers['iquantize8x8'] += time.time() - start_time
    return re


# transform a 8x8 matrix to linear 64 zig-zag array
def zigzag8x8(block):
    re = block.take(zigzagarr)
    return re


# transform a linear 64 zig-zag array to 8x8 matrix
def izigzag8x8(arr):
    re = np.take(arr, izigzagmat)
    return re


def encode8x8(block):
    start_time = time.time()
    zig = zigzag8x8(block)
    # run-length coding
    mask = np.packbits(np.where(zig == 0, 0, 1))
    re = bytearray(mask) + bytearray(zig[zig != 0])
    timers['encode8x8'] += time.time() - start_time
    return re


def decode8x8(fp):
    start_time = time.time()
    arr = np.zeros((64,), dtype = np.int8)
    mask = np.unpackbits(np.frombuffer(fp.read(8), dtype = np.uint8))
    inds = np.arange(0, 64)[mask == 1]
    for ind in inds:
        arr[ind] = ord(fp.read(1))
    re = izigzag8x8(arr)
    timers['decode8x8'] += time.time() - start_time
    return re


class ImageCompression:
    def __init__(self):
        print("Read image")
        self.args = parseArg()
        self.use_huff = self.args.huff
        self.huffman = None
        self.compressed = None
        self.latest_img = None
        self.mode = self.args.m
        self.is_bqi = self.args.im.endswith('.bqi')
        self.qmat = None
        if self.args.c:
            self.qmat = generate_q_matrix(self.args.c)
        self.read_img()

    def compress(self):
        if not self.args.c:
            print("Skip compressing. (Use \"-c\" to set quality of compression)")
            return
        print("Compressing")
        encoded_arr = bytearray()
        center = np.full((8, 8), 128)

        layer_shape = self.latest_img.shape[:2]
        layer_cnt = self.latest_img.shape[2]
        wblock = layer_shape[0] // 8
        hblock = layer_shape[1] // 8
        workload = layer_cnt * wblock * hblock
        workcur = 0
        print(f'\r[{(workcur*100)/workload:0.1f}%] {workcur}/{workload}', end = '')
        for i in range(layer_cnt):
            layer = self.latest_img[:, :, i]
            for x in range(0, layer.shape[0] - 7, 8):
                for y in range(0, layer.shape[1] - 7, 8):
                    workcur += 1
                    block = (layer[x: x + 8, y: y + 8] - center)
                    dct = dct8x8(block)
                    qtz = quantize8x8(dct, self.qmat)
                    qtz = np.clip(qtz, -128, 127).astype(np.uint8)
                    encoded = encode8x8(qtz)
                    encoded_arr += encoded
                print(f'\r[{(workcur*100)/workload:0.1f}%] {workcur}/{workload}', end = '')
        print()
        if self.use_huff:
            self.huffman = Huff.Encoder(np.frombuffer(encoded_arr, dtype = np.uint8))
            self.compressed = self.huffman.encode()
        else:
            self.compressed = encoded_arr
        return self.compressed

    def decompress(self, fp, shape_out, qmat):
        print("Decompressing")

        layers = []  # for different layers like RGB has 3 layers
        center = np.full((8, 8), 128)
        wblock = shape_out[0] // 8
        hblock = shape_out[1] // 8
        layer_cnt = shape_out[2]
        workload = layer_cnt * wblock * hblock
        workcur = 0
        print(f'\r[{(workcur*100)/workload:0.1f}%] {workcur}/{workload}', end = '')
        for i in range(layer_cnt):
            layer = np.ndarray(shape = shape_out[:2])
            for x in range(0, shape_out[0] - 7, 8):
                for y in range(0, shape_out[1] - 7, 8):
                    workcur += 1
                    decoded = decode8x8(fp)
                    iqtz = iquantize8x8(decoded, qmat)
                    idct = idct8x8(iqtz)
                    layer[x: x + 8, y: y + 8] = np.clip(idct + center, 0, 255)
                print(f'\r[{(workcur*100)/workload:0.1f}%] {workcur}/{workload}', end = '')
            layers.append(layer)
        print()

        self.latest_img = np.stack(layers, axis = 2)
        return self.latest_img

    def show_img(self):
        if self.args.d:
            print(f'Showing {self.mode} Image of shape: {self.latest_img.shape}')
            Image.fromarray(self.latest(), self.mode).show()

    # Return the Displayable and Convertable shape of latest_img
    def latest(self):
        if self.latest_img.shape[2] == 1:
            return self.latest_img.reshape(self.latest_img.shape[:2]).astype(np.uint8)
        else:
            return self.latest_img.astype(np.uint8)

    def read_img(self):
        fp = open(self.args.im, 'rb')
        if not self.is_bqi:
            with Image.open(self.args.im) as im:
                self.latest_img = np.asarray(im.convert(self.mode))
                print('Input Image Shape: ', self.latest_img.shape)
                print('Input Image Mode: ', im.mode)
        else:
            input_mode = ''
            c = ord(fp.read(1))
            while c != 0:
                input_mode += chr(c)
                c = ord(fp.read(1))
            shape_out = np.frombuffer(fp.read(12), dtype = np.uint32)
            qmat = np.frombuffer(fp.read(64), dtype = np.uint8).reshape((8, 8))
            if self.use_huff:
                self.huffman = Huff.Decoder(fp)
                bit_count = int(np.frombuffer(fp.read(4), dtype = np.uint32)[0])
            print(f'Input Image Shape: {shape_out}')
            print(f'Input Image Mode: {input_mode}')
            print(f'Input Image Quantization Matrix: {qmat}')
            byte_stream = fp
            if self.use_huff:
                byte_stream = io.BytesIO(self.huffman.decode(fp, bit_count))
            re = self.decompress(byte_stream, shape_out, qmat)
            if self.mode != input_mode:
                re = np.asarray(Image.fromarray(self.latest(), input_mode).convert(self.mode))
            self.latest_img = re

        fp.close()
        if len(self.latest_img.shape) == 2:
            self.latest_img = self.latest_img.reshape((*self.latest_img.shape, 1))
        print(f'Processed Image Shape: {self.latest_img.shape}')
        return self.latest_img

    def save_img(self):
        if not self.args.o:
            print("No output specified. (Use \"-o\" to specify output image name)")
            return
        print("Saving compressed image")
        start_time = time.time()
        if not (self.args.o.endswith('.bqi')):
            print(self.latest_img.shape)
            Image.fromarray(self.latest_img.astype(np.uint8), self.mode).save(self.args.o)
            return
        fp = open(self.args.o, mode = 'wb')
        # Save compressed image to fp here
        fp.write(bytearray(self.mode + '\0', encoding = 'ASCII'))
        tmp_shape = self.latest_img.shape
        if len(tmp_shape) == 2:
            tmp_shape = (*tmp_shape, 1)
        fp.write(bytearray(np.asarray(tmp_shape, dtype = np.uint32)))
        fp.write(bytearray(self.qmat))
        if self.use_huff:
            self.huffman.save_tree(fp)
        fp.write(self.compressed)
        fp.flush()
        timers['save_img'] += time.time() - start_time


class Huff:
    class Node:
        def __init__(self, data = None, left = None, right = None):
            self.data = data
            self.left = left
            self.right = right

        def dump(self, tabs = 0):
            tab = tabs * ' '
            print(tab, self)
            if self.left:
                self.left.dump(tabs + 2)
            if self.right:
                self.right.dump(tabs + 2)

        def save(self, fp):
            hasdata = self.data is not None
            hasleft = self.left is not None
            hasright = self.right is not None
            fp.write(int.to_bytes(1 * hasdata + 2 * hasleft + 4 * hasright, 1, 'big', signed = False))
            if hasdata:
                fp.write(int.to_bytes(self.data, 1, 'big', signed = False))
            if hasleft:
                self.left.save(fp)
            if hasright:
                self.right.save(fp)

        def load(self, fp):
            flags = int.from_bytes(fp.read(1), 'big', signed = False)
            hasdata = flags & 1
            hasleft = flags & 2
            hasright = flags & 4
            if hasdata:
                self.data = fp.read(1)
            if hasleft:
                self.left = Huff.Node()
                self.left.load(fp)
            if hasright:
                self.right = Huff.Node()
                self.right.load(fp)

        def __repr__(self):
            return f'||{self.data} {self.left is not None} {self.right is not None}||'

    class Encoder:
        def __init__(self, arr: np.ndarray):
            self.dict = [None] * 256
            self.code = bytearray()
            self.hist_dic = {}
            self.arr = arr.flatten()

            hist, bins = np.histogram(self.arr, 256, [0, 256])
            bins = bins.astype(np.uint8)
            minHeap = sorted([(f, Huff.Node(int(c))) for c, f in zip(bins, hist)],
                             key = lambda x: x[0], reverse = True)
            while len(minHeap) > 1:
                freq_node_pair = (minHeap[-1][0] + minHeap[-2][0], Huff.Node())
                freq_node_pair[1].left = minHeap[-2][1]
                freq_node_pair[1].right = minHeap[-1][1]
                #minHeap = minHeap[:-2]
                #minHeap = sorted(minHeap[:-2] + [freq_node_pair], key = lambda x: x[0], reverse = True)
                minHeap = list(heapq.merge(minHeap[:-2], [freq_node_pair], key = lambda x: x[0], reverse = True))
            self.root = minHeap[0][1]
            self.toHuffmanCode(self.root)
            print(f'Before: {hist.sum() * 8} Now: {sum(f*len(self.dict[c]) for c, f in zip(bins, hist))}')

        def toHuffmanCode(self, tree, s = ""):
            if tree.data is not None:
                self.dict[int(tree.data)] = np.array(list(s), dtype = np.uint8)
                return

            self.toHuffmanCode(tree.left, s + '0')
            self.toHuffmanCode(tree.right, s + '1')

        def encode(self):
            start_time = time.time()
            print('Huffman Encoding')
            bit_count = 0
            workcur = 0
            workload = len(self.arr)
            bit_arr = np.ndarray((len(self.arr) * 8, ), dtype = np.uint8)
            for val in self.arr:
                workcur += 1
                print(f'\r[{(workcur*100)/workload:0.1f}%] {workcur}/{workload}', end = '')
                cur = len(self.dict[val])
                bit_arr[bit_count: bit_count + cur] = self.dict[val]
                bit_count += cur
            print()
            encoded_bytes = bytearray(np.packbits(bit_arr[:bit_count]))
            timers['HuffEncode'] += time.time() - start_time
            return bytearray(np.asarray([bit_count], dtype = np.uint32)) + encoded_bytes

        def save_tree(self, fp):
            self.root.save(fp)

    class Decoder:
        def __init__(self, fp):
            self.root = Huff.Node()
            self.root.load(fp)

        def decode(self, fp, size):
            start_time = time.time()
            print("Huffman Decoding")
            decoded_bytes = bytearray()
            buf = np.unpackbits(np.frombuffer(fp.read(1 + (size - size % 8) // 8), dtype = np.uint8))
            fp.flush()
            temp = self.root
            workcur = 0
            workload = size
            for bit in buf[:size]:
                workcur += 1
                print(f'\r[{(workcur*100)/workload:0.1f}%] {workcur}/{workload}', end = '')
                if bit == 0:
                    temp = temp.left
                else:
                    temp = temp.right

                if temp.data is not None:
                    decoded_bytes += temp.data
                    temp = self.root
            print()
            timers['HuffDecode'] += time.time() - start_time
            return decoded_bytes


def main():
    print(parseArg().__dict__)
    imcmp = ImageCompression()
    imcmp.compress()
    imcmp.show_img()
    imcmp.save_img()
    print('\n\nFunction Timers:', sorted(timers.items(), key = lambda x: (x[1], x[0]), reverse = True), end = '\n\n')


if __name__ == "__main__":
    main()
