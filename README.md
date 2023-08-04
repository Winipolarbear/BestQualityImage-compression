# BestQualityImage-compression
a python NumPy based image compression method that enables usage of various color spaces

This document outlines the detailed specifications and usage of a novel image compression methodology, bqi (Best Quality Image), inspired by the JPEG compression technique. Remarkably, this innovative project is capable of accepting monochromatic images as input, compressing them using a diverse selection of color spaces, each delivering a marginally unique output effect. We have incorporated the principles of clean Python object-oriented design into the development of this project. We introduce a highly versatile encoding and decoding algorithm for the bqi image format. 

Here are some of the primary features of our new method:

It leverages the Discrete Cosine Transform (DCT) and sparse array, making it a JPEG-inspired compression technique.
By default, it uses the RGB color space, although it has been designed to accommodate virtually all standard color spaces.
It is compatible with widely accepted image formats such as jpg and png.
It possesses a compression ratio that is approximately equivalent to that of the png format.
It offers the flexibility to set the quality of compression anywhere between 1 to 100.
It provides the option to use Huffman Coding to enhance encoding/decoding speed or space efficiency, although this is not generally recommended due to its slower operation and incompatibility with the standard mode.
Command Guide:
We have also provided a command guide for the convenience of users:

'im InputImagePath': A mandatory parameter specifying the path of the input image.
'-m': Enables the specification of the color mode, for example, RGB or YCbCr.
'-huff': Activates Huffman Coding. This is not recommended due to its slow operation and incompatibility with the standard mode.
'-c Quality': Defines the compression quality from 1 to 100. Absence of this parameter will bypass the compression process.
'-d': Displays the input image. Automatically decompresses if the file has a .bqi extension.
'-o OutputImagePath': Sets the path for the output image. If the path ends with .bqi, the system will automatically compress the image.

