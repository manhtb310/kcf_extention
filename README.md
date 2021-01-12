# KCF extention: parallel implementations of SCT Tracking on GPU (Jetson TX2i)

# Reference:
[1]: https://github.com/jongwon20000/SCT
[2]: https://github.com/CTU-IIG/kcf
[3]: https://github.com/Shanigen/kcf

## Compilation

There are multiple ways how to compile the code.


### Using cmake gui

```shellsession
$ git submodule update --init
$ mkdir build
$ cmake-gui .
```

- Use the just created build directory as "Where to build the
  binaries".
- Press "Configure".
- Choose desired build options. Each option has a comment briefly
  explaining what it does.
- Press "Generate" and close the window.

```shellsession
$ make -C build
```
### Command line

```shellsession
$ git submodule update --init
$ mkdir build
$ cd build
$ cmake [options] ..  # see the tables below
$ make
```

The `cmake` options below allow to select, which version to build.

The following table shows how to configure different FFT
implementations.

|Option| Description |
| --- | --- |
| `-DFFT=OpenCV` | Use OpenCV to calculate FFT.|
| `-DFFT=fftw` | Use fftw and its `plan_many` and "New-array execute" functions. If `std::async`, OpenMP or cuFFTW is not used the plans will use 2 threads by default.|
| `-DFFT=cuFFTW` | Use cuFFTW interface to cuFFT library.|
| `-DFFT=cuFFT` | Use cuFFT. This version also uses pure CUDA implementation of `ComplexMat` class and Gaussian correlation.|

With all of these FFT version additional options can be added:

|Option| Description |
| --- | --- |
| `-DBIG_BATCH=ON` | Concatenate matrices of different scales to one big matrix and perform all computations on this matrix. This improves performance of GPU FFT offloading. |
| `-DOPENMP=ON` | Parallelize certain operation with OpenMP. With `-DBIG_BATCH=OFF` it runs computations for differenct scales in parallel, with `-DBIG_BATCH=ON` it parallelizes the feature extraction, which runs on the CPU. With `fftw`, Ffftw's plans will execute in parallel.|
| `-DCUDA_DEBUG=ON` | Adds calls cudaDeviceSynchronize after every CUDA function and kernel call.|
| `-DOpenCV_DIR=/opt/opencv-3.3/share/OpenCV` | Compile against a custom OpenCV version. |
| `-DASYNC=ON` | Use C++ `std::async` to run computations for different scales in parallel. This mode of parallelization was present in the original implementation. Here, it is superseeded with -DOPENMP. This doesn't work with `BIG_BATCH` mode.|

See also the top-level `Makefile` for other useful cmake parameters
such as extra compiler flags etc.

## Running

No matter which method is used to compile the code, the result will be
a `kcf_vot` binary.

It operates on an image sequence created according to [VOT 2014
methodology][10]. Alternatively, you can use a video file or a camera
as an input. You can find some image sequences in [vot2016
datatset][11].

The binary can be run as follows:

1. `./kcf_vot [options]`

   The program looks for `groundtruth.txt` or `region.txt` and
   `images.txt` files in current directory.

   - `images.txt` contains a list of images to process, each on a
     separate line.
   - `groundtruth.txt` contains the correct location of the tracked
     object in each image as four corner points listed clockwise
     starting from bottom left corner. Only the first line from this
     file is used.
   - `region.txt` is an alternative way of specifying the location of
     the object to track via its bounding box (top_left_x, top_left_y,
     width, height) in the first frame.

2. `./kcf_vot [options] <directory>`

   Looks for `groundtruth.txt` or `region.txt` and `images.txt` files
   in the given `directory`.

3. `./kcf_vot [options] <path/to/region.txt or groundtruth.txt> <path/to/images.txt> [path/to/output.txt]`

4. `./kcf_vot [options] <file>`

   Reads the images from video `<file>`.

5. `./kcf_vot [options] <number>`

   Captures the images from camera `<number>`.

By default the program generates file `output.txt` containing the
bounding boxes of the tracked object in the format "top_left_x,
top_left_y, width, height".

[10]: http://www.votchallenge.net/
[11]: http://www.votchallenge.net/vot2016/dataset.html

### Options

| Options | Description |
| ------- | ----------- |
| --fit, -f[W[xH]] | Specifies the dimension to which the extracted patches should be scaled. Best performance is achieved for powers of two; the smaller number the higher performance but worse accuracy. No dimension or zero rounds the dimensions to the nearest smaller power of 2, a single dimension `W` will result in patch size of `W`×`W`. The numbers should be divisible by 4. |
| --visualize, -v[delay_ms] | Visualize the output, optionally with specified delay. If the delay is 0 the program will wait for a key press. |
| --output, -o <output.txt>	 | Specify name of output file with rectangle coordinates. |
| --video_out, -O <output.avi>	 | Specify name of output video file. |
| --debug, -d				 | Generate debug output. |
| --visual_debug, -p[p\|r] | Show graphical window with debugging information (either **p**atch or filter **r**esponse). |
| --box, -b[X,Y,W,H] | Specify initial bounding box via command line rather than via `region.txt` or `groundtruth.txt` or by selecting it with mouse (if no coordinates are given). |
| --box_out, -B <box.txt> | Specify the file name where to store manually specified bounding boxes (with the <kbd>i</kbd> key) |





