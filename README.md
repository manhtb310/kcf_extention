# KCF extention: parallel implementations of SCT Tracking on GPU (Jetson TX2i)

# Reference:
[1] https://github.com/jongwon20000/SCT
[2] https://github.com/CTU-IIG/kcf


### Command line

```shellsession
$ git submodule update --init
$ mkdir build
$ cd build
$ cmake [options] ..  # see the tables below
$ make
```

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





