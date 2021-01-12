# KCF extention: Implementation of SCT Tracking on GPU (Jetson TX2i)

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

`./kcf_vot [options] <path/to/region.txt or groundtruth.txt> <path/to/images.txt> [path/to/output.txt]`

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





