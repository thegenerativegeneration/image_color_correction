## Example workflow for Athlone captures

First create some working directories. I used `recordings` for the original video files, `images` for the files I would use for calibration, `luts` for the lookup tables and the converted images, `output` for the output videos after conversion.

```
mkdir recording images luts output
cp ...../*.mkv recording/
```

Next we have to select the reference image for every video. I used `vlc` on the mac, and its `Save Snapshot` command. Name each image in a reasonable name.

```
open recording/disnuc101_0_0119-1137.mkv
mv  ~/Pictures/vlcsnap-2024-02-29-12h39m12s218.png images/disnuc101.png
open recording/disnuc103_0_0119-1137.mkv
mv ~/Pictures/vlcsnap-2024-02-29-12h43m18s959.png images/disnuc103.png
open recording/disnuc202_0_0119-1137.mkv
mv ~/Pictures/vlcsnap-2024-02-29-12h44m07s448.png images/disnuc202.png
open recording/disnuc301_0_0119-1137.mkv
mv ~/Pictures/vlcsnap-2024-02-29-12h44m57s905.png images/disnuc301.png
open recording/iti-nuc-01_0_0119-1137.mkv
mv ~/Pictures/vlcsnap-2024-02-29-12h46m19s148.png images/iti-nuc-01.png
open recording/iti-nuc-02_0_0119-1137.mkv
mv ~/Pictures/vlcsnap-2024-02-29-12h47m24s422.png images/iti-nuc-02.png
open recording/iti-nuc-03_0_0119-1137.mkv
mv ~/Pictures/vlcsnap-2024-02-29-12h48m03s172.png images/iti-nuc-03.png
open recording/iti-nuc-04_0_0119-1137.mkv
mv ~/Pictures/vlcsnap-2024-02-29-12h48m38s671.png images/iti-nuc-04.png
```

Now inspect all the images for the one with the best color. I used the `Finder` with its `spacebar` preview.
```
open images
```

In my case `disnuc101` looks best. We call this the refereence. Create the LUTs by locating the same area in each image, and comparing the color curves of each of these to the color curve of the reference.

```
python ../color_correction.py --reference images/disnuc101.png --output luts --savelut .cube --raw_aruco images/*.png
```

For each of the images a number is printed, the higher this number the more the LUT will change the colors. Obviously for the image that is used as the reference this number should be `0.0`.

We now want to inspect the colormapped images. I used the `Finder` again.

```
open luts
```

If you don't like the results: remove everything in `luts`, go back two steps, select a different reference, try again.

Now convert each video. Note that this will lose the Depth and Infrared channels, if used like this.

```
to be provided
```
