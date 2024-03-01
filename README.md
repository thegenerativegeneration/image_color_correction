
# Color correcting images and videos

This repository contains tools to color-correct images and videos. The main tool is `color_correction.py`.

This repository started life as a fork of <https://github.com/dazzafact/image_color_correction>, where the main algorithms come from. Just packaged differently for different use cases.

> A word of warning: the command line interface of the tools leave something to be desired.
>
> Another word of warning: the color mapping does not take the gamma curves of the captured images or videos into account (yet).
> So for images or videos with `gamma != 1` results will not be correct, and the more the luminance has to be changed to worse the results will be.

Start by creating a Python venv and installing the needed packages:

```
python -m venv .venv
. .venv/bin/activate # Mac/Linux
&.venv\scripts\activate.ps1 # Windows powershell
pip install -r requirements.txt
```

## Usage with reference color card

If you have a specific type of reference color card, and you have captured that card as part of your recording: life is easy. (See below for the specific card). You pass a known correct version of the reference color card as `--ref` option. You pass your input image as a positional parameter. The script will find the reference card in the input image and adjust the colors of the _whole input image_ so that the colors of the reference card match the correct reference card.

You can use `--output` to save the resulting image and `--view` to view it. You can also use `--savelut` to save the lookup table for re-use, for example on other images or videos. The LUT is saved in `.cube` format.

## Usage without reference color card

If:

- you haven't captured a reference color card, and
- you have captured a specific type of Aruco marker, and
- you have captures from multiple different cameras, and
- you are primarily interested in ensuring all cameras "look the same"

this can be done too. Basically by selecting the "best looking" of the captures and using that as the reference. A workflow is explained in [sample_usage.md](sample_usage.md).

## Converting more images and videos

After saving the LUT in the previous step you can convert more images, which need not contain a captured reference card (because you have the LUT already).

Use the `--loadlut` option in stead of the `--ref` option.

Converting videos can be done with the `video_map_color.py` script. 

## Original README file

Here is the original readme file from <https://github.com/dazzafact/image_color_correction>, there is a lot of useful information in here but also a lot of things that are no longer true.

## White-Balance with Color Cards 
A python function to correct image White-Balance using Color Cards, detecting with [CV2 Aruco](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html).
Base Idea: https://pyimagesearch.com/2021/02/15/automatic-color-correction-with-opencv-and-python/

You just need an already optimized color input image and another image which is not color optimized. Both images with Color Card, using ArUCo Marker (you can glue them on the Corners of every imageCard to for detecting)

`python color_correction.py --reference ref.jpg  --input test.jpg --out output.jpg`

## Image Inputs

**First you need a color optimized Image as Reference using a Color Card with [Aruco Markers](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html) (or follow the Link to [purchase the Panton Color Card](https://www.pantone.com/eu/de/pantone-color-match-card)).**


![image](https://user-images.githubusercontent.com/67874406/187918735-78967b36-ce77-47cc-8a17-773ea856d988.png)

**1.) A ReferenceImage used as the basis for all further image processes. The colors of this Card image should be optimized to your liking.**

`--reference ref.jpg`

![image](https://user-images.githubusercontent.com/67874406/187906176-23303477-0dd7-4ef8-ae05-1e36f3e82de7.png)


 **2.) you have to choose a none color optimized Image with the a Color card, to detect the Color Difference between both images**

`--input test.jpg`

![image](https://user-images.githubusercontent.com/67874406/187906327-8a42dcf2-c312-4ce7-b336-6f8d4f310788.png)

**3.) As result you get the Final color optimized Output Image based on the reference Histogram Colors**

![image](https://user-images.githubusercontent.com/67874406/187906458-244286b9-70c5-4b6f-8f35-bdee9908573a.png)


## Python command

### **Output File**
`python color_correction.py --reference raw.jpg  --input test.jpg --out output.jpg`


### **Special output File width**
`python color_correction.py --reference raw.jpg  --input test.jpg --out output.jpg  --width 1280`


### **with output Preview**
`python color_correction.py --reference raw.jpg  --input test.jpg --view`

### **with an output Preview and file output**
`python color_correction.py --reference raw.jpg  --input test.jpg --out output.jpg --view`


Blog: https://pyimagesearch.com/2021/02/15/automatic-color-correction-with-opencv-and-python/

Stackoverflow: https://stackoverflow.com/questions/70233645/color-correction-using-opencv-and-color-cards/73566972#73566972
