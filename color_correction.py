"""
Image White Balancing using CV2 and COlor Correction Cards with ArUCo Markers
Author: https://pyimagesearch.com/2021/02/15/automatic-color-correction-with-opencv-and-python/

Modify Stephan Krol 
G-Mail: Stephan.Krol.83[at]
Website: https://CouchBoss.de

"""


from typing import Optional, cast, List
from imutils.perspective import four_point_transform
import numpy as np
from numpy.typing import NDArray
import argparse
import imutils
import cv2
import cv2.typing
import sys
from os.path import exists
import os.path as pathfile
from PIL import Image
import matplotlib.pyplot as plt

OurImageType = NDArray[np.uint8]
OurImageType1Channel = NDArray[np.uint8]
OurColormap = List[NDArray[np.int32]]
OurColormap1Channel = NDArray[np.int32]

class ColorCorrector:

    def __init__(self, *, debug : bool = False):
        self.debug = debug
        self.colormap : Optional[OurColormap] = None

    def find_color_card(self, image : OurImageType) -> Optional[OurImageType]:
        """Find a color card in an image. Return another image, with only the color card, with perspective fixed"""
        # load the ArUCo dictionary, grab the ArUCo parameters, and
        # detect the markers in the input image
        arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        arucoParams = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
        (corners, ids, _) = detector.detectMarkers(image)
        if self.debug:
            outputImage = image.copy()
            cv2.aruco.drawDetectedMarkers(outputImage, corners, ids)
            winTitle = f"Markers."
            cv2.imshow(winTitle, outputImage)
            while True:
                ch = cv2.waitKey()
                if ch == 27:
                    break
                print(f"ignoring key {ch}")
            cv2.destroyWindow(winTitle)
        # try to extract the coordinates of the color correction card
        try:
            # otherwise, we've found the four ArUco markers, so we can
            # continue by flattening the ArUco IDs list
            ids = ids.flatten()

            # extract the top-left marker
            i = np.squeeze(np.where(ids == 923))
            topLeft = np.squeeze(corners[i])[0]

            # extract the top-right marker
            i = np.squeeze(np.where(ids == 1001))
            topRight = np.squeeze(corners[i])[1]

            # extract the bottom-right marker
            i = np.squeeze(np.where(ids == 241))
            bottomRight = np.squeeze(corners[i])[2]

            # extract the bottom-left marker
            i = np.squeeze(np.where(ids == 1007))
            bottomLeft = np.squeeze(corners[i])[3]

        # we could not find color correction card, so gracefully return
        except:
            return None

        # build our list of reference points and apply a perspective
        # transform to obtain a top-down, bird’s-eye view of the color
        # matching card
        cardCoords = np.array([topLeft, topRight,
                            bottomRight, bottomLeft])
        card = four_point_transform(image, cardCoords)
        # return the color matching card to the calling function
        return cast(OurImageType, card)

    def find_raw_aruco(self, image : OurImageType) -> Optional[OurImageType]:
        """Find an aruco marker in an image. Return another image, with only the aruco marker, with perspective fixed"""
        # load the ArUCo dictionary, grab the ArUCo parameters, and
        # detect the markers in the input image
        arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        arucoParams = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
        (marker_areas, marker_ids, _) = detector.detectMarkers(image)
        if self.debug:
            outputImage = image.copy()
            cv2.aruco.drawDetectedMarkers(outputImage, marker_areas, marker_ids)
            winTitle = f"Markers."
            cv2.imshow(winTitle, outputImage)
            while True:
                ch = cv2.waitKey()
                if ch == 27:
                    break
                print(f"ignoring key {ch}")
            cv2.destroyWindow(winTitle)
        # try to extract the coordinates of the color correction card
        i = 0
        area = marker_areas[i]
        # I don't understand what detectMarkers returns. There seems to be an extra first dimension
        if area.shape[0] != 4:
            area = area[0]
        topLeft = area[0]
        topRight = area[1]
        bottomRight = area[2]
        bottomLeft = area[3]

        # build our list of reference points and apply a perspective
        # transform to obtain a top-down, bird’s-eye view of the color
        # matching card
        cardCoords = np.array([topLeft, topRight,
                            bottomRight, bottomLeft])
        card = four_point_transform(image, cardCoords)
        # return the color matching card to the calling function
        return cast(OurImageType, card)


    def _create_colormap_1channel(self, source : OurImageType1Channel, template : OurImageType1Channel) -> OurColormap1Channel:
        """
        Helper function: create single-channel colormap.

        Return modified full image array so that the cumulative density function of
        source array matches the cumulative density function of the template.
        """
        src_values, _, src_counts = np.unique(source.ravel(),
                                                            return_inverse=True,
                                                            return_counts=True)
        tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)

        # calculate normalized quantiles for each array
        src_quantiles = np.cumsum(src_counts) / source.size
        tmpl_quantiles = np.cumsum(tmpl_counts) / template.size

        interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)

        # Here we compute values which the channel RGB value of full image will be modified to.
        colormap_1channel = np.ones(256, dtype=np.int32) * -1
    
        # first compute which values in src image transform to and mark those values.

        for i in range(0, len(interp_a_values)):
            frm = src_values[i]
            to = interp_a_values[i]
            colormap_1channel[frm] = to

        # some of the pixel values might not be there in interp_a_values, interpolate those values using their
        # previous and next neighbours
        prev_value = -1
        prev_index = -1
        for i in range(0, 256):
            if colormap_1channel[i] == -1:
                next_index = -1
                next_value = -1
                for j in range(i + 1, 256):
                    if colormap_1channel[j] >= 0:
                        next_value = colormap_1channel[j]
                        next_index = j
                if prev_index < 0:
                    colormap_1channel[i] = (i + 1) * next_value / (next_index + 1)
                elif next_index < 0:
                    colormap_1channel[i] = prev_value + ((255 - prev_value) * (i - prev_index) / (255 - prev_index))
                else:
                    colormap_1channel[i] = prev_value + (i - prev_index) * (next_value - prev_value) / (next_index - prev_index)
            else:
                prev_value = colormap_1channel[i]
                prev_index = i
        return colormap_1channel

    def create_colormap(self, source : OurImageType, template : OurImageType) -> None:
        """Given a source image capture of a colorcard and a template for that colorcard return the colormap to correct the source image."""
        rv : OurColormap = []
        assert source.shape[-1] == template.shape[-1]
        for channel in range(source.shape[-1]):
            map = self._create_colormap_1channel(source[..., channel], template[..., channel])
            rv.append(map)
        self.colormap = rv

    def load_colormap(self, filename : str) -> None:
        """Load a colormap from a .cube file"""
        with open(filename) as fp:
            line = fp.readline()
            if line != 'LUT_1D_SIZE 256\n':
                print(f"{filename}: header: only LUT_1d_SIZE 256 supported. Got {line}")
                assert False
            line = fp.readline()
            if line != 'LUT_1D_INPUT_RANGE 0 255\n':
                print(f"{filename}: header: only LUT_1D_INPUT_RANGE 0 255 supported. Got {line}")
                assert False
            colormap_r = np.ones(256, dtype=np.int32) * -1
            colormap_g = np.ones(256, dtype=np.int32) * -1
            colormap_b = np.ones(256, dtype=np.int32) * -1
            for i in range(256):
                line = fp.readline()
                line = line.strip()
                r, g, b = line.split()
                colormap_r[i] = int(r)
                colormap_g[i] = int(g)
                colormap_b[i] = int(b)
            self.colormap = [colormap_r, colormap_g, colormap_b]

    def save_colormap(self, filename : str) -> None:
        """Save the colormap to a .cube file"""
        assert not self.colormap is None
        # .cube file format gotten from https://resolve.cafe/developers/luts/
        with open(filename, "w") as fp:
            fp.write('LUT_1D_SIZE 256\n')
            fp.write('LUT_1D_INPUT_RANGE 0 255\n')
            for i in range(256):
                r = self.colormap[0][i]
                g = self.colormap[1][i]
                b = self.colormap[2][i]
                fp.write(f"{r} {g} {b}\n")


    def _map_image_color_1channel(self, full_1channel : OurImageType1Channel, colormap_1channel : OurColormap1Channel) -> OurImageType1Channel:
        """Helper function: map color for a single channel"""
        wid = full_1channel.shape[1]
        hei = full_1channel.shape[0]
        ret2 = np.zeros((hei, wid), dtype=np.uint8)
        for i in range(0, hei):
            for j in range(0, wid):
                ret2[i][j] = colormap_1channel[full_1channel[i][j]]
        return ret2

    def map_image(self, fullImage : OurImageType) -> OurImageType:
        """Given a source image return a new image that has the colors mapped"""
        assert not self.colormap is None
        assert len(self.colormap) == fullImage.shape[-1]
        matched = np.empty(fullImage.shape, dtype=fullImage.dtype)
        for channel in range(len(self.colormap)):
            matched_channel = self._map_image_color_1channel(fullImage[..., channel], self.colormap[channel])
            matched[..., channel] = matched_channel
        return matched
        
    def full_process_images(self, inputCard : OurImageType, referenceCard : OurImageType, fullImage : OurImageType) -> OurImageType:
        """
            Return modified full image, by using histogram equalization on input and
            reference cards and applying that transformation on fullImage.
        """
        self.create_colormap(inputCard, referenceCard)
        rv = self.map_image(fullImage)
        if self.debug:
            self.plot_colormap()
        return rv

    def plot_colormap(self) -> None:
        """Show a plot of the colormap as color curves"""
        assert not self.colormap is None
        ngraph = len(self.colormap)
        fig, axis = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True)
        colors = ['red', 'green', 'blue']
        for i in range(ngraph):
            data = self.colormap[i]
            axis.plot(data, color=colors[i], label=colors[i])
        axis.set_aspect(1)
        axis.set_xticks(np.arange(0, 257, 32))
        axis.set_yticks(np.arange(0, 257, 32))
        axis.set_xlabel("Input")
        axis.set_ylabel("Output")
        axis.grid(linestyle='dotted')
        fig.legend()
        plt.show()


def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--reference", required=False,
                    help="path to the input reference image")
    ap.add_argument("--loadlut", metavar="FILE", help="Don't search for colorcards, load a .cube FILE for colormapping")
    ap.add_argument("-w", "--width0", required=False,
                    help="Image Size")
    ap.add_argument("-v", "--view", required=False, default=False, action='store_true',
                    help="Image Preview?")
    ap.add_argument("-o", "--output", required=False, default=False,
                    help="Image Output Path")
    ap.add_argument("--savelut", metavar="FILE", help="Save the colormapping to FILE in .cube format")
    ap.add_argument("-i", "--input", required=True,
                    help="path to the input image to apply color correction to")
    ap.add_argument("-d", "--debug", action="store_true", help="Debug card finder")
    ap.add_argument("--raw_aruco", action="store_true", help="Don't search for a colorcard, search for a raw aruco marker")
    args = vars(ap.parse_args())

    corrector = ColorCorrector(debug=args["debug"])

    img1 : Optional[OurImageType] = None
    # load the reference image and input images from disk

    if args["loadlut"]:
        corrector.load_colormap(args["loadlut"])
    else:
        print("[INFO] loading images...")

        img1 = cast(OurImageType, cv2.imread(args["input"]))
        file_exists = pathfile.isfile(args["reference"])

        if not file_exists:
            print('[WARNING] Referenz File not exisits '+str(args["reference"]))
            sys.exit()


        raw = cast(OurImageType, cv2.imread(args["reference"]))

        height, width0, channels = raw.shape
        height2, width1, channels = img1.shape


        # resize the reference and input images

        newWidth=width0//3
        countStep=400
        goOn=False
        # First try without resizing
        raw_ = raw
        img1_ = img1
        if args["raw_aruco"]:
            rawCard = corrector.find_raw_aruco(raw_)
            imageCard = corrector.find_raw_aruco(img1_)
        else:
            rawCard = corrector.find_color_card(raw_)
            imageCard = corrector.find_color_card(img1_)
        if not rawCard is None and not imageCard is None:
            goOn = True

        while goOn==False and newWidth<=width0:

            raw_ = imutils.resize(raw, newWidth)
            img1_ = imutils.resize(img1, newWidth)

        
            print("[INFO] Finding color matching cards width "+ repr(newWidth)+"px")
            rawCard = corrector.find_color_card(raw_)
            imageCard = corrector.find_color_card(img1_)
            
            if rawCard is None or imageCard is None:
                oldW =newWidth
                newWidth +=countStep
                print("[INFO] Could not find color with width "+ repr(oldW)+"px. Try width:"+ repr(newWidth)+"px")
                continue
            else:
                goOn=True
                break

        if(goOn is False):
            print("[WARNING] Could not find color matching cards in both images. Try a highter/better Resolution")
            
            sys.exit()
        assert not rawCard is None
        assert not imageCard is None
        if args['view']:
                cv2.imshow("Reference", raw_)
                cv2.imshow("Input", img1_)
        # show the color matching card in the reference image and input image,
        # respectively
        if args['view']:
            cv2.imshow("Reference Color Card", rawCard)
            cv2.imshow("Input Color Card", imageCard)
        corrector.create_colormap(imageCard, rawCard)
    # apply histogram matching from the color matching card in the
    # reference image to the color matching card in the input image
    print("[INFO] matching images...")

    result2 : Optional[OurImageType] = None

    if args["input"]:
        if img1 is None:
            img1 = cast(OurImageType, cv2.imread(args["input"]))
        
        if args["width0"]:
            width=int(args["width0"])
            if width>1:    
                print('resize Final: '+repr(width))
                img1 = cast(OurImageType, imutils.resize(img1, width))

        result2 = corrector.map_image(img1)

    # show our input color matching card after histogram matching
    #cv2.imshow("Input Color Card After Matching", inputCard)

    if args["savelut"]:
        corrector.save_colormap(args["savelut"])

    if args['view']:
        if not result2 is None:
            cv2.imshow("Output image", result2)

    if args['output']:
        assert not result2 is None
        file_ok = exists(args['output'].lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')))

        if file_ok:
            cv2.imwrite(args['output'], result2)
            print("[SUCCESSUL] Your Image was written to: "+args['output']+"")
        else:
            print("[WARNING] Sorry, But this is no valid Image Name "+args['output']+"\nPlease Change Parameter!")

    if args['view']:
        cv2.waitKey(0)

if __name__ == "__main__":
    main()