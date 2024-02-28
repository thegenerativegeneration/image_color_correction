import cv2
import sys
import argparse
import color_correction

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", action="store_true", help="View resultant video")
    parser.add_argument("--lutfile", help="Color lookup table, in .cube format")
    parser.add_argument("--output", help="Output file for resultant video")
    parser.add_argument("input", help="Input video file")
    args = parser.parse_args()

    corrector = None
    if args.lutfile:
        corrector = color_correction.ColorCorrector()
        corrector.load_colormap(args.lutfile)

    input_file = cv2.VideoCapture(args.input)
    if not input_file.isOpened():
        print(f"{sys.argv[0]}: {args.input}: Failed to open input video file")
        sys.exit(1)
    output_file = None
    #output_file = cv2.VideoWriter(args.output, )
    count = 0
    while True:
        ret, frame = input_file.read()
        if not ret: break
        if corrector:
            new_frame = corrector.map_image(frame)
            frame = new_frame
        if args.view:
            cv2.imshow(f"mapped", frame)
            cv2.waitKey(0)
        if output_file:
            assert 0
        count += 1
    print(f"{sys.argv[0]}: processed {count} frames")

if __name__ == "__main__":
    main()
