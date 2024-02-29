import cv2
import cv2.videoio_registry
import sys
import argparse
import color_correction

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", action="store_true", help="View resultant video")
    parser.add_argument("--lutfile", help="Color lookup table, in .cube format")
    parser.add_argument("--output", help="Output file for resultant video")
    parser.add_argument("--count", metavar="NUM", type=int, help="Convert only NUM frames (default: all)")
    parser.add_argument("input", help="Input video file")
    args = parser.parse_args()

    if False:
        backends = cv2.videoio_registry.getBackends()
        for backend in backends:
            name = cv2.videoio_registry.getBackendName(backend)
            print(f"xxxjack backend {backend} name {name}")
    corrector = None
    if args.lutfile:
        corrector = color_correction.ColorCorrector()
        corrector.load_colormap(args.lutfile)

    input_file = cv2.VideoCapture(args.input, cv2.CAP_FFMPEG)
    if not input_file.isOpened():
        print(f"{sys.argv[0]}: {args.input}: Failed to open input video file")
        sys.exit(1)
    output_file = None
    fourcc = int(input_file.get(cv2.CAP_PROP_FOURCC))
    fps = int(input_file.get(cv2.CAP_PROP_FPS))
    width = int(input_file.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_file.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{args.input}: {width} * {height} @ {fps} fps, 4CC=0x{fourcc:08x}")
    if args.output:
        output_file = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        if not output_file.isOpened():
            print(f"{sys.argv[0]}: {args.output}: Failed to open output video file")
            sys.exit(1)
    framenum = 0
    while True:
        if args.count and framenum >= args.count:
            break
        ret, frame = input_file.read()
        if not ret: break
        if corrector:
            new_frame = corrector.map_image(frame)
            frame = new_frame
        if args.view:
            cv2.imshow(f"mapped", frame)
            cv2.waitKey(1)
        if output_file:
            output_file.write(frame)
        framenum += 1
        print(f"xxxjack did image {framenum}")
    print(f"{sys.argv[0]}: processed {framenum} frames")
    if input_file:
        input_file.release()
    if output_file:
        output_file.release()

if __name__ == "__main__":
    main()
