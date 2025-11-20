import os
import time
import argparse
import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
from depthanythingv2_openvino import DepthAnythingV2OpenVino

def show_results(source_img, depth_img, output_dir, input_dir):
    if output_dir != '' and not os.path.exists(output_dir):
        os.mkdir(output_dir)

    basename = os.path.basename(input_dir)

    plt.subplot(121)
    plt.title("source")
    plt.imshow(cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB))
    plt.axis("on")

    plt.subplot(122)
    plt.title("depth")
    plt.imshow(cv2.cvtColor(depth_img, cv2.COLOR_BGR2RGB))
    plt.axis("on")

    if output_dir != '':
        plt.imsave(os.path.join(output_dir, basename), cv2.cvtColor(depth_img, cv2.COLOR_BGR2RGB))
        plt.savefig(os.path.join(output_dir, 'plt_' + basename))

    # plt.show()

    return

def main(args):
    if not os.path.exists(args.input_dir):
        print("[ERR] no --input_dir specified")
        return

    input_files = sorted(glob.glob(os.path.join(args.input_dir, "*."+args.input_ext)))
    if len(input_files) < 1:
        print('no input files')
        return

    #--- model 初期化 ---
    print("loading model with pretrained weights...")
    model = DepthAnythingV2OpenVino(args.model_path, args.device, args.input_size)
    print("model load completed.")

    # 出力先フォルダの作成
    if args.output_dir is not None:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    for input_file in input_files:
        print(input_file)
        # 入力画像読み込み
        source_img = cv2.imread(input_file)

        # 入力画像をモデルの画像入力サイズへリサイズ
        dst_size = model.get_inputsize()
        resized_img = cv2.resize(source_img, (dst_size, dst_size),interpolation=cv2.INTER_LANCZOS4)

        # 推論
        start = time.perf_counter()
        depth_img, depth_raw = model.infer_image(resized_img)
        end = time.perf_counter()
        print("{:.2f} sec".format(end-start))

        # 入力画像サイズにリサイズ
        dst_img = cv2.resize(depth_img, (source_img.shape[1], source_img.shape[0]),interpolation=cv2.INTER_LANCZOS4)

        # 結果表示
        show_results(source_img, dst_img, args.output_dir, input_file)

        base_filename = os.path.splitext(os.path.basename(input_file))[0]
        # 生データ保存
        if args.output_csv != '':
            np.savetxt(os.path.join(args.output_csv, base_filename + '.csv'), depth_raw, delimiter=',')
        if args.output_npy != '':
            np.save(os.path.join(args.output_npy, base_filename + '.npy'), depth_raw.astype(np.float32))

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='入力単位をフォルダにし複数画像を一度に処理する')
    parser.add_argument('--model_path', type=str, help='Path to the int8 Quantized OpenVINO IR model .xml file')
    parser.add_argument('--input_dir', type=str, help='Input directory path')
    parser.add_argument('--input_ext', type=str, help='Input file ext')
    parser.add_argument('--output_dir', type=str, default='', help='Path to the output directory')
    parser.add_argument('--output_npy', type=str, default='', help='npy firectory name for raw depth data')
    parser.add_argument('--output_csv', type=str, default='', help='csv firectory name for raw depth data')
    parser.add_argument('--device', type=str, default='AUTO', help='AUTO/CPU/GPU')
    parser.add_argument("--input_size", type=int, default=518, help="Input tensor size in multiples of patch height 14")
    args = parser.parse_args()

    main(args)
