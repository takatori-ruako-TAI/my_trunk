import os
import argparse
import glob
import numpy as np
import cv2
import random


def clip_random(in_path, out_size, out_dir, base_filename, out_ext, seed_value):
    # 画像の読み込み
    if os.path.exists(in_path):
        in_image = cv2.imread(in_path)
    else:
        print("[ERR] in file cannot load ", in_path)
        return
    width = in_image.shape[1]  # 画像幅
    height = in_image.shape[0]  # 画像高さ

    size_min = out_size // 2
    size_max = max(min(width, height) // 2, size_min) # 短辺の半分サイズまで取る
    max_row = height // size_min + 1
    max_col = width // size_min + 1
    print('img size', width, height, 'sq size', size_min, size_max)
    print('row-col', max_row, max_col)

    random.seed(seed_value)

    out_path_format = os.path.join(out_dir, base_filename + '_{:07}'.format(seed_value) + '_{:05}.' + out_ext)
    out_num = 0
    cursor = [0, 0]
    for row in range(max_row):
        go_next_row = False
        for col in range(max_col):
            clip_size = random.randint(size_min, size_max)
            cursorx, cursory = cursor
            if cursorx + clip_size >= width:
                cursorx = width - clip_size + 1
                go_next_row = True
            if cursory + clip_size >= height:
                cursory = height - clip_size + 1

            out_path = out_path_format.format(out_num)
            clipped_image = in_image[cursory:cursory+clip_size, cursorx:cursorx+clip_size]
            print('output', out_num, cursor, clip_size, clipped_image.shape, out_path)
            resized_image = cv2.resize(clipped_image, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(out_path, resized_image, [cv2.IMWRITE_WEBP_QUALITY, 100])
            cursor[0] += max(clip_size, size_min)
            out_num += 1

            if go_next_row or cursor[0] >= width:
                break

        cursor[0] = 0
        cursor[1] += max(clip_size, size_min)
        if cursor[1] >= height:
            break

def maxclip(in_path, out_size, out_dir, base_filename, out_ext):
    # 画像の読み込み
    if os.path.exists(in_path):
        in_image = cv2.imread(in_path)
    else:
        print("[ERR] in file cannot load ", in_path)
        return
    width = in_image.shape[1]  # 画像幅
    height = in_image.shape[0]  # 画像高さ
    clip_size = min(width, height)

    max_row = height // clip_size + 1
    max_col = width // clip_size + 1

    out_path_format = os.path.join(out_dir, base_filename + '_maxclip' + '_{:05}.' + out_ext)
    out_num = 0
    cursor = [0, 0]
    for row in range(max_row):
        go_next_row = False
        for col in range(max_col):
            cursorx, cursory = cursor
            if cursorx + clip_size >= width:
                cursorx = width - clip_size + 1
                go_next_row = True
            if cursory + clip_size >= height:
                cursory = height - clip_size + 1

            out_path = out_path_format.format(out_num)
            clipped_image = in_image[cursory:cursory+clip_size, cursorx:cursorx+clip_size]
            print('output', out_num, cursor, clip_size, clipped_image.shape, out_path)
            resized_image = cv2.resize(clipped_image, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(out_path, resized_image, [cv2.IMWRITE_WEBP_QUALITY, 100])
            cursor[0] += clip_size
            out_num += 1

            if go_next_row or cursor[0] >= width:
                break

        cursor[0] = 0
        cursor[1] += clip_size
        if cursor[1] >= height:
            break


def main(args):
    if not os.path.exists(args.in_img):
        print("[ERR] no --in_img specified")
        return
    in_paths = sorted(glob.glob(os.path.join(args.in_img, "*."+args.in_ext)))
    if len(in_paths) < 1:
        print('no input files')
        return

    # 出力先フォルダの作成
    if args.out_img is not None:
        if not os.path.exists(args.out_img):
            os.makedirs(args.out_img)

    if args.mode == 'random':
        if args.seed is None:
            seed_value = random.randint(0, 999999)
        else:
            seed_value = args.seed
        out_random_dir = os.path.join(args.out_img, '{:07}'.format(seed_value))
        if not os.path.exists(out_random_dir):
            os.makedirs(out_random_dir)

    for in_path in in_paths:
        base_filename = os.path.splitext(os.path.basename(in_path))[0]
        print(base_filename, '----------')
        if args.mode == 'random':
            clip_random(in_path, args.out_size, out_random_dir, base_filename, args.out_ext, seed_value)
        elif args.mode == 'max':
            maxclip(in_path, args.out_size, args.out_img, base_filename, args.out_ext)
        else:
            print('set mode')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='画像から正方形を切り出す')
    parser.add_argument('--mode', default='None', type=str, help="どう切り出すかのモード(random/..)")
    parser.add_argument('--in_img', default=None, type=str, help="入力画像フォルダパス")
    parser.add_argument('--in_ext', default='webp', type=str, help="入力画像の拡張子 ")
    parser.add_argument('--seed', default=None, type=int, help="[mode:random] 乱数シード値")

    parser.add_argument('--out_size', default=322, type=int, help="出力正方形サイズ")

    parser.add_argument('--out_img', default=None, type=str, help="出力画像のフォルダパス ")
    parser.add_argument('--out_ext', default='webp', type=str, help="出力画像の拡張子 ")
    args = parser.parse_args()

    main(args)
