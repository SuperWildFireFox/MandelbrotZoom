import argparse
from decimal import Decimal
import math
import os.path
from FLOAT.FLOAT256 import *
from util.generate_colors import get_rgb_color
import time


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=300, help='最终输出图片宽度')
    parser.add_argument("--height", type=int, default=200, help='最终输出图片高度')
    parser.add_argument("--SUPER_SAMPLE_MUL", type=int, default=4, help='超采样比例（越大渲染越慢，但是能适当增强渲染结果）')
    parser.add_argument("--center_x", type=str, default="0.3602404434376143632361252444495453084"
                                                        "826078079585857504883758147401953460592"
                                                        "181003117529367227734263962337", help='缩放中心实部值(x)')
    parser.add_argument("--center_y", type=str, default="-0.6413130610648031748603750151793020665"
                                                        "7949495228230525955617754306444857417275"
                                                        "3690255637023068968116237074", help='缩放中心虚部值(y)')
    parser.add_argument("--zoom", type=float, default=1.4, help='两桢之间缩放倍数比例')
    parser.add_argument("--save_path", type=str, default="result", help='结果保存位置')
    parser.add_argument("--MAX_DRAW", type=int, default=500, help='最大渲染的图片数')
    return parser.parse_args()


args = get_args()
half_x = str2F256("2")  # -2~2 为渲染区域实部范围（等于x轴范围）
half_y = str2F256(str(Decimal(2) * args.height / args.width))
colors = get_rgb_color()  # 获取计算好的颜色
zoom = str2F256(str(Decimal(1) / Decimal(args.zoom)))  # 获取高精度缩放倍数倒数
size_w = str2F256(str(Decimal(1) / Decimal(args.SUPER_SAMPLE_MUL * args.width)))  # 预先计算倒数以减少计算量
size_h = str2F256(str(Decimal(1) / Decimal(args.SUPER_SAMPLE_MUL * args.height)))
center_x = str2F256(args.center_x)
center_y = str2F256(args.center_y)

# taichi数据区
# 开启高级优化连续两个mulfpu会很诡异地在第二个乘法处输出错误结果，故关闭
ti.init(arch=ti.gpu, default_fp=ti.f64, default_ip=ti.i64, advanced_optimization=False)

# 画板
pixels = ti.Vector.field(n=3, dtype=ti.u8, shape=(args.width, args.height))
pixels_2d = ti.field(dtype=ti.u8, shape=(args.width, args.height))
super_pixels = ti.Vector.field(n=3, dtype=ti.u8, shape=(args.SUPER_SAMPLE_MUL * args.width,
                                                        args.SUPER_SAMPLE_MUL * args.height))
super_pixels_2d = ti.field(dtype=ti.u8, shape=(args.SUPER_SAMPLE_MUL * args.width,
                                               args.SUPER_SAMPLE_MUL * args.height))
# 颜色
RGB = ti.Vector.field(n=3, dtype=ti.u8, shape=767)
RGB.from_numpy(colors)
# 缩放
global_scale = F256.field(shape=())
global_scale[None] = F256(1, 0, 0, 0)
# 坐标缓存
bdPosition = F256.field(shape=(args.SUPER_SAMPLE_MUL * args.width, args.SUPER_SAMPLE_MUL * args.height, 2))


@ti.kernel
# 预计算基点坐标
def prepare():
    print("preparing...")
    # 初始化缩放
    p0 = ti.i32(0)
    p1 = ti.i32(1)
    for i64, j64 in super_pixels_2d:
        # 似乎暂时ti.i64无法作为索引
        i = ti.cast(i64, ti.i32)
        j = ti.cast(j64, ti.i32)
        t = mulfpu(size_w, F256(2 * i, 0, 0, 0))
        t = add256(t, neg256(F256(1, 0, 0, 0)))
        bdPosition[i, j, p0] = mulfpu(half_x, t)
        t = mulfpu(size_h, F256(2 * j, 0, 0, 0))
        t = add256(t, neg256(F256(1, 0, 0, 0)))
        bdPosition[i, j, p1] = mulfpu(half_y, t)
    print("prepare done")


@ti.func
# 设置颜色
def set_color(n_iter: ti.int32, max_iter: ti.int32):
    color_result = ti.Vector([0, 0, 0], dt=ti.uint8)
    if n_iter < max_iter:
        n_iter = ti.cast((n_iter + 0) % 767, ti.i32)  # +n可以用来偏移色带以带来不同渲染效果
        # c = ti.log(ti.log(ti.sqrt(zrsqr + zisqr))) * logBase #连续色带算法
        # c_i = iter + 5 - c
        # c_i_int = ti.cast(c_i, ti.uint32) % 767
        # color_result = RGB[c_i_int]
        color_result = RGB[n_iter]
    return color_result


@ti.kernel
def cal_mandelbrot(max_iter: ti.int32):
    global_scale[None] = mulfpu(global_scale[None], zoom)
    p0 = ti.i32(0)
    p1 = ti.i32(1)
    for i64, j64 in super_pixels_2d:
        i = ti.cast(i64, ti.i32)
        j = ti.cast(j64, ti.i32)
        cr = mulfpu(bdPosition[i, j, p0], global_scale[None])
        cr = add256(cr, center_x)
        ci = mulfpu(bdPosition[i, j, p1], global_scale[None])
        ci = add256(ci, center_y)
        zr = F256(0)
        zi = F256(0)
        zrsqr = F256(0)
        zisqr = F256(0)
        iterations = p0
        while add256(zrsqr, zisqr)[0] < 4 and iterations < max_iter:
            zi = mulfpu(zr, zi)
            zi = add256(zi, zi)
            zi = add256(zi, ci)
            zr = add256(zrsqr, neg256(zisqr))
            zr = add256(zr, cr)
            zrsqr = sqrfpu(zr)
            zisqr = sqrfpu(zi)
            iterations += p1
        super_pixels[i, j] = set_color(iterations, max_iter)


@ti.kernel
def paint():
    SUPER_SAMPLE_MUL_SQR = args.SUPER_SAMPLE_MUL * args.SUPER_SAMPLE_MUL
    for i_i64, j_i64 in pixels_2d:
        i = ti.cast(i_i64, ti.i32)
        j = ti.cast(j_i64, ti.i32)
        base_i = ti.cast(args.SUPER_SAMPLE_MUL * i, ti.i32)
        base_j = ti.cast(args.SUPER_SAMPLE_MUL * j, ti.i32)
        color = ti.Vector([0.0, 0.0, 0.0])
        for i_t_i64 in ti.static(range(args.SUPER_SAMPLE_MUL)):
            for j_t_i64 in ti.static(range(args.SUPER_SAMPLE_MUL)):
                i_t = ti.cast(i_t_i64, ti.i32)
                j_t = ti.cast(j_t_i64, ti.i32)
                color += super_pixels[base_i + i_t, base_j + j_t]
        color = color / SUPER_SAMPLE_MUL_SQR
        color_uint = ti.cast(color, ti.uint8)
        pixels[i, j] = color_uint


if __name__ == '__main__':
    prepare()
    gui = ti.GUI("Mandelbrot Zoom", res=(args.width, args.height))
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    loop = 0
    while gui.running:
        loop += 1
        if loop > args.MAX_DRAW:
            break
        m_iter = ((loop // 100) + 1) * 4000  # 这个计算公式不合理，只是临时凑数
        cal_mandelbrot(m_iter)
        print(loop, F2562str(global_scale[None]))
        # gui.set_image(super_pixels)
        paint()
        gui.set_image(pixels)
        filename = f'{args.save_path}/frame_{loop:05d}.png'
        gui.show(filename)
