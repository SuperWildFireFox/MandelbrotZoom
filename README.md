# MandelbrotZoom

## 程序说明

Mandelbrot缩放实验，利用taichi完成计算与渲染，cuda友好，最高支持10^55倍缩放。

其中10^55倍缩放采用4个int64模拟float256计算，最小支持2E-192精度。



## 使用方法

note: 由于一些bug，安装taichi环境时请自行编译最新版的taichi仓代码，或等taichi-1.0.5正式版发布后再用pip安装。

```
usage: main.py [-h] [--width WIDTH] [--height HEIGHT] [--SUPER_SAMPLE_MUL SUPER_SAMPLE_MUL] [--center_x CENTER_X]
               [--center_y CENTER_Y] [--zoom ZOOM] [--save_path SAVE_PATH] [--MAX_DRAW MAX_DRAW]

optional arguments:
  -h, --help            show this help message and exit
  --width WIDTH         最终输出图片宽度
  --height HEIGHT       最终输出图片高度
  --SUPER_SAMPLE_MUL SUPER_SAMPLE_MUL
                        超采样比例（越大渲染越慢，但是能适当增强渲染结果）
  --center_x CENTER_X   缩放中心实部值(x)
  --center_y CENTER_Y   缩放中心虚部值(y)
  --zoom ZOOM           两桢之间缩放倍数比例
  --save_path SAVE_PATH
                        结果保存位置
  --MAX_DRAW MAX_DRAW   最大渲染的图片数

```

可以直接输入

```
python main.py
```

查看默认参数下的缩放结果



## 参考资料

- [Vectorized fp128 for OpenCL](http://www.bealto.com/mp-mandelbrot_fp128-opencl.html)
- [mul_hi函数模拟](https://stackoverflow.com/questions/25095741/how-can-i-multiply-64-bit-operands-and-get-128-bit-result-portably/25096197#25096197)
- [Faster Fractals Through Algebra](https://randomascii.wordpress.com/2011/08/13/faster-fractals-through-algebra/)
- [taichi wiki](https://docs.taichi-lang.org/zh-Hans/docs/)
- [mandelbrot平滑着色](https://stackoverflow.com/questions/16500656/which-color-gradient-is-used-to-color-mandelbrot-in-wikipedia)
- [Mandelbrot：超级采样与多重采样？](http://www.fractalforums.com/index.php?topic=18782.msg72520#msg72520)