from scipy.interpolate import pchip_interpolate
import numpy as np


def get_rgb_color():
    # set up the control points for your gradient
    yR_observed = [0, 0, 32, 237, 255, 0, 0, 32]
    yG_observed = [2, 7, 107, 255, 170, 2, 7, 107]
    yB_observed = [0, 100, 203, 255, 0, 0, 100, 203]

    x_observed = [-.1425, 0, .16, .42, .6425, .8575, 1, 1.16]

    # Create the arrays with the interpolated values
    x = np.linspace(min(x_observed), max(x_observed), num=1000)
    yR = pchip_interpolate(x_observed, yR_observed, x)
    yG = pchip_interpolate(x_observed, yG_observed, x)
    yB = pchip_interpolate(x_observed, yB_observed, x)

    # Convert them back to python lists
    x = list(x)
    yR = list(yR)
    yG = list(yG)
    yB = list(yB)

    # Find the indexs where x crosses 0 and crosses 1 for slicing
    start = 0
    end = 0
    for i in x:
        if i > 0:
            start = x.index(i)
            break

    for i in x:
        if i > 1:
            end = x.index(i)
            break

    # Slice away the helper data in the begining and end leaving just 0 to 1
    x = x[start:end]
    yR = np.expand_dims(yR[start:end], 0)
    yG = np.expand_dims(yG[start:end], 0)
    yB = np.expand_dims(yB[start:end], 0)
    colors = np.concatenate((yR, yG, yB), axis=0)
    return colors.T.astype(np.uint8)


if __name__ == '__main__':
    colors = get_rgb_color()
