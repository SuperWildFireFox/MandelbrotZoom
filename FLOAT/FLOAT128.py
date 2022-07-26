import taichi as ti
from decimal import Decimal, getcontext

"""
设置外部精度
2**-96 = 1.3e-29 28位精度
"""
getcontext().prec = 50

F128 = ti.types.vector(4, ti.int32)
_INT2 = ti.types.vector(2, ti.int32)

F128_ALL = F128(-1)


def cvt_int_bin(n, bit=32):
    assert -0x80000000 <= n <= 0x7fffffff
    if n < 0:
        n = n + 2 ** bit
    bin_str = bin(n).replace("0b", "")
    if len(bin_str) < bit:
        bin_str = "0" * (bit - len(bin_str)) + bin_str
    return bin_str


def cvt_int_hex(n, bit=32):
    if n < 0:
        n = n + 2 ** bit
    hex_str = hex(n).replace("0x", "")
    byte = bit % 8
    if len(hex_str) < byte:
        hex_str = "f" * (byte - len(hex_str)) + hex_str
    return hex_str


def cvt_int_uint(n, bit=32):
    assert -0x80000000 <= n <= 0x7fffffff
    if n < 0:
        n = n + 2 ** bit
    return n


def cvt_uint_int(n, bit=32):
    assert 0 <= n <= 0xffffffff
    if n > 0x7fffffff:
        n = n - 2 ** bit
    return n


def F128_print(f):
    print("==============")
    print(f)
    print("DEC:", f[0], f[1], f[2], f[3])
    print("BIN:", cvt_int_bin(f[0]), cvt_int_bin(f[1]), cvt_int_bin(f[2]), cvt_int_bin(f[3]))
    print("HEX:", cvt_int_hex(f[0]), cvt_int_hex(f[1]), cvt_int_hex(f[2]), cvt_int_hex(f[3]))
    print("==============")


# 字符串转F128，不支持科学计数法，不要有前缀加号，不要有前缀0
# 补码形式存储，保证二进制正确
def str2F128(s: str, bit=32):
    lim = bit * 3
    res = F128(0)
    neg = False
    if s[0] == '-':
        neg = True
    if s.find(".") == -1:
        r = s
        d = "0"
    else:
        r, d = s.split(".")
    res[0] = int(r)
    assert -0x80000000 <= res[0] <= 0x7fffffff
    d = Decimal("0." + d)
    if neg:
        d = 1 - d
        res[0] -= 1
    bit_list = []
    while True:
        d = d * 2
        if d < 1:
            bit_list.append("0")
        elif d > 1:
            bit_list.append("1")
            d = d - 1
        elif d == 1:
            bit_list.append("1")
            break
        if len(bit_list) == lim:
            break
    if len(bit_list) != lim:
        bit_list = bit_list + ["0" for i in range(lim - len(bit_list))]
    res[1] = cvt_uint_int(int("".join(bit_list[0:bit]), 2))
    res[2] = cvt_uint_int(int("".join(bit_list[bit:bit * 2]), 2))
    res[3] = cvt_uint_int(int("".join(bit_list[bit * 2:bit * 3]), 2))
    return res


# F128转字符串
def F1282str(f):
    d = Decimal(f[0])
    bit_str = cvt_int_bin(f[1]) + cvt_int_bin(f[2]) + cvt_int_bin(f[3])
    base = Decimal(0.5)
    mul = Decimal(0.5)
    for c in bit_str:
        if c == '1':
            d = d + base
        base = base * mul
    return d


@ti.func
# F128,增加1最小单位
def inc128(u) -> F128:
    h = u == F128_ALL
    c = F128(h[1] & h[2] & h[3] & 1, h[2] & h[3] & 1, h[3] & 1, 1)
    return u + c


@ti.func
# 取二进制意义上的相反数
def neg128(u) -> F128:
    u = u ^ F128_ALL
    return inc128(u)


@ti.func
# return U<V
def uint32_less(u, v):
    u0: ti.uint32 = u[0]
    u1: ti.uint32 = u[1]
    u2: ti.uint32 = u[2]
    u3: ti.uint32 = u[3]
    v0: ti.uint32 = v[0]
    v1: ti.uint32 = v[1]
    v2: ti.uint32 = v[2]
    v3: ti.uint32 = v[3]
    return F128(u0 < v0, u1 < v1, u2 < v2, u3 < v3)


@ti.func
# F128+F128
def add128(u, v) -> F128:
    s = u + v
    # h = s < u
    h = uint32_less(s, u)
    c1 = F128(h[1], h[2], h[3], h[0]) & F128(1, 1, 1, 0)
    h = s == F128_ALL
    c2 = F128((c1[1] | (c1[2] & h[2])) & h[1], c1[2] & h[2], 0, 0)
    return s + c1 + c2


@ti.func
# u<<1
def shl128(u):
    h = ti.bit_shr(u, 31) & F128(0, 1, 1, 1)
    return u << 1 | F128(h[1], h[2], h[3], h[0])


@ti.func
# 计算两个int32位乘法，返回高32位与低32位
def mul_hi_lo(op1: ti.int32, op2: ti.int32) -> _INT2:
    u1 = op1 & 0xffff
    v1 = op2 & 0xffff
    t = u1 * v1
    w3 = t & 0xffff
    k = ti.bit_shr(t, 16)
    op1 = ti.bit_shr(op1, 16)
    t = op1 * v1 + k
    k = t & 0xffff
    w1 = ti.bit_shr(t, 16)
    op2 = ti.bit_shr(op2, 16)
    t = (u1 * op2) + k
    k = ti.bit_shr(t, 16)
    hi = op1 * op2 + w1 + k
    lo = (t << 16) + w3
    return _INT2(hi, lo)


@ti.func
# return U*V
def mulfpu(u, v):
    neg = 0
    if u[0] < 0:
        u = neg128(u)
        neg += 1
    if v[0] < 0:
        v = neg128(v)
        neg += 1
    _t = mul_hi_lo(u[1], v[1])
    s = F128(u[0] * v[0], _t[0], _t[1], mul_hi_lo(u[2], v[2])[0])
    _t = mul_hi_lo(u[0], v[1])
    _t1 = mul_hi_lo(u[0], v[3])
    t1 = F128(_t[0], _t[1], _t1[0], _t1[1])
    _t = mul_hi_lo(v[0], u[1])
    _t1 = mul_hi_lo(v[0], u[3])
    t2 = F128(_t[0], _t[1], _t1[0], _t1[1])
    s = add128(s, add128(t1, t2))
    _t = mul_hi_lo(u[0], v[2])
    t1 = F128(0, _t[0], _t[1], mul_hi_lo(u[1], v[3])[0])
    _t = mul_hi_lo(v[0], u[2])
    t2 = F128(0, _t[0], _t[1], mul_hi_lo(v[1], u[3])[0])
    s = add128(s, add128(t1, t2))
    _t = mul_hi_lo(u[1], v[2])
    t1 = F128(0, 0, _t[0], _t[1])
    _t = mul_hi_lo(v[1], u[2])
    t2 = F128(0, 0, _t[0], _t[1])
    s = add128(s, add128(t1, t2))
    res = add128(s, F128(0, 0, 0, 3))
    if neg == 1:
        res = neg128(res)
    return res


@ti.func
# return U^2
def sqrfpu(u):
    if u[0] < 0:
        u = neg128(u)
    _t = mul_hi_lo(u[1], u[1])
    s = F128(u[0] * u[0], _t[0], _t[1], mul_hi_lo(u[2], u[2])[0])
    _t = mul_hi_lo(u[0], u[1])
    _t1 = mul_hi_lo(u[0], u[3])
    t = F128(_t[0], _t[1], _t1[0], _t1[1])
    s = add128(s, shl128(t))
    _t = mul_hi_lo(u[0], u[2])
    t = F128(0, _t[0], _t[1], mul_hi_lo(u[1], u[3])[0])
    s = add128(s, shl128(t))
    _t = mul_hi_lo(u[1], u[2])
    t = F128(0, 0, _t[0], _t[1])
    s = add128(s, shl128(t))
    res = add128(s, F128(0, 0, 0, 3))
    return res
