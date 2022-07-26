import taichi as ti
from decimal import Decimal, getcontext

"""
设置外部精度
2**-192 = 1.6e-58
"""
getcontext().prec = 100

F256 = ti.types.vector(4, ti.int64)
_INT2 = ti.types.vector(2, ti.int64)

F256_ALL = F256(-1)

INT64_MIN = -0x8000000000000000
INT64_MAX = 0x7fffffffffffffff
UINT64_MAX = 0xffffffffffffffff


def cvt_int_bin(n, bit=64):
    assert INT64_MIN <= n <= INT64_MAX
    if n < 0:
        n = n + 2 ** bit
    bin_str = bin(n).replace("0b", "")
    if len(bin_str) < bit:
        bin_str = "0" * (bit - len(bin_str)) + bin_str
    return bin_str


def cvt_int_hex(n, bit=64):
    if n < 0:
        n = n + 2 ** bit
    hex_str = hex(n).replace("0x", "")
    byte = bit % 8
    if len(hex_str) < byte:
        hex_str = "f" * (byte - len(hex_str)) + hex_str
    return hex_str


def cvt_int_uint(n, bit=64):
    assert INT64_MIN <= n <= INT64_MAX
    if n < 0:
        n = n + 2 ** bit
    return n


def cvt_uint_int(n, bit=64):
    assert 0 <= n <= UINT64_MAX
    if n > INT64_MAX:
        n = n - 2 ** bit
    return n


def F256_print(f):
    print("==============")
    print(f)
    print("DEC:", f[0], f[1], f[2], f[3])
    print("BIN:", cvt_int_bin(f[0]), cvt_int_bin(f[1]), cvt_int_bin(f[2]), cvt_int_bin(f[3]))
    print("HEX:", cvt_int_hex(f[0]), cvt_int_hex(f[1]), cvt_int_hex(f[2]), cvt_int_hex(f[3]))
    print("==============")


# 字符串转F256，不支持科学计数法，不要有前缀加号，不要有前缀0
# 补码形式存储，保证二进制正确
def str2F256(s: str, bit=64):
    lim = bit * 3
    res = F256(0)
    neg = False
    if s[0] == '-':
        neg = True
    if s.find(".") == -1:
        r = s
        d = "0"
    else:
        r, d = s.split(".")
    res[0] = int(r)
    assert INT64_MIN <= res[0] <= INT64_MAX
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


# F256转字符串
def F2562str(f):
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
# F256,增加1最小单位
def inc256(u) -> F256:
    h = u == F256_ALL
    c = F256(h[1] & h[2] & h[3] & 1, h[2] & h[3] & 1, h[3] & 1, 1)
    return u + c


@ti.func
# 取相反数
def neg256(u) -> F256:
    u = u ^ F256_ALL
    return inc256(u)


@ti.func
# return u<v
def uint64_less(u, v):
    u0: ti.uint64 = u[0]
    u1: ti.uint64 = u[1]
    u2: ti.uint64 = u[2]
    u3: ti.uint64 = u[3]
    v0: ti.uint64 = v[0]
    v1: ti.uint64 = v[1]
    v2: ti.uint64 = v[2]
    v3: ti.uint64 = v[3]
    return F256(u0 < v0, u1 < v1, u2 < v2, u3 < v3)


@ti.func
# F256+F256
def add256(u, v) -> F256:
    s = u + v
    # h = s < u
    h = uint64_less(s, u)
    c1 = F256(h[1], h[2], h[3], h[0]) & F256(1, 1, 1, 0)
    h = s == F256_ALL
    c2 = F256((c1[1] | (c1[2] & h[2])) & h[1], c1[2] & h[2], 0, 0)
    return s + c1 + c2


@ti.func
# u<<1
def shl256(u):
    h = ti.bit_shr(u, 63) & F256(0, 1, 1, 1)
    return u << 1 | F256(h[1], h[2], h[3], h[0])


@ti.func
# 计算两个int64位乘法，返回高64位与低64位
def mul_hi_lo(op1: ti.int64, op2: ti.int64) -> _INT2:
    u1 = op1 & ti.i64(0xffffffff)
    v1 = op2 & ti.i64(0xffffffff)
    t = u1 * v1
    w3 = t & ti.i64(0xffffffff)
    k = ti.bit_shr(t, 32)
    op1 = ti.bit_shr(op1, 32)
    t = op1 * v1 + k
    k = t & ti.i64(0xffffffff)
    w1 = ti.bit_shr(t, 32)
    op2 = ti.bit_shr(op2, 32)
    t = (u1 * op2) + k
    k = ti.bit_shr(t, 32)
    hi = op1 * op2 + w1 + k
    lo = (t << 32) + w3
    return _INT2(hi, lo)


@ti.func
# return U*V
def mulfpu(u, v):
    neg = 0
    usign = ti.bit_shr(u[0], 63)
    vsign = ti.bit_shr(v[0], 63)
    if usign == 1:
        u = neg256(u)
        neg += 1
    if vsign == 1:
        v = neg256(v)
        neg += 1
    _t = mul_hi_lo(u[1], v[1])
    s = F256(u[0] * v[0], _t[0], _t[1], mul_hi_lo(u[2], v[2])[0])
    _t = mul_hi_lo(u[0], v[1])
    _t1 = mul_hi_lo(u[0], v[3])
    t1 = F256(_t[0], _t[1], _t1[0], _t1[1])
    _t = mul_hi_lo(v[0], u[1])
    _t1 = mul_hi_lo(v[0], u[3])
    t2 = F256(_t[0], _t[1], _t1[0], _t1[1])
    s = add256(s, add256(t1, t2))
    _t = mul_hi_lo(u[0], v[2])
    t1 = F256(0, _t[0], _t[1], mul_hi_lo(u[1], v[3])[0])
    _t = mul_hi_lo(v[0], u[2])
    t2 = F256(0, _t[0], _t[1], mul_hi_lo(v[1], u[3])[0])
    s = add256(s, add256(t1, t2))
    _t = mul_hi_lo(u[1], v[2])
    t1 = F256(0, 0, _t[0], _t[1])
    _t = mul_hi_lo(v[1], u[2])
    t2 = F256(0, 0, _t[0], _t[1])
    s = add256(s, add256(t1, t2))
    res = add256(s, F256(0, 0, 0, 3))
    if neg == 1:
        res = neg256(res)
    return res


@ti.func
# return U^2
def sqrfpu(u):
    usign = ti.bit_shr(u[0], 63)
    if usign == 1:
        u = neg256(u)
    _t = mul_hi_lo(u[1], u[1])
    s = F256(u[0] * u[0], _t[0], _t[1], mul_hi_lo(u[2], u[2])[0])
    _t = mul_hi_lo(u[0], u[1])
    _t1 = mul_hi_lo(u[0], u[3])
    t = F256(_t[0], _t[1], _t1[0], _t1[1])
    s = add256(s, shl256(t))
    _t = mul_hi_lo(u[0], u[2])
    t = F256(0, _t[0], _t[1], mul_hi_lo(u[1], u[3])[0])
    s = add256(s, shl256(t))
    _t = mul_hi_lo(u[1], u[2])
    t = F256(0, 0, _t[0], _t[1])
    s = add256(s, shl256(t))
    res = add256(s, F256(0, 0, 0, 3))
    return res
