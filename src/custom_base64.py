"""
A custom implementation of base64.

Uses:
0 - 9 (10)
a - z (26)
A - Z (26)
~     (1)
_     (1)

"""

# Lookup variables used for conversion
_int_to_char_code: list[int] = []
_char_code_to_int: dict[int, int] = {}
_size = 64


def fill_lookup_variables():
    for char_code in range(ord("0"), ord("9") + 1):
        _char_code_to_int[char_code] = len(_int_to_char_code)
        _int_to_char_code.append(char_code)

    for char_code in range(ord("a"), ord("z") + 1):
        _char_code_to_int[char_code] = len(_int_to_char_code)
        _int_to_char_code.append(char_code)

    for char_code in range(ord("A"), ord("Z") + 1):
        _char_code_to_int[char_code] = len(_int_to_char_code)
        _int_to_char_code.append(char_code)

    for char_code in (ord(ch) for ch in "~_"):
        _char_code_to_int[char_code] = len(_int_to_char_code)
        _int_to_char_code.append(char_code)


fill_lookup_variables()
assert len(_int_to_char_code) == _size
assert len(_char_code_to_int) == _size


def encode_num2bytes(num: int, signed: bool = True) -> bytes:
    b64_digits: list[int] = []

    x = abs(num)
    if x == 0:
        b64_digits.append(0)

    while x > 0:
        rem = x % _size
        b64_digits.append(rem)
        x = x // _size

    b64_chars = [_int_to_char_code[digit] for digit in b64_digits]
    if signed:
        sign = b"+"[0]
        if num < 0:
            sign = b"-"[0]
        b64_chars.append(sign)

    ret = bytes(b64_chars[::-1])
    return ret


def decode_bytes2num(data: bytes) -> int:
    sign = None
    if data[0] == b"+"[0]:
        sign = 1
    elif data[0] == b"-"[0]:
        sign = -1

    if sign is not None:
        data = data[1:]

    val = 0
    for ch in data:
        val = _size * val + _char_code_to_int[ch]

    if sign is not None:
        val *= sign

    return val


def _dev_encode_decode():
    for num in range(1000):
        x1 = encode_num2bytes(num)
        x2 = encode_num2bytes(-num)
        if num != 0:
            assert x1.replace(b"+", b"-") == x2, f"Error at {num}"

        n1 = decode_bytes2num(x1)
        n2 = decode_bytes2num(x2)

        assert n1 == num
        assert n2 == -num

        y1 = encode_num2bytes(num, signed=False)
        m1 = decode_bytes2num(y1)
        assert m1 == num

        print(num, x1.decode(), x2.decode(), y1.decode())
