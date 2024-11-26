def find_largest_or(a: list["int"], b: list["int"]) -> int:
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("a and b must have the same length and be non-empty.")
    offset_size = len(a) - 1
    max_or = 0
    while offset_size > 0:
        i = 0
        while True:
            left = i
            right = left + offset_size
            if right >= len(a):
                break
            a_or = 0
            b_or = 0
            for j in range(left, right + 1):
                a_or |= a[j]
                b_or |= b[j]
            max_or = max(max_or, a_or + b_or)
            i += 1
        offset_size -= 1
    return max_or


if __name__ == "__main__":
    _ = input()
    a = list(map(int, input().split(" ")))
    b = list(map(int, input().split(" ")))
    print(find_largest_or(a, b))
