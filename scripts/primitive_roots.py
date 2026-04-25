from sympy import factorint


def find_prim_root(prime):
    qs = [p for p, _ in factorint(prime - 1).items()]
    g = 2
    found = False
    while not found:
        found = True
        for q in qs:
            found &= pow(g, (prime - 1) // q, prime) != 1
            if not found:
                break
        g += 1
    return g - 1


def main():
    p1 = 5937362789990400001
    g1 = find_prim_root(p1)
    print(f"p  = {p1}")
    print(f"g  = {g1}")
    print(f"g3 = {pow(g1, (p1- 1) // 3, p1)}")
    print(f"g5 = {pow(g1, (p1- 1) // 5, p1)}")
    print()
    p2 = 8122312296706867201
    g2 = find_prim_root(p2)
    print(f"p  = {p2}")
    print(f"g  = {g2}")
    print(f"g3 = {pow(g2, (p2- 1) // 3, p2)}")
    print(f"g5 = {pow(g2, (p2- 1) // 5, p2)}")
    print()
    p3 = 7552325468867788801
    g3 = find_prim_root(p3)
    print(f"p  = {p3}")
    print(f"g  = {g3}")
    print(f"g3 = {pow(g3, (p3- 1) // 3, p3)}")
    print(f"g5 = {pow(g3, (p3- 1) // 5, p3)}")


if __name__ == "__main__":
    main()
