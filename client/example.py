from parity_machines import PermutationMachine


if __name__ == '__main__':
    k, n, g = 2, 30, 64

    ppm1 = PermutationMachine(k, n, g)
    ppm2 = PermutationMachine(k, n, g)
    eve = PermutationMachine(k, n, g)

    count = 0

    while not (ppm1.synchronized() and ppm2.synchronized()):
        x, pi = ppm1.generate_input()

        tau1, tau2, tauEve = ppm1(x, pi), ppm2(x, pi), eve(x, pi)
        ppm1.update(tau2)
        ppm2.update(tau1)

        if tau1 == tau2 == tauEve:
            eve.update(tau1)

        count += 1

    print(f'Keys synchronized after {count} iterations:')
    print(f'PPM1: {ppm1.get_key().hex()}')
    print(f'PPM2: {ppm2.get_key().hex()}')
    print(f'Eve:  {eve.get_key().hex()}')
