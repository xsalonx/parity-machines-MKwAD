from parity_machines import PermutationMachine


if __name__ == '__main__':
    k, n, g = 2, 30, 256

    ppm1 = PermutationMachine(k, n, g)
    ppm2 = PermutationMachine(k, n, g)
    eve = PermutationMachine(k, n, g)

    count = 0

    while not (ppm1.sync_level() == 1 and ppm2.sync_level() == 1):
        x = ppm1.generate_input()

        tau1, tau2, tauEve = ppm1(x), ppm2(x), eve(x)
        ppm1.update(tau2)
        ppm2.update(tau1)

        if tau1 == tau2 == tauEve:
            eve.update(tau1)

        count += 1

    print(f'Keys synchronized after {count} iterations:')
    print(f'PPM1: {ppm1.get_key().hex()}')
    print(f'PPM2: {ppm2.get_key().hex()}')
    print(f'Eve:  {eve.get_key().hex()}')
