from parity_machines import PermutationMachine


if __name__ == '__main__':
    ppm1 = PermutationMachine(k=2, n=3)
    ppm2 = PermutationMachine(k=2, n=3)

    while not (ppm1.synchronized() and ppm2.synchronized()):
        x = ppm1.generate_input()

        tau1, tau2 = ppm1(x), ppm2(x)
        ppm1.update(tau2)
        ppm2.update(tau1)

    print(ppm1.get_key().hex())
    print(ppm2.get_key().hex())
    print()
