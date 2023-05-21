import pickle
import socket
import struct
from argparse import ArgumentParser
from tqdm import tqdm

from parity_machines import *


parity_machine = {
    'permutation': (PermutationMachine, {
        'k': 2,
        'n': 30,
        'g': 256
    }), 
    'tree': (TreeParityMachine, {
        'k': 20,
        'n': 60,
        'l': 30,
        'update_rule': 'hebbian'
    })
}


def bind_socket(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', port))
    s.listen()
    conn, addr = s.accept()

    print(addr)
    return s, conn


def connect_socket(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    return s


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--host', required=True, type=str)
    args.add_argument('--port', required=True, type=int)
    args.add_argument('--type', required=True, type=str)
    args.add_argument('--first', action='store_true')
    args = args.parse_args()

    # Create parity machine
    machine, kwargs = parity_machine[args.type]
    pm = machine(**kwargs)
    x_len = len(pickle.dumps(pm.generate_input()))

    # Connect to the peer
    if args.first:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('127.0.0.1', args.port))
        sock.listen()
        s, _ = sock.accept()
    else:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((args.host, args.port))

    # Create progress bar
    bar = tqdm(total=100, ncols=100)
    count = 0

    while not pm.sync_level() == 1:
        # Generate input and permutation vectors
        if args.first:
            x = pm.generate_input()
            s.send(pickle.dumps(x))
        else:
            x = pickle.loads(s.recv(x_len))

        # Compute tau and send it to the peer
        tau = pm(x).item()
        s.send(struct.pack("<i", tau))

        # Receive tau from the peer and update the machine
        tau_peer = struct.unpack("<i", s.recv(4))
        pm.update(tau_peer)

        # Update the progress bar
        bar.n = int(pm.sync_level() * 100)
        bar.refresh()
        count += 1

    print(f'Keys synchronized after {count} iterations:')
    print(pm.get_key().hex())

    # Close the sockets
    bar.close()
    s.close()

    if args.first:
        sock.close()
