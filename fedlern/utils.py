import math
import random
import pickle
import torch
import os

def gcd(a, b):
    """
    Recursive function, outputs the greatest common divisor of a and b
    """
    if a == 0:
        return b
    return gcd(b % a, a)

def gcde(a, b):
    """
    Outputs the greatest common divisor of a and b, gcd,
    and the coefficients of the Bezout's identity
    """
    if a == 0:
        return (b, 0, 1)
    else:
        gcd, x1, y1 = gcde(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return (gcd, x, y)
    
def lcm(a, b):
    """
    Outputs the least common multiple of a and b
    """
    return (a * b) // gcd(a, b)

def lfxn(x, n):
    """
    L function such that L(x) = (x-1)/n. The result
    must always be an integer.
    :return int: (x - 1) / m
    """
    y = (x - 1)/n
    assert y - int(y) == 0
    return int(y)

def powmod(base, exp, mod):
    """
    Returns (base^exp) % mod where base, exp, and mod
    are integers.

    :return int: (a ** b) % c
    """
    if base == 1:
        return 1
    else:
        return pow(base, exp, mod)


def invmod(a, m):
    """
    Calculates the multiplicative inverse of a in modulo p
    Source: http://code.activestate.com/recipes/576737-inverse-modulo-p/)
    :param a: a value
    :param m: modulo
    :return: b, such that (a * b) % m = 1 
    """
    g, x, y = gcde(a, m)
    if g != 1:
        raise Exception('modular inverse does not exist')
    else:
        return (x % m + m ) % m
    
def mulmod(a, b, c):
    """
    Return a * b mod c, where a, b, c
    are integers.

    :return int: (a * b) % c
    """
    return a * b % c

def is_prime(n: int):
    """
    Check wether the given number n is prime
    """
    # Check if the number is 2 or 3
    if n == 2 or n == 3:
        return True
    
    # Check if the number is even or less than 2
    if n <= 1 or n % 2 == 0:
        return False
    
    # Check if the number is divisible by any odd number up to the square root of n
    for i in range(3, int(math.sqrt(n))+1, 2):
        if n % i == 0:
            return False
    
    return True


def generate_large_prime(bits: int):
    # Generate a random odd number of the given size
    p = random.getrandbits(bits)
    if p % 2 == 0:
        p += 1
    
    # Check if the number is prime
    while not is_prime(p):
        p += 2
    
    return p

def primes_list(n: int):
    """
    Generate a list with the first primes up to n
    """
    out = list()
    sieve = [True] * (n+1)
    for p in range(2, n+1):
        if (sieve[p]):
            out.append(p)
            for i in range(p, n+1, p):
                sieve[i] = False
    return out


def send_pckld_bytes(sockt, msg):
    """
    Function to send & unload pickle bytes
    """
    # Dump the msg to pickle bytes
    msg_bytes = pickle.dumps(msg)
    # Get the size of the msg bytes
    size_bytes = len(msg_bytes).to_bytes(4, byteorder='big')

    # Send the size first (4bits) and the pickled bytes
    sockt.sendall(size_bytes)
    sockt.sendall(msg_bytes)


def recv_pckld_bytes(sockt):
    """
    Function to receive & unload pickle bytes
    """
    # Receive the size bytes
    size_bytes = b''
    while len(size_bytes) < 4:
        size_bytes += sockt.recv(4 - len(size_bytes))
    size = int.from_bytes(size_bytes, byteorder='big')
    # Receive msg from server
    msg_bytes = b''
    while len(msg_bytes) < size:
        msg_bytes += sockt.recv(4096)

    recvd_msg = pickle.loads(msg_bytes)
    return recvd_msg

def print_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
    os.remove('tmp.pt')

def evaluate_model(model, test_loader, device, criterion=None):

    model.eval()
    model.to(device)

    running_loss = 0
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            if criterion is not None:
                loss = criterion(outputs, labels).item()
            else:
                loss = 0

            # statistics
            running_loss += loss * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)

    return eval_loss, eval_accuracy