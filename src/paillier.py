

import random
from utils import *
import math

class Paillier:
    def __init__(self, key_length):
        # Generate two large prime numbers, p and q
        p = generate_large_prime(key_length // 2)
        q = generate_large_prime(key_length // 2)
        assert p != q
        assert is_prime(p)
        assert is_prime(q)

        # Compute n = p * q and lambda = lcm(p-1, q-1)
        n = p * q
        phi = (p - 1) * (q - 1)
        assert gcd(n, phi) == 1

        # Key Generation
        g = random.randint(1, n**2)
        #lmda = lcm(p-1, q-1)
        #print(int(lmda))
        lmda = math.lcm(p-1, q-1)
        self.mu = powmod(lfxn(powmod(g, lmda, n**2), n), -1, n)
        
        
        # Compute the public key (n, g) and the private key (lambda)
        self.public_key = (n, g)
        self.private_key = lmda
    
    def encrypt(self, m):
        # Generate a random r that is relatively prime to n
        n, g = self.public_key
        while True:
            r = random.randint(1, n-1)
            if gcd(r, n) == 1:
                break
        
        # Compute the ciphertext c = g^m * r^n mod n^2
        c = (powmod(g, m, n**2) * powmod(r, n, n**2)) % (n**2)
        return c
    
    def decrypt(self, c):
        # # Compute the plaintext m = L(c^lambda mod n^2) / L(g^lambda mod n^2) mod n
        # n, g = self.public_key
        # lmda = self.private_key
        # m = (lfxn(powmod(c, lmda, n ** 2), n) * self.mu) % n
        # #m = (lfxn(c^lmda % n^2,n) / lfxn(g^lmda % n^2,n)) % n

        # return m
    
        # Compute the plaintext m = L(c^lambda mod n^2) / L(g^lambda mod n^2) mod n
        n, g = self.public_key
        l = self.private_key
        
        u = pow(c, l, n**2)
        v = pow(g, l, n**2)
        x = (u - 1) // n
        y = (v - 1) // n
        
        return (x * invmod(y, n)) % n


    def e_add(self, a, b):
        """Add one encrypted integer to another"""
        n, g = self.public_key
        return (a * b) % (n*n)

    def e_add_const(self, a, k):
        """Add constant k to an encrypted integer"""
        n, g = self.public_key
        return a * powmod(g, k, n) % n

    def e_mul_const(self, a, k):
        """Multiplies an ancrypted integer by a constant k"""
        n, g = self.public_key
        return powmod(a, k, n)
    
if __name__ == '__main__':
    crypto = Paillier(75)

    ciphertexts = [crypto.encrypt(x) for x in range(100)]
    print(ciphertexts)

    plaintexts = [crypto.decrypt(c) for c in ciphertexts]
    print(plaintexts)
    print(crypto.decrypt(ciphertexts[0] * ciphertexts[1]))
    print(crypto.decrypt(ciphertexts[1] * ciphertexts[3]))

    for i in range(49):
        sum= crypto.e_add(ciphertexts[i], ciphertexts[99-i])
        print(crypto.decrypt(sum))