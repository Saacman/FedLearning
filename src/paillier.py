

import random
import math

class Paillier:
    def __init__(self, key_length):
        # Generate two large prime numbers, p and q
        p = self.generate_large_prime(key_length // 2)
        q = self.generate_large_prime(key_length // 2)
        
        # Compute n = p * q and lambda = lcm(p-1, q-1)
        n = p * q
        phi = (p-1) * (q-1)
        l = self.lcm(p-1, q-1)
        
        # Generate a random number g that is relatively prime to n^2
        while True:
            g = random.randint(1, n**2)
            if self.gcd(g, n**2) == 1:
                break
        
        # Compute the public key (n, g) and the private key (lambda)
        self.public_key = (n, g)
        self.private_key = l
    
    def encrypt(self, m):
        # Generate a random r that is relatively prime to n
        n, g = self.public_key
        while True:
            r = random.randint(1, n-1)
            if self.gcd(r, n) == 1:
                break
        
        # Compute the ciphertext c = g^m * r^n mod n^2
        c = (pow(g, m, n**2) * pow(r, n, n**2)) % (n**2)
        return c
    
    def decrypt(self, c):
        # Compute the plaintext m = L(c^lambda mod n^2) / L(g^lambda mod n^2) mod n
        n, g = self.public_key
        l = self.private_key
        
        u = pow(c, l, n**2)
        v = pow(g, l, n**2)
        x = (u - 1) // n
        y = (v - 1) // n
        
        return (x * self.mod_inv(y, n)) % n
    
    def generate_large_prime(self, bits):
        # Generate a random odd number of the given size
        p = random.getrandbits(bits)
        if p % 2 == 0:
            p += 1
        
        # Check if the number is prime
        while not self.is_prime(p):
            p += 2
        
        return p
    
    def is_prime(self, n):
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
    
    def gcd(self, a, b):
        while b != 0:
            a, b = b, a % b
        return a
    
    def lcm(self, a, b):
        return a * b // self.gcd(a, b)
    
    def mod_inv(self, a, m):
        # Compute the modular inverse of a mod m using the extended Euclidean algorithm
        if self.gcd(a, m) != 1:
            raise ValueError("a is not invertible modulo m")
        
        x, y, u, v = 

class PaillierE:

    def __init__(self, p=None, q=None, n=None, g=None, l=None, mu=None):
        if p and q:
            self.p = p
            self.q = q
            self.n = p * q
            self.g = self.n + 1
            self.l = (self.p - 1) * (self.q - 1)
            self.mu = self._calculate_mu()
        elif n and g:
            self.n = n
            self.g = g
            self.l = None
            self.mu = None
        else:
            raise ValueError("Invalid parameters provided.")

    def _calculate_mu(self):
        return mod_inverse(self.l, self.n)

    def _generate_keys(self, key_length):
        self.p, self.q = generate_prime_pair(key_length)
        self.n = self.p * self.q
        self.g = self.n + 1
        self.l = (self.p - 1) * (self.q - 1)
        self.mu = self._calculate_mu()

    def encrypt(self, plaintext):
        if plaintext < 0 or plaintext >= self.n:
            raise ValueError("Plaintext must be between 0 and n-1")
        r = random.randint(1, self.n)
        c = pow(self.g, plaintext, self.n ** 2) * pow(r, self.n, self.n ** 2) % (self.n ** 2)
        return c

    def decrypt(self, ciphertext):
        if not self.l or not self.mu:
            raise ValueError("Cannot decrypt without private key")
        c = pow(ciphertext, self.l, self.n ** 2)
        plaintext = ((c - 1) // self.n * self.mu) % self.n
        return plaintext