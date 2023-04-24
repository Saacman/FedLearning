

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
            if self.egcd(g, n**2)[0] == 1:
                break
        
        # Compute the public key (n, g) and the private key (lambda)
        self.public_key = (n, g)
        self.private_key = l
    
    def encrypt(self, m):
        # Generate a random r that is relatively prime to n
        n, g = self.public_key
        while True:
            r = random.randint(1, n-1)
            if self.egcd(r, n)[0] == 1:
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
    
    # def gcd(self, a, b):
    #     while b != 0:
    #         a, b = b, a % b
    #     return a
    
    def egcd(self, a, b):
        """
        Extended Euclidean algorith, computes the greatest common diviser and the 
        coefficients of Bezout's identity.
        Source: https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
        
        :param a: a value
        :param b: b value
        :return: greatest common divisor and coefficients of Bezout's identity
        """
        if a == 0:
            return (b, 0, 1)
        else:
            g, y, x = self.egcd(b % a, a)
            return (g, x - (b // a) * y, y)
    
    def lcm(self, a, b):
        return a * b // self.egcd(a, b)[0]
    
    def mod_inv(self, a, m):
        """
        Calculates the multiplicative inverse of a in modulo p
        Source: http://code.activestate.com/recipes/576737-inverse-modulo-p/)
        :param a: a value
        :param m: modulo
        :return: b, such that (a * b) % m = 1 
        """
        # if self.gcd(a, m) != 1:
        #     raise ValueError("a is not invertible modulo m")
        
        #x, y, u, v = 
        g, x, y = self.egcd(a, m)
        if g != 1:
            raise Exception('modular inverse does not exist')
        else:
            return x % m
        

if __name__ == '__main__':
    print("Hola")