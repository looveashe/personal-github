#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2026/7/25 23:05
# @Author  : leopold
# @File    : test.py
# @Software: PyCharm
import sys
sys.stdout.reconfigure(line_buffering=True)

def fibonacci(n):
    """
    Generate the first n numbers of the Fibonacci sequence.
    
    Args:
        n (int): Number of Fibonacci numbers to generate.
        
    Returns:
        list: A list containing the first n Fibonacci numbers.
    """
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib


def multiplication_table():
    """
    Print the multiplication table (9x9) to the console.
    """
    for i in range(1, 10):
        for j in range(1, i + 1):
            print(f"{j}*{i}={i*j:2d}", end="  ")
        print()

if __name__ == "__main__":
    print(fibonacci(10))
    print(multiplication_table())