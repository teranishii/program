#!/usr/bin/python
# -*- coding: utf-8 -*-

#218014042 寺西弘樹

import unittest
from sample import prime 

class Testprime(unittest.TestCase):
    def testprime(self):
        value=100
        expected=[2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]
        actual=prime(value)
        self.assertTrue(expected,actual)

if __name__ == '__main__':
    unittest.main()