#!/usr/bin/env python3

"""
CSF and determinants (inefficient code).

Reference: [MEST] Helgaker, T., Jorgensen, P., & Olsen, J. (2014).
    /Molecular electronic-structure theory./ John Wiley & Sons.
"""

import numpy as np


def next_permutation(a):
    """Generate the lexicographically next permutation inplace"""
    # Find the largest index i such that a[i] < a[i + 1]. If no such
    # index exists, the permutation is the last permutation
    for i in reversed(range(len(a) - 1)):
        if a[i] < a[i + 1]:
            break  # found
    else:  # no break: not found
        return False  # no next permutation
    # Find the largest index j greater than i such that a[i] < a[j]
    j = next(j for j in reversed(range(i + 1, len(a))) if a[i] < a[j])
    # Swap the value of a[i] with that of a[j]
    a[i], a[j] = a[j], a[i]
    # Reverse sequence from a[i + 1] up to and including the final element a[n]
    a[i + 1:] = reversed(a[i + 1:])
    return True


# determinant
class DET:
    def __init__(self, string):
        # a string of "0/a/b/2"
        # len(string) == n_sites == number of spatial orbitals
        self.string = string
        self.n_sites = len(string)
        self.n_unpaired = string.count('a') + string.count('b')
        self.n_elec = self.n_unpaired + string.count('2') * 2
        self.two_sz = string.count('a') - string.count('b')

    def __repr__(self):
        return ''.join(self.string)

    def __eq__(self, other):
        return self.string == other.string

    # generate all possible determinant for n_unpaired electrons
    @staticmethod
    def generate_det_strings(n_unpaired):
        if n_unpaired == 0:
            return [""]
        else:
            a = DET.generate_det_strings(n_unpaired - 1)
            return [ia + "a" for ia in a] + [ia + "b" for ia in a]

    def get_occ_string(self):
        mp = {'0': '0', 'a': '1', 'b': '1', '2': '2'}
        return [mp[x] for x in self.string]


# configuration state function
class CSF:
    def __init__(self, string):
        # a string of "0/+/-/2"
        # len(string) == n_sites == number of spatial orbitals
        self.string = string
        self.n_sites = len(string)
        self.n_unpaired = string.count('+') + string.count('-')
        self.n_elec = self.n_unpaired + string.count('2') * 2
        self.two_s = string.count('+') - string.count('-')

    def __repr__(self):
        return ''.join(self.string)

    def __eq__(self, other):
        return self.string == other.string

    # generate all possible S+/S- operator strings for n_unpaired electrons
    # for a target multiplicity
    # MEST Fig. 2.1
    @staticmethod
    def generate_operator_strings(n_unpaired, multi):
        if n_unpaired % 2 == multi % 2 or multi > n_unpaired + 1 or multi < 1:
            return []
        if n_unpaired == 0:
            return [""] if multi == 1 else []
        a = CSF.generate_operator_strings(n_unpaired - 1, multi - 1)
        b = CSF.generate_operator_strings(n_unpaired - 1, multi + 1)
        return [ia + "+" for ia in a] + [ib + "-" for ib in b]

    # generate all possible CSF for n_elec electrons for a target multiplicity
    # in n_sites spatial orbitals
    @staticmethod
    def generate_csfs(n_sites, n_elec, multi):
        csfs = []
        if n_elec % 2 == multi % 2:
            return []
        for n_unpaired in range(n_elec % 2, n_sites + 1, 2):
            n_doubly_occupied = (n_elec - n_unpaired) // 2
            n_empty = n_sites - n_unpaired - n_doubly_occupied
            occ_string = list("0" * n_empty + "1" *
                              n_unpaired + "2" * n_doubly_occupied)
            unpaired_strings = CSF.generate_operator_strings(n_unpaired, multi)
            unpaired_strings = [np.array(list(x), dtype=str)
                                for x in unpaired_strings]
            found = True
            while found:
                occ_base = np.array(occ_string, dtype=str)
                for s in unpaired_strings:
                    occ_tmp = occ_base.copy()
                    occ_tmp[occ_tmp == '1'] = s
                    csfs.append(CSF(list(occ_tmp)))
                found = next_permutation(occ_string)
        return csfs

    def get_occ_string(self):
        mp = {'0': '0', '+': '1', '-': '1', '2': '2'}
        return [mp[x] for x in self.string]

    # return coupling coefs between this CSF and a DET
    # MEST Eq. (2.6.10) (2.6.5) (2.6.6)
    def get_det_coupling_coef(self, det):
        if self.get_occ_string() != det.get_occ_string():
            return 0.0
        elif det.two_sz < -self.two_s or det.two_sz > self.two_s:
            return 0.0
        if self.n_unpaired == 0:
            return 1.0

        def fac(s, m, f, g):
            if f == '+' and g == 'a':
                return np.sqrt((s + m) / (s + s))
            elif f == '+' and g == 'b':
                return np.sqrt((s - m) / (s + s))
            elif f == '-' and g == 'a':
                return -np.sqrt((s + 1 - m) / (s + s + 2))
            elif f == '-' and g == 'b':
                return np.sqrt((s + 1 + m) / (s + s + 2))

        def coup(k, pc, pd, c, d, r):
            if k == 0:
                return r
            if c[0] == '0' or c[0] == '2':
                return coup(k, pc, pd, c[1:], d[1:], r)
            else:
                jc = pc + {'+': 0.5, '-': -0.5}[c[0]]
                jd = pd + {'a': 0.5, 'b': -0.5}[d[0]]
                return coup(k - 1, jc, jd, c[1:], d[1:], fac(jc, jd, c[0], d[0]) * r)

        return coup(self.n_unpaired, 0, 0, self.string, det.string, 1.0)

    # return all determinants and coefficients in this CSF
    def get_determinants(self):
        dets = []
        occ_string = self.get_occ_string()
        unpaired_strings = DET.generate_det_strings(self.n_unpaired)
        unpaired_strings = [np.array(list(x), dtype=str)
                            for x in unpaired_strings]
        occ_base = np.array(occ_string, dtype=str)
        for s in unpaired_strings:
            occ_tmp = occ_base.copy()
            occ_tmp[occ_tmp == '1'] = s
            dets.append(DET(list(occ_tmp)))
        return [(det, self.get_det_coupling_coef(det)) for det in dets
                if abs(self.get_det_coupling_coef(det)) > 1E-10]


if __name__ == "__main__":
    n = 5
    # all CSFs for n spatial sites, n electrons with S = 3/2
    csfs_left = CSF.generate_csfs(n_sites=n, n_elec=n, multi=4)

    # all CSFs for n * 2 spatial sites, n * 2 electrons with S = 0
    csfs_singlets = CSF.generate_csfs(n_sites=n * 2, n_elec=n * 2, multi=1)

    # check whether left half of csf is in csfs_left
    singlets = []
    for csf in csfs_singlets:
        if CSF(csf.string[:n]) in csfs_left:
            singlets.append(csf)

    # print all singlets obtained from coupling S = 3/2 and S = 3/2
    print(singlets)

    # print determinants coupled to the first CSF
    coefs = singlets[0].get_determinants()
    print("%r = %s" %
          (singlets[0], ' + '.join(['(%6.3f) %s' % (f, d) for d, f in coefs])))
    print("norm = %f" % np.linalg.norm([c[1] for c in coefs]))
