from textwrap import wrap
import numpy as np

# config
k = 7
states = 2 ** (k - 1)
g0 = 0x133
g1 = 0x171


# generate state transition LUTs
def xor_reduce_poly(data, poly):
    """ XOR reduces bits that are selected by the 'poly' """
    return int(bin(data & poly).count('1') & 1)


# input bit is considered an additional state (MSb) in this LUT, thus it has (states * 2) states
output_lut = [(xor_reduce_poly(x, g0) << 1) | xor_reduce_poly(x, g1) for x in range(2 ** k)]

# def calc_error(actual, expected):
#     error = 0
#     if actual[0] != expected[0]:
#         error += 1
#
#     if actual[1] != expected[1]:
#         error += 1
#
#     return error
#
#
# def conv_core(state: int, bit: int) -> str:
#     state = bin(state)[2:].zfill(6)
#     out_a = int(bit) ^ int(state[1]) ^ int(state[2]) ^ int(state[4]) ^ int(state[5])
#     out_b = int(bit) ^ int(state[0]) ^ int(state[1]) ^ int(state[2]) ^ int(state[5])
#     output = str(out_a) + str(out_b)
#     return output
#
#
# def viterbi(rx):
#     def outp(state: str, bit: str):
#         out_a = int(bit) ^ int(state[0])
#         out_b = int(bit) ^ int(state[0]) ^ int(state[1])
#         output = str(out_b) + str(out_a)
#         return output
#
#     def butterfly(state, expected, scores):
#         parent1 = (state << 1) % 4
#         parent2 = (parent1 + 1) % 4
#
#         state_bits = 2
#         bit = (state << 1) >> state_bits
#         out1 = outp(bin(parent1)[2:].zfill(state_bits), str(bit))
#         error1 = calc_error(out1, expected)
#
#         out2 = outp(bin(parent2)[2:].zfill(state_bits), str(bit))
#         error2 = calc_error(out2, expected)
#
#         fe1 = scores[parent1][0] + error1
#         fe2 = scores[parent2][0] + error2
#
#         new_bit = '1' if state >= 2 else '0'
#         if fe1 < fe2:
#             return fe1, scores[parent1][1] + new_bit
#         else:
#             return fe2, scores[parent2][1] + new_bit
#
#     scores = [(0, ''), (1000, ''), (1000, ''), (1000, '')]
#     for expect in wrap(rx, 2):
#         scores = [butterfly(i, expect, scores) for i in range(4)]
#
#     print(scores)
#     min_score = np.argmin([x[0] for x in scores])
#     bits = scores[min_score][1]
#     return bits
#
#
# s = '111011000110'
# print(viterbi(s))

# # propagate all states
#
#
# def f(shr, bit):
#     out_a = int(bit) ^ int(shr[1]) ^ int(shr[2]) ^ int(shr[4]) ^ int(shr[5])
#     out_b = int(bit) ^ int(shr[0]) ^ int(shr[1]) ^ int(shr[2]) ^ int(shr[5])
#     output = str(out_a) + str(out_b)
#     shr = bit + shr[:-1]  # advance the shift register
#     return shr, output
#
#
# class Node:
#     instances = {}
#
#     def __init__(self, state):
#         self.state = state
#         self.path_in = []
#         self.path_out = []
#         Node.instances[self.state] = self
#
#
# @dataclass
# class Path:
#     start: Node
#     end: Node
#     bit: str
#     output: str
#
#
# def populate_outputs_recursive(base):
#     def populate(bit):
#         shr, output = f(base.state, bit)
#         if shr in Node.instances:
#             new_node = Node.instances[shr]
#         else:
#             new_node = Node(shr)
#
#         path = Path(base, new_node, bit, output)
#         base.path_out += [path]
#         new_node.path_in += [path]
#
#         if not new_node.path_out:
#             populate_outputs_recursive(new_node)
#
#     populate('0')
#     populate('1')
# #
# # base = Node('000000')
# # populate_outputs_recursive(base)
# # pass
