"""
The DATA field, composed of SERVICE, PSDU, tail, and pad parts, shall be scrambled with a length-127
PPDU-synchronous scrambler. The octets of the PSDU are placed in the transmit serial bit stream, bit 0 first
and bit 7 last.
"""


from wifi.bits import bits


def apply(data: bits) -> bits:
    output = bits('')
    shr = bits('1011101')
    for bit in data:
        feedback = shr[3] ^ shr[6]
        shr = feedback + shr[:-1]
        output += bit ^ feedback
    return output


def undo(scrambled: bits) -> bits:
    return apply(scrambled)


def test_i152():
    """
    IEEE Std 802.11-2016: I.1.4.2 Coding the SIGNAL field bits
    """

    # Table I-13—The DATA bits before scrambling
    input = bits('00000000000000000010000001000000000000000111010000000000000001100001000010110011111011000110010'
                 '100000000000001000110101110000000001111001000111100000000000001100001000010110101110111001111010'
                 '100000000000000000101001011110110100111100011010000000100010001100100111010010110111001100001011'
                 '000101110000001001100111000001110100001100100111011010110000001001111011001100110000001000010011'
                 '010010110011011101001011001110110100101100010111010011110001101000101000000100010100001101010111'
                 '0111001100001011000101110101001100100111000000100111101100110011000000100101000100011011010011110'
                 '1100111010010110101011101011011000110100010100000110001010010110010011101010011010110100100101100'
                 '1110110110011101001011001001110101001100010011000000100111011101010011000000100001011100100111010'
                 '1001101000011011100110110011001000010001101101000000000000000000000000000000000000000000000000')

    # Table I-15—The DATA bits after scrambling ( tail bits set to 0! )
    expect = bits('011011000001100110001001100011110110100000100001111101001010010101100001010011111101011110101'
                  '110001001000000110011110011001110101110010010111100010100111001100011000000000111100011010110'
                  '110011111000111111100000100101011000001101011000100101001101010011001111111110111100000100000'
                  '100101011100011110101001100011100100000110100000110111110001110010010100001100110010001000110'
                  '011011001101111101101010001111011000000011011101010010000001001110110010111111011111110000110'
                  '101100011110111110001100101001011101011011100001000111110011110011010101001000010000001111111'
                  '010111110010101001110100010101010100010010000001000111010011011001111010010011101111001101100'
                  '100111000110101111011011111000111000000000010001000001001100110100001011111011000101000100111'
                  '000101110011100100010101101000001110110010010101000101101001000100010000000000001101110001111'
                  '111000011101111001011001001')

    output = apply(input)

    # test against standard -> restore the tail bits to 0
    tail_zeroed = list(output)
    tail_zeroed[816:822] = ['0', '0', '0', '0', '0', '0']
    tail_zeroed = ''.join(tail_zeroed)
    assert tail_zeroed == expect

    # test reverse
    rev = undo(output)
    assert rev == input

