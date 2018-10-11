
def convolutional_encoder(data):
    output = ''
    shr = '000000'
    for bit in data:
        out_a = int(bit) ^ int(shr[1]) ^ int(shr[2]) ^ int(shr[4]) ^ int(shr[5])
        out_b = int(bit) ^ int(shr[0]) ^ int(shr[1]) ^ int(shr[2]) ^ int(shr[5])
        output += str(out_a) + str(out_b)
        shr = bit + shr[:-1] # advance the shift register

    return output


def test_signal():
    """
    IEEE Std 802.11-2016: I.1.4.2 Coding the SIGNAL field bits
    """

    # IEEE Std 802.11-2016: Table I-7—Bit assignment for SIGNAL field
    input = '101100010011000000000000'

    # IEEE Std 802.11-2016: Table I-8—SIGNAL field bits after encoding
    expected = '110100011010000100000010001111100111000000000000'
    output = convolutional_encoder(input)
    assert output == expected
